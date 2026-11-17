"""What does a 33-label mixed-effects fit actually identify? The variance components, measured.

The framework's premises are quantitative, and two of them are checkable directly on a fitted
model rather than inferred from learning curves:

1. **Phase 1's premise** is that the target's random effects $\\mathbf{b}_T$ are worth buying
   outright: they must therefore actually move away from zero within the budget. If the fitted
   random-effect SDs collapse towards their prior mode and $\\hat{\\mathbf{b}}_T \\approx
   \\mathbf{0}$, then self-querying cannot be helping *through* $\\mathbf{b}_T$, and whatever
   makes it work is something else -- covariate shift between the target's blocks and everyone
   else's, for instance, which is a different mechanism with different implications.
2. **The singleton-attenuation refinement** discounts a stranger's first label by
   $1/(1 + w\\,\\mathbf{z}'\\mathbf{D}\\mathbf{z})$. Whether that reorders anything depends entirely
   on the size and the *spread* of $\\mathbf{z}'\\mathbf{D}\\mathbf{z}$ across candidates: a factor
   that is 0.97 for every candidate is algebraically correct and practically inert.

This script reports both, at several budgets, for a sample of targets: the fitted random-effect
SDs, the magnitude of the target's own posterior random effects against the fixed-effect
contribution, and the distribution of the attenuation factor.

    python two_stage/vc_diagnostic.py --targets 6 --split 0
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from acquisitions import singleton_attenuation, target_block                 # noqa: E402
from laplace_variance import _blocks, laplace_cov                           # noqa: E402
from load_data import get_target, load                                      # noqa: E402
from preprocess_data import canonical_ids, prepare_one                      # noqa: E402
from utility import (MixedEffectsClassifier, attach_person, logit_fixed, parse_vc,  # noqa: E402
                     random_part, seed_set, select_slopes)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="computer_work")
    ap.add_argument("--block", type=int, default=15)
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--targets", type=int, default=6, help="participants to sample")
    ap.add_argument("--budgets", type=int, nargs="+", default=[3, 9, 18, 33])
    ap.add_argument("--n_slopes", type=int, default=5)
    ap.add_argument("--self_frac", type=float, default=1.0,
                    help="fraction of the budget drawn from the target's own blocks (1.0 "
                         "imitates `calib`, 0.2 imitates a pool-wide acquisition)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "diagnosis", "variance_components.csv"))
    a = ap.parse_args()

    t = get_target(a.target)
    data, cols = load(t, verbose=False)
    ids = canonical_ids(data)
    sp, ids, (tr_day, _) = prepare_one(data, cols, seed=a.split, W=a.block, ids=ids)
    slope_idx = select_slopes(sp.Z, sp.y, sp.pid, a.n_slopes, cols)
    Xclf = attach_person(sp.Z, sp.owner, sp.users)
    rows = []
    rng = np.random.default_rng(a.seed)
    chosen = list(rng.choice(sp.users, min(a.targets, len(sp.users)), replace=False))
    for u in chosen:
        pid = ids[u]
        own = sp.rows_of(u)
        loc, _ = seed_set(sp.y[own], tr_day[u], np.random.default_rng([a.seed, a.split, 11]),
                          3, 1)
        labelled = list(own[loc])
        for B in a.budgets:
            need = B - len(labelled)
            if need > 0:
                n_self = int(round(a.self_frac * need))
                pool_self = np.setdiff1d(own, labelled)
                pool_other = np.setdiff1d(np.arange(len(sp.y)), np.r_[own, labelled])
                take = list(rng.choice(pool_self, min(n_self, len(pool_self)), replace=False))
                take += list(rng.choice(pool_other, max(need - len(take), 0), replace=False))
                labelled += take
            y_al = np.full(len(sp.y), np.nan)
            y_al[labelled] = sp.y[labelled]
            clf = MixedEffectsClassifier(persons=sp.users, slope_idx=slope_idx,
                                         classes=[0.0, 1.0], missing_label=np.nan)
            clf.fit(Xclf, y_al)
            if clf.fit_failed_:
                continue
            res = clf.res_
            _, _, k_vc, sd = _blocks(res)
            vc = parse_vc(res)
            Zte, yte = sp.test_of(u)
            fixed = logit_fixed(res, Zte)
            rand = random_part(vc, u, Zte, slope_idx)
            bidx = target_block(res, u)
            Sig = laplace_cov(res)
            w = np.clip(clf.predict_proba(Xclf)[:, 1] * (1 - clf.predict_proba(Xclf)[:, 1]),
                        1e-12, None)
            persons = list(clf.persons)
            who = [persons[int(c)] for c in Xclf[:, -1]]
            w_eff = singleton_attenuation(sp.Z, who, clf, res, w)
            fac = w_eff / w
            known = np.array([p in {p2 for _c, p2 in
                                    __import__("laplace_variance").vc_spec(res)} for p in who])
            rows.append(dict(
                target=pid, budget=B, n_people_labelled=len(set(np.asarray(who)[labelled])),
                sd_min=float(np.min(sd)) if k_vc else np.nan,
                sd_median=float(np.median(sd)) if k_vc else np.nan,
                sd_max=float(np.max(sd)) if k_vc else np.nan,
                b_T_absmax=float(np.max(np.abs([vc.get(("intercept", u), 0.0)]
                                               + [vc.get((f"f{k}", u), 0.0)
                                                  for k in slope_idx]))),
                sd_random_part=float(np.std(rand)), sd_fixed_part=float(np.std(fixed)),
                rand_over_fixed=float(np.std(rand) / (np.std(fixed) + 1e-12)),
                post_sd_bT=float(np.sqrt(np.mean(np.diag(Sig)[bidx]))) if len(bidx) else np.nan,
                atten_mean_new=float(fac[~known].mean()) if (~known).any() else np.nan,
                atten_sd_new=float(fac[~known].std()) if (~known).any() else np.nan,
                atten_min_new=float(fac[~known].min()) if (~known).any() else np.nan))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print(df.groupby("budget").mean(numeric_only=True).round(4).to_string())
    print(f"\n-> {a.out}")
    print("\nReading: sd_* are the fitted random-effect SDs; rand_over_fixed is how much of the "
          "\ntarget's predictive spread the random effects supply; atten_* is the singleton "
          "\nattenuation factor on a stranger's first label (1.0 = the refinement does nothing).")


if __name__ == "__main__":
    main()
