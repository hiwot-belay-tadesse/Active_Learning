"""Reference AUROCs for a finished run: what the curves are climbing towards, and from.

Two models, both fitted on 100% of the split's training labels -- no active learning, no budget:

  mixed_full   the SAME mixed-effects model the AL loop fits, given every label in the pool.
               Per-participant random intercept and slopes, so each target is scored with its
               own personalised prediction. This is the ceiling: the best this model class can
               do on this split if labels were free.
  global_prior one pooled logistic regression for everybody -- no random effects, no
               personalisation -- carrying the SAME Gaussian prior the mixed model puts on its
               fixed effects (fe_p). This is the default global reference, because an
               unpenalised pooled fit would differ from mixed_full in TWO ways (per-person
               parameters AND regularisation of the population ones) and the contrast would
               confound them. Here the only difference left is the random effects.
  global_full  the same pooled model unpenalised, computed only with --with_unpenalised. Worth
               a look once: on COMPUTER_WORK the prior halves the coefficient norm (10.3 -> 5.3)
               for +0.005 test AUROC, so the plain fit overfits mildly.
  global_sklearn  a plain scikit-learn LogisticRegression(penalty=None) on the same pooled rows,
               always computed. Mathematically the same estimator as global_full -- it is there
               as an independent implementation, so the two agreeing is a check on the
               statsmodels path rather than new information about the data.

Together they bracket the curves. A target whose AL curve sits below `global_full` is being
hurt by personalisation on few labels; one that reaches `mixed_full` with 33 labels has nothing
left to gain from more.

Both are scored on the target's own held-out days, the same test rows the AL curves use, so the
numbers are directly comparable.

    python reference_models.py --target computer_work --contains r10x3
    python reference_models.py --run results/computer_work/<folder>

Writes ``analysis/reference_auroc.csv`` inside the run (split, target, model, auroc, auprc),
which plots/per_person_grid.py picks up automatically.
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_data import get_target, load                                   # noqa: E402
from paths import analysis_path, done_splits, find_run, read_config      # noqa: E402
from preprocess_data import prepare_one                                  # noqa: E402
import statsmodels.api as sm                                            # noqa: E402
from utility import (feature_names, fit_global, fit_mixed, frame, logit_fixed,  # noqa: E402
                     parse_vc, random_part, scores, select_slopes, vc_formulas)


def fit_global_prior(df, feats, fe_p, n):
    """Pooled GLM with a Gaussian N(0, fe_p^2) prior on every coefficient.

    statsmodels divides the log-likelihood by n in fit_regularized, so the alpha matching a
    prior precision of 1/fe_p^2 on the summed scale is 1/(n * fe_p^2). Features are already
    standardised, so one alpha for all of them is meaningful.
    """
    mod = sm.GLM.from_formula("y ~ " + " + ".join(feats), data=df, family=sm.families.Binomial())
    return mod.fit_regularized(alpha=1.0 / (n * fe_p ** 2), L1_wt=0.0)


def fit_global_sklearn(df, feats, max_iter=2000):
    """Plain pooled logistic regression with scikit-learn -> (intercept, coef).

    `penalty=None` makes it genuinely unregularised. scikit-learn's DEFAULT is L2 with C=1.0,
    which would be a third penalised model rather than the plain fit asked for, so it is set
    explicitly here. Same rows, same standardised features and same intercept as the statsmodels
    fits, so the three pooled models differ only in solver and penalty.
    """
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty=None, max_iter=max_iter)
    clf.fit(df[feats].to_numpy(float), df["y"].to_numpy(float))
    return float(clf.intercept_[0]), clf.coef_[0].astype(float)

warnings.filterwarnings("ignore")


def references_for_split(data, cols, split_id, cfg, ids=None):
    """-> (rows, ids) with one entry per (target, model) for this split."""
    sp, ids, _ = prepare_one(data, cols, seed=split_id, W=cfg["block"], ids=ids)
    feats = feature_names(sp.Z.shape[1])
    slope_idx = select_slopes(sp.Z, sp.y, sp.pid, cfg.get("n_slopes", 5))
    df = frame(sp.Z, sp.y, sp.pid)

    t0 = time.time()
    fe_p = cfg.get("fe_p", 1.0)
    g = fit_global(df, feats)
    gp = fit_global_prior(df, feats, fe_p, len(df))
    sk_b0, sk_beta = fit_global_sklearn(df, feats)
    fe_g, fe_gp = np.asarray(g.params, float), np.asarray(gp.params, float)
    print(f"  split {split_id}: global GLMs on {len(df)} rows in {time.time() - t0:.0f}s "
          f"(||beta|| {np.linalg.norm(fe_g[1:]):.1f} unpenalised vs "
          f"{np.linalg.norm(fe_gp[1:]):.1f} with the fe_p={fe_p} prior, "
          f"{np.linalg.norm(sk_beta):.1f} sklearn unpenalised)", flush=True)

    if cfg.get("_only_global"):
        res = vc = None
    else:
        t0 = time.time()
        res = fit_mixed(df, feats, vc_formulas(slope_idx),
                        vcp_p=cfg.get("vcp_p", 0.5), fe_p=fe_p)
        vc = parse_vc(res)
        print(f"  split {split_id}: mixed model ({len(slope_idx)} slopes) in "
              f"{time.time() - t0:.0f}s", flush=True)

    # train-side AUROC too: the gap between train and test is the overfitting diagnostic
    tr_global = scores(sp.y, fe_g[0] + sp.Z @ fe_g[1:])[0]
    tr_prior = scores(sp.y, fe_gp[0] + sp.Z @ fe_gp[1:])[0]
    print(f"  split {split_id}: pooled TRAIN auroc {tr_global:.4f} unpenalised, "
          f"{tr_prior:.4f} with prior", flush=True)

    rows = []
    for u in sp.users:
        Zte, yte = sp.test_of(u)
        pid = sp.ids[u]
        out = {"global_prior": fe_gp[0] + Zte @ fe_gp[1:],
               "global_sklearn": sk_b0 + Zte @ sk_beta}
        if cfg.get("_with_unpenalised"):
            out["global_full"] = fe_g[0] + Zte @ fe_g[1:]
        if res is not None:
            # personalised: population effects + THIS participant's random intercept and slopes
            out["mixed_full"] = logit_fixed(res, Zte) + random_part(vc, pid, Zte, slope_idx)
        for name, lo in out.items():
            auroc, auprc = scores(yte, lo)
            rows.append(dict(split=split_id, target=pid, model=name,
                             auroc=auroc, auprc=auprc, n_test=len(yte),
                             n_pos_test=int(np.sum(yte == 1)),
                             train_auroc=tr_global if name == "global_full" else
                             (tr_prior if name == "global_prior" else np.nan)))
    return rows, ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None)
    ap.add_argument("--target", default="computer_work")
    ap.add_argument("--contains", default=None)
    ap.add_argument("--splits", nargs="*", type=int, default=None,
                    help="default: every split the run has")
    ap.add_argument("--only_global", action="store_true",
                    help="skip the slow mixed fit; merge into an existing reference CSV")
    ap.add_argument("--with_unpenalised", action="store_true",
                    help="also fit the unpenalised pooled GLM (global_full), as a diagnostic")
    a = ap.parse_args()

    run = a.run or find_run(a.target, a.contains)
    cfg = read_config(run)
    splits = a.splits if a.splits is not None else done_splits(run)
    if not splits:
        raise SystemExit(f"no splits found in {run}")
    print(f"{run}\n  target {cfg['target']}, W={cfg['block']}, splits {splits}", flush=True)

    cfg["_only_global"] = a.only_global
    cfg["_with_unpenalised"] = a.with_unpenalised
    data, cols = load(get_target(cfg["target"]), verbose=True)
    rows, ids = [], None
    for k in splits:
        r, ids = references_for_split(data, cols, k, cfg, ids)
        rows += r
    R = pd.DataFrame(rows)
    out = analysis_path(run, "reference_auroc.csv")
    if os.path.exists(out):                       # keep models this call did not recompute
        old = pd.read_csv(out)
        old = old[~old.model.isin(set(R.model)) | ~old.split.isin(set(R.split))]
        R = pd.concat([old, R], ignore_index=True)
    R.to_csv(out, index=False)

    m = R.pivot_table(index="target", columns="model", values="auroc")
    print(f"\nmean over {len(splits)} split(s) and {len(m)} targets:")
    for name, note in (("mixed_full", "personalised, 100% of labels"),
                       ("global_prior", "pooled, same fe_p prior as the mixed model"),
                       ("global_full", "pooled, unpenalised (statsmodels)"),
                       ("global_sklearn", "pooled, plain sklearn LogisticRegression")):
        if name in m:
            print(f"  {name:13s} {m[name].mean():.4f}   ({note})")
    if "mixed_full" in m and "global_prior" in m:
        d = (m.mixed_full - m.global_prior)
        print(f"  personalisation gain vs global_prior {d.mean():+.4f}, "
              f"better for {int((d > 0).sum())}/{len(m)} targets")
    if "global_prior" in m and "global_full" in m:
        d = (m.global_prior - m.global_full)
        print(f"  prior gain on the pooled model      {d.mean():+.4f}, "
              f"better for {int((d > 0).sum())}/{len(m)} targets")
    print("saved", out)


if __name__ == "__main__":
    main()
