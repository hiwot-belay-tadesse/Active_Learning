"""Does personalisation improve with FEWER random slopes?

The mixed model gives every participant a random intercept plus a random slope on each of the
top `n_slopes` features. At n_slopes=5 and 43 participants that is 43 + 5*43 = 258 random-effect
parameters fitted on ~11k blocks, which is a lot of per-person structure to estimate from a few
hundred blocks each. If the model is over-parameterised, cutting the slope count should RAISE
the personalised ceiling rather than lower it.

This refits the 100%-label mixed model at several slope counts on exactly the splits a finished
run used, and compares each against that run's `global_prior` reference (the pooled model with
the same fixed-effect prior -- the floor personalisation has to beat).

    python slope_sweep.py --target computer_work --contains r10x3 --slopes 0 1 2 3 5

n_slopes=0 keeps only the random intercept: each participant gets their own baseline rate but
the same feature weights as everyone else.

Writes ``analysis/slope_sweep.csv`` (split, target, n_slopes, auroc, auprc) inside the run.
Nothing here re-runs active learning; it changes only the model, not the acquisition.
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
from utility import (feature_names, fit_mixed, frame, logit_fixed, parse_vc,  # noqa: E402
                     random_part, scores, select_slopes, vc_formulas)

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None)
    ap.add_argument("--target", default="computer_work")
    ap.add_argument("--contains", default=None)
    ap.add_argument("--slopes", nargs="+", type=int, default=[0, 1, 2, 3, 5])
    ap.add_argument("--splits", nargs="*", type=int, default=None)
    a = ap.parse_args()

    run = a.run or find_run(a.target, a.contains)
    cfg = read_config(run)
    splits = a.splits if a.splits is not None else done_splits(run)
    print(f"{run}\n  slope counts {a.slopes} on splits {splits}", flush=True)

    data, cols = load(get_target(cfg["target"]), verbose=True)
    rows, ids = [], None
    for k in splits:
        sp, ids, _ = prepare_one(data, cols, seed=k, W=cfg["block"], ids=ids)
        feats = feature_names(sp.Z.shape[1])
        df = frame(sp.Z, sp.y, sp.pid)
        ranked = select_slopes(sp.Z, sp.y, sp.pid, max(a.slopes) or 1)   # one ranking, nested sets
        for ns in a.slopes:
            idx = ranked[:ns]
            t0 = time.time()
            res = fit_mixed(df, feats, vc_formulas(idx),
                            vcp_p=cfg.get("vcp_p", 0.5), fe_p=cfg.get("fe_p", 1.0))
            vc = parse_vc(res)
            for u in sp.users:
                Zte, yte = sp.test_of(u)
                pid = sp.ids[u]
                auroc, auprc = scores(yte, logit_fixed(res, Zte) + random_part(vc, pid, Zte, idx))
                rows.append(dict(split=k, target=pid, n_slopes=ns, auroc=auroc, auprc=auprc))
            print(f"  split {k}, {ns} slopes: {time.time() - t0:.0f}s", flush=True)

    R = pd.DataFrame(rows)
    out = analysis_path(run, "slope_sweep.csv")
    R.to_csv(out, index=False)

    piv = R.pivot_table(index="target", columns="n_slopes", values="auroc")
    ref = os.path.join(run, "analysis", "reference_auroc.csv")
    base = None
    if os.path.exists(ref):
        G = pd.read_csv(ref)
        name = "global_prior" if "global_prior" in set(G.model) else "global_full"
        base = G[G.model == name].groupby("target").auroc.mean()
        print(f"\nbaseline {name}: {base.mean():.4f}")
    print(f"\npersonalised ceiling by slope count ({len(splits)} splits, {len(piv)} targets):")
    for ns in a.slopes:
        line = f"  {ns:2d} slopes  {piv[ns].mean():.4f}"
        if base is not None:
            d = (piv[ns] - base).dropna()
            line += (f"   vs pooled {d.mean():+.4f}, better for "
                     f"{int((d > 0).sum())}/{len(d)} targets")
        print(line)
    print("saved", out)


if __name__ == "__main__":
    main()
