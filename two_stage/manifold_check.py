"""Does TW-FEG actually buy on the target's manifold? The design intent, checked directly.

The benchmark measures outcomes. This measures the *mechanism*: for every target, rank every
pooled block by the estimated density ratio $r(x) = p_T(x)/p_{\\text{pool}}(x)$ -- the same
quantity Phase 2 integrates against -- and ask what percentile of that ranking each method's
purchases sit at. Uniform sampling sits at 50 by construction. A method that is deliberately
target-weighted should sit above it; a method that maximises geometric coverage or distance has
no reason to, and a method that chases isolation should sit *below* it, because the target's
manifold is by definition where a participant's own blocks are dense.

This is worth separating from the outcome, because "the criterion does what it was designed to
do" and "doing that helps" are different claims, and the second can fail while the first holds.
Self-queries are excluded from the summary: the target's own blocks trivially sit at the top of
their own density ratio, so leaving them in would turn this into a restatement of the self-share.

    python two_stage/manifold_check.py --splits 0 1 2 --runs <run folder> [<run folder> ...]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from density_ratio import target_ratio                                      # noqa: E402
from load_data import get_target, load                                      # noqa: E402
from preprocess_data import canonical_ids, prepare_one                      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="computer_work")
    ap.add_argument("--block", type=int, default=15)
    ap.add_argument("--splits", type=int, nargs="+", default=[0])
    ap.add_argument("--runs", nargs="+", required=True, help="run folders to read queries from")
    ap.add_argument("--out", default=os.path.join(HERE, "diagnosis", "manifold_check.csv"))
    a = ap.parse_args()

    files = []
    for r in a.runs:
        files += sorted(glob.glob(os.path.join(r, "queries_split*.csv")))
    Q = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    Q = Q.drop_duplicates(["method", "split", "target", "round", "row"])
    Q = Q[Q.split.isin(a.splits)]

    t = get_target(a.target)
    data, cols = load(t, verbose=False)
    ids = canonical_ids(data)
    rows = []
    for k in a.splits:
        sp, _, _ = prepare_one(data, cols, seed=k, W=a.block, ids=ids)
        pid_of = {ids[u]: u for u in sp.users}
        Qk = Q[Q.split == k]
        for pid, g in Qk.groupby("target"):
            u = pid_of.get(pid)
            if u is None:
                continue
            r = target_ratio(sp.Z, sp.owner, u, seed=0)
            pct = pd.Series(r).rank(pct=True).to_numpy() * 100
            for m, gm in g.groupby("method"):
                sel = gm[~gm.is_self]
                rows.append(dict(split=k, target=pid, method=m,
                                 pct_all=float(np.mean(pct[gm.row.to_numpy()])),
                                 pct_nonself=float(np.mean(pct[sel.row.to_numpy()]))
                                 if len(sel) else np.nan,
                                 n=len(gm), n_nonself=len(sel)))
        print(f"split {k} done", flush=True)
    D = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    D.to_csv(a.out, index=False)
    agg = (D.groupby("method")
            .agg(pct_nonself=("pct_nonself", "mean"),
                 se=("pct_nonself", lambda v: v.std(ddof=1) / max(np.sqrt(v.notna().sum()), 1)),
                 pct_all=("pct_all", "mean"), n_targets=("target", "nunique"))
            .sort_values("pct_nonself", ascending=False))
    print("\nPercentile of the target's density ratio r(x) at which each method's purchases sit")
    print("(50 = indistinguishable from uniform; self-queries excluded from pct_nonself)")
    print(agg.round(2).to_string())
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
