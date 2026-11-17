"""Who does the framework actually help, and why not everyone?

The pooled gain in §5.1 is a mean over 43 personalised models, and a mean can be produced by a
uniform small gain or by a few large ones. This script answers three questions in order, because
the second is only worth asking if the first says there is something to explain.

1. **Is the per-participant variation real?** Each participant's gap is averaged over 10 splits,
   and the per-split spread is large. Two checks: the gap's own t-test per participant, and the
   split-half rank correlation of the whole ranking (the parent repo's check in
   ``plots/per_person_grid.py``). Sorting 43 noisy numbers always produces winners and losers;
   the question is whether the same ones win on a different half of the splits.

2. **If it is real, what predicts it?** Per-participant correlates, then a standardised OLS, so
   that predictors which are themselves correlated (a participant's block count and their
   isolation, say) do not each get credit for the same variance.

3. **What is NOT the explanation?** Two plausible mechanisms are tested and rejected here rather
   than left as speculation: that pool-wide random already buys the target's own blocks for
   large contributors, and that participants with almost no positives in their own data cannot
   benefit from self-querying.

    python two_stage/who_benefits.py --run <run folder>
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from pool_attributes import load_attributes                                  # noqa: E402

PREDICTORS = ["knn_dist", "n_pool", "pos_rate", "isolation", "impurity", "local_purity",
              "random_self_n", "random_level", "n_pos_test"]


def split_half(W, n=400, seed=0):
    """-> mean Spearman correlation between the per-participant gap on two halves of the splits.

    0 says the ordering of participants is luck and nothing in it should be interpreted; 1 says
    a different set of splits would rank them the same way.
    """
    rng = np.random.default_rng(seed)
    cols = list(W.columns)
    out = []
    for _ in range(n):
        p = rng.permutation(cols)
        a, b = W[list(p[:len(p) // 2])].mean(axis=1), W[list(p[len(p) // 2:])].mean(axis=1)
        m = a.notna() & b.notna()
        if m.sum() > 2:
            out.append(stats.spearmanr(a[m], b[m]).statistic)
    return float(np.mean(out)) if out else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--method", default="twostage")
    ap.add_argument("--ref", default="random")
    ap.add_argument("--metric", default="macro_prauc")
    ap.add_argument("--target", default="computer_work")
    ap.add_argument("--block", type=int, default=15)
    ap.add_argument("--out", default=os.path.join(HERE, "diagnosis", "who_benefits.csv"))
    a = ap.parse_args()

    R = pd.concat([pd.read_csv(f) for f in
                   sorted(glob.glob(os.path.join(a.run, "per_round_split*.csv")))],
                  ignore_index=True)
    Q = pd.concat([pd.read_csv(f) for f in
                   sorted(glob.glob(os.path.join(a.run, "queries_split*.csv")))],
                  ignore_index=True)
    last = R[(R["round"] == R["round"].max()) & R.method.isin([a.method, a.ref])]
    w = last.pivot_table(index=["target", "split"], columns="method", values=a.metric)
    cell = (w[a.method] - w[a.ref]).rename("gap").reset_index()

    # --- 1. is the variation real? ----------------------------------------
    g = cell.groupby("target")["gap"].agg(gap="mean", sd=lambda v: v.std(ddof=1), n="size")
    g["se"] = g.sd / np.sqrt(g.n)
    g["t"] = g.gap / g.se
    g["p"] = 2 * stats.t.sf(g.t.abs(), g.n - 1)
    W = cell.pivot(index="target", columns="split", values="gap")
    rho = split_half(W)
    # Spearman-Brown: the reliability of the FULL set of splits implied by the half-split one
    rel = 2 * rho / (1 + rho) if rho == rho else np.nan
    print(f"per-participant gap ({a.method} - {a.ref}, {a.metric}), {int(g.n.iloc[0])} splits:")
    print(f"  positive point estimate      {int((g.gap > 0).sum())}/{len(g)}")
    print(f"  significantly POSITIVE       {int(((g.gap > 0) & (g.p < 0.05)).sum())}")
    print(f"  significantly NEGATIVE       {int(((g.gap < 0) & (g.p < 0.05)).sum())}")
    print(f"  indistinguishable from zero  {int((g.p >= 0.05).sum())}")
    print(f"  split-half rank correlation  {rho:+.3f}  -> reliability of the full average "
          f"{rel:.2f} (Spearman-Brown)")

    # --- 2. what predicts it? ---------------------------------------------
    rs = Q[Q.method == a.ref].groupby(["target", "split"]).is_self.sum() \
          .rename("random_self_n").reset_index()
    lvl = last[last.method == a.ref].groupby(["target", "split"])[a.metric] \
            .mean().rename("random_level").reset_index()
    npt = last[last.method == a.method].groupby(["target", "split"]).n_pos_test.mean() \
            .reset_index()
    _, P = load_attributes(a.target, sorted(cell.split.unique()), a.block, verbose=False)
    P = P.rename(columns={"owner": "target"})
    D = (cell.merge(rs, on=["target", "split"], how="left")
             .merge(lvl, on=["target", "split"]).merge(npt, on=["target", "split"])
             .merge(P[["target", "split", "n_pool", "pos_rate", "isolation", "knn_dist",
                       "impurity", "local_purity"]], on=["target", "split"]))
    per = D.groupby("target").mean(numeric_only=True)
    rows = []
    for c in PREDICTORS:
        r = stats.pearsonr(per[c], per.gap)
        rows.append(dict(predictor=c, pearson_r=r.statistic, p=r.pvalue,
                         spearman=stats.spearmanr(per[c], per.gap).statistic))
    corr = pd.DataFrame(rows).sort_values("pearson_r", key=abs, ascending=False)
    print("\nper-participant correlates of the gap (n = %d participants):" % len(per))
    print(corr.round(4).to_string(index=False))

    use = ["knn_dist", "n_pool", "pos_rate", "random_self_n"]
    X = per[use]
    X = (X - X.mean()) / X.std()
    Xd = np.column_stack([np.ones(len(X)), X.to_numpy()])
    b, *_ = np.linalg.lstsq(Xd, per.gap.to_numpy(), rcond=None)
    res = per.gap.to_numpy() - Xd @ b
    se = np.sqrt(np.sum(res ** 2) / (len(per) - Xd.shape[1])
                 * np.diag(np.linalg.inv(Xd.T @ Xd)))
    r2 = 1 - np.var(res) / np.var(per.gap)
    print("\nstandardised OLS (one SD of each predictor -> change in gap):")
    for n, bi, s in zip(["intercept"] + use, b, se):
        print(f"  {n:15s} {bi:+.4f}  (SE {s:.4f}, t = {bi / s:+.2f})")
    print(f"  R^2 = {r2:.3f}" + (f"; of the RELIABLE variance, {r2 / rel:.0%}"
                                 if rel == rel and rel > 0 else ""))

    # --- 3. what is not the explanation? ----------------------------------
    print("\nrejected mechanisms:")
    r = stats.pearsonr(per.random_self_n, per.gap)
    print(f"  'random already buys your blocks': r = {r.statistic:+.3f} (p = {r.pvalue:.3f}); "
          f"random buys {per.random_self_n.min():.1f}-{per.random_self_n.max():.1f} of 30 "
          f"from the target, so there is nothing to crowd out")
    lo, hi = per[per.pos_rate < 0.08], per[per.pos_rate >= 0.08]
    t = stats.ttest_ind(lo.gap, hi.gap, equal_var=False)
    print(f"  'too few positives of your own to query': {len(lo)} participants under 8% "
          f"positive, mean gap {lo.gap.mean():+.4f} vs {hi.gap.mean():+.4f} "
          f"(p = {t.pvalue:.3f})")

    out = per.join(g[["se", "t", "p"]])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.sort_values("gap", ascending=False).to_csv(a.out)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
