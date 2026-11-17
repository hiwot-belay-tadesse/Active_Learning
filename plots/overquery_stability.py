"""Is "who gets over-queried" a property of the method, or of the split?

Every figure so far pools the 20 splits before dividing, which is an unweighted average of
them (each (target, split) contributes exactly rounds x batch purchases, so the denominators
are equal). That hides whether a source's deviation is reproducible or an accident of which
days happened to be held out.

Here the deviation is computed INSIDE each split -- its own purchases against its own pool
share, since the day partition changes the pool too -- and then summarised across splits:

    dev[m, k, s] = (queries from s in split k) / (all queries in split k)  -  poolshare[k, s]

A point is the mean over splits, the bar is +-1 SD across them, and the faint dots behind it
are the 20 individual splits. Sources are ordered by the mean, so a method whose ordering is
real shows a clean diagonal of points with bars that do not reach across their neighbours.

The number in each panel title is the split-half rank stability: Spearman between the source
ordering scored on two disjoint halves of the splits, averaged over random halvings. 1.0 means
the ordering reproduces exactly; 0 means the ordering is noise.
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402
from scipy.stats import spearmanr                                        # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_data import get_target, load                                   # noqa: E402
from paths import (RESULTS_ROOT, analysis_path, figure_path, run_folders,
                   run_glob)               # noqa: E402
from preprocess_data import canonical_ids, prepare_one                   # noqa: E402

ORDER = ["laplace", "eer", "typiclust", "coreset", "probcover", "random"]
LABEL = {"laplace": "EU"}
COL = {"laplace": "black", "typiclust": "purple", "coreset": "seagreen",
       "probcover": "dodgerblue", "random": "darkorange", "eer": "magenta"}


def split_half_rho(M, n=200, seed=0):
    """Mean Spearman between source orderings scored on two disjoint halves of the splits."""
    ks = list(M.columns)
    if len(ks) < 4:
        return np.nan
    rng, out = np.random.default_rng(seed), []
    for _ in range(n):
        p = rng.permutation(ks)
        a, b = M[p[:len(ks) // 2]].mean(axis=1), M[p[len(ks) // 2:]].mean(axis=1)
        ok = a.notna() & b.notna()
        if ok.sum() > 2:
            out.append(spearmanr(a[ok], b[ok]).statistic)
    return float(np.nanmean(out)) if out else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="computer_work")
    ap.add_argument("--block", type=int, default=15)
    ap.add_argument("--contains", default="r10x3")
    ap.add_argument("--run", default=None, help="run folder the outputs are written into")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--for_target", default=None, metavar="Mk",
                    help="restrict to ONE target's purchases, i.e. 'when the model being "
                         "built is Mk's, who supplies its labels, and how much does that "
                         "move between splits?' The denominator drops from rounds x batch x "
                         "targets to rounds x batch (30 here), so a single source's share "
                         "moves in steps of 1/30 and the bars are correspondingly wider.")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = os.path.join(RESULTS_ROOT, a.target)
    qf = run_glob(a.target, f"*{a.contains}*", "queries_split*.csv")
    if not qf:
        raise SystemExit(f"no queries_split*.csv under {root}/*{a.contains}*")
    Q = pd.concat([pd.read_csv(f) for f in qf], ignore_index=True) \
          .drop_duplicates(["method", "split", "target", "round", "row"])
    ms = a.methods or ([m for m in ORDER if m in set(Q.method)] +
                       [m for m in sorted(set(Q.method)) if m not in ORDER])
    run = a.run or os.path.dirname(qf[0])

    # Pool share per split: the day partition changes which rows are in the pool at all, so a
    # split's deviations have to be measured against that split's own pool, not a global mean.
    data, cols = load(get_target(a.target), verbose=True)
    ids = canonical_ids(data)
    P = {}
    for k in sorted(Q.split.unique()):
        sp, _, _ = prepare_one(data, cols, seed=int(k), W=a.block, ids=ids)
        n = pd.Series([sp.ids[u] for u in sp.owner]).value_counts()
        P[int(k)] = n / n.sum()
    pids = list(P[min(P)].sort_values(ascending=False).index)

    if a.for_target is not None:
        if a.for_target not in set(Q.target):
            raise SystemExit(f"{a.for_target} is not a target in this run; "
                             f"have {sorted(set(Q.target), key=lambda z: int(z[1:]))}")
        Q = Q[Q.target == a.for_target]
    DEV = {}
    for m in ms:
        q = Q[Q.method == m]
        cols_ = {}
        for k, g in q.groupby("split"):
            sh = g.owner.value_counts(normalize=True).reindex(pids).fillna(0.0)
            cols_[int(k)] = sh - P[int(k)].reindex(pids).fillna(0.0)
        DEV[m] = pd.DataFrame(cols_).reindex(pids)

    ncol = min(a.cols, len(ms))
    nrow = int(np.ceil(len(ms) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.0 * nrow), squeeze=False)
    rows = []
    for ax, m in zip(axes.ravel(), ms):
        M = DEV[m]
        mu, sd = M.mean(axis=1), M.std(axis=1, ddof=1)
        o = mu.sort_values(ascending=False).index
        x = np.arange(len(o))
        for k in M.columns:                       # the individual splits, behind the summary
            ax.scatter(x, 100 * M.loc[o, k], s=3, color=COL.get(m, "grey"), alpha=0.18, lw=0)
        ax.errorbar(x, 100 * mu[o], yerr=100 * sd[o], fmt="o", ms=3, lw=0, elinewidth=1.1,
                    color=COL.get(m, "grey"), ecolor=COL.get(m, "grey"), capsize=2, alpha=0.95)
        ax.axhline(0, color="black", lw=0.8)
        rho = split_half_rho(M)
        ax.set_title(f"{LABEL.get(m, m)}   split-half rank stability "
                     f"{'n/a' if not np.isfinite(rho) else f'{rho:+.2f}'}",
                     fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(o, rotation=90, fontsize=5)
        if a.for_target is not None and a.for_target in list(o):
            tick = ax.get_xticklabels()[list(o).index(a.for_target)]
            tick.set_color("crimson")
            tick.set_fontweight("bold")
        ax.set_ylabel("query share $-$ pool share (pp)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        rows.append(dict(method=m, rho=rho, mean_abs_dev=float(mu.abs().mean()),
                         mean_sd_across_splits=float(sd.mean()),
                         sd_over_spread=float(sd.mean() / (mu.max() - mu.min()))))
    for ax in axes.ravel()[len(ms):]:
        ax.axis("off")
    scope = ("across all targets" if a.for_target is None
             else f"when the target is {a.for_target} (its own row marked in red)")
    fig.suptitle(f"{a.target}: who each method over-queries {scope}, measured inside each of "
                 f"the {Q.split.nunique()} splits. Point = mean, bar = +-1 SD across splits, "
                 f"faint dots = the individual splits.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = figure_path(run, a.out or ("overquery_stability.png" if a.for_target is None
                                     else f"overquery_stability_{a.for_target}.png"))
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    S = pd.DataFrame(rows).set_index("method")
    print(S.round(4).to_string())
    S.to_csv(analysis_path(run, "overquery_stability"
                           + ("" if a.for_target is None else f"_{a.for_target}") + ".csv"))
    print("saved", out)


if __name__ == "__main__":
    main()
