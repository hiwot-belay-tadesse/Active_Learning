"""One figure per acquisition: the learning curve above, who it queried below, same x-axis.

Two stacked panels sharing the round axis, so a bend in the AUROC curve can be read against
what the method was buying at that moment:

  1. mean test AUROC over targets, +-1 SE, with random drawn behind it for reference
  2. source x round heatmap of `query share - pool share`, aggregated over targets and splits.
     Red = that participant supplied more of the round's labels than their size in the pool
     warrants, blue = less. Rows are sorted by that method's own marginal deviation, so the
     participants it OVER-queries sit at the top and the ones it UNDER-queries at the bottom,
     and the order differs from method to method by design.

The colour scale is per-figure and TWO-SLOPED: `query share - pool share` cannot fall below
`-pool_share`, because you cannot buy negative labels, so the negative and positive ranges are
not the same size and a symmetric scale would hide one of them. Each side gets its own half of
the map; colour distance is therefore not comparable across zero, nor across methods.

    python plots/al_and_query_share.py                  # one png per method
    python plots/al_and_query_share.py --methods laplace
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_data import get_target, load                                   # noqa: E402
from paths import RESULTS_ROOT, figure_path, run_folders, run_glob                              # noqa: E402
from preprocess_data import canonical_ids, prepare_one                   # noqa: E402

ORDER = ["laplace", "eer", "typiclust", "coreset", "probcover", "random"]
LABEL = {"laplace": "EU"}
COL = {"laplace": "black", "eer": "magenta", "typiclust": "purple",
       "coreset": "seagreen", "probcover": "dodgerblue", "random": "darkorange"}


def _drop(Q, rules):
    """Remove `method` rows that came from a folder matching `folder`, for each FOLDER:METHOD."""
    for rule in rules:
        folder, _, method = rule.partition(":")
        bad = Q._src.str.contains(folder, regex=False) & (Q.method == method)
        if bad.any():
            print(f"  dropped {int(bad.sum())} {method} rows from folders matching {folder!r}")
        Q = Q[~bad]
    return Q


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="computer_work")
    ap.add_argument("--block", type=int, default=15)
    ap.add_argument("--contains", default="r10x3")
    ap.add_argument("--methods", nargs="*", default=None, help="default: all found")
    ap.add_argument("--run", default=None, help="run folder the figures are written into")
    ap.add_argument("--sort", default="overquery", choices=["overquery", "pool", "id"],
                    help="row order: by the method's marginal deviation (default), pool size, "
                         "or participant number")
    ap.add_argument("--drop", nargs="*", default=[], metavar="FOLDER:METHOD",
                    help="drop one method's rows from folders matching a substring, e.g. "
                         "`--drop -random_r10x3:random` to ignore the stale random arm that "
                         "lives inside the paired run folders when a corrected standalone "
                         "random run exists. Without this the two copies both survive the "
                         "dedup (their rows genuinely differ) and are silently pooled.")
    a = ap.parse_args()

    root = os.path.join(RESULTS_ROOT, a.target)
    qf = run_glob(a.target, f"*{a.contains}*", "queries_split*.csv")
    rf = run_glob(a.target, f"*{a.contains}*", "per_round_split*.csv")
    if not qf:
        raise SystemExit(f"no queries_split*.csv under {root}/*{a.contains}*")
    Q = pd.concat([pd.read_csv(f).assign(_src=f) for f in qf], ignore_index=True) \
          .drop_duplicates(["method", "split", "target", "round", "row"])
    Q = _drop(Q, a.drop)
    D = pd.concat([pd.read_csv(f).assign(_src=f) for f in rf], ignore_index=True)
    D = _drop(D, a.drop).drop_duplicates(["method", "split", "target", "round"])
    splits = sorted(Q.split.unique())
    ms = a.methods or ([m for m in ORDER if m in set(Q.method)] +
                       [m for m in sorted(set(Q.method)) if m not in ORDER])
    R = int(Q["round"].max())
    print(f"{len(Q)} queries, {len(D)} curve rows, methods {ms}, rounds 1-{R}")

    data, cols = load(get_target(a.target), verbose=True)
    ids = canonical_ids(data)
    shares = []
    for k in splits:
        sp, _, _ = prepare_one(data, cols, seed=k, W=a.block, ids=ids)
        n = pd.Series([sp.ids[u] for u in sp.owner]).value_counts()
        shares.append(n / n.sum())
        print(f"  split {k}: {len(sp.owner)} pool rows", flush=True)
    P = pd.concat(shares, axis=1).mean(axis=1)
    pids = list(P.sort_values(ascending=False).index)                    # biggest pool first
    P = P.reindex(pids)
    self_ref = float(P.mean())                       # what an unbiased sampler gives as "self"

    run = a.run or run_folders(a.target, f"*laplace*{a.contains}*")[0]
    rnds = np.arange(1, R + 1)

    for m in ms:
        Qm, Dm = Q[Q.method == m], D[D.method == m]
        fig, axes = plt.subplots(2, 1, figsize=(9.5, 9.0), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2.0, 3.4]))

        # --- 1. learning curve, with random behind it -----------------------------------
        if m != "random":
            g = D[D.method == "random"]
            if len(g):
                ref = g.pivot_table(index="split", columns="round", values="auroc")
                axes[0].plot(ref.columns, ref.mean(), color=COL["random"], lw=1.2,
                             alpha=0.8, zorder=1, label="random")
        per = Dm.pivot_table(index="split", columns="round", values="auroc")
        mu = per.mean()
        se = per.std(ddof=1) / np.sqrt(per.notna().sum())
        axes[0].plot(mu.index, mu.values, color=COL.get(m, "grey"), lw=2.0,
                     label=LABEL.get(m, m), zorder=3)
        axes[0].fill_between(mu.index, mu - se, mu + se, color=COL.get(m, "grey"),
                             alpha=0.18, lw=0, zorder=2)
        axes[0].set_ylabel("mean test AUROC", fontsize=9)
        axes[0].legend(fontsize=8, loc="lower right")
        axes[0].grid(alpha=0.25, lw=0.6)
        axes[0].set_title(f"{LABEL.get(m, m)} -- learning curve and query provenance "
                          f"({a.target}, W={a.block}, {len(splits)} split(s), "
                          f"{Dm.target.nunique()} targets)", fontsize=11, fontweight="bold")

        # self-share is no longer drawn -- reported to stdout only, and per target by
        # plots/per_person_al_and_query.py
        ss = Qm.groupby("round").is_self.mean().reindex(rnds)

        # --- 2. source x round enrichment ------------------------------------------------
        if a.sort == "id":
            rows = sorted(pids, key=lambda z: int(str(z)[1:]))
        elif a.sort == "pool":
            rows = list(P.sort_values(ascending=False).index)
        else:                                   # most over-queried first, least last
            tot = Qm.owner.value_counts().reindex(pids).fillna(0.0)
            rows = list((tot / tot.sum() - P).sort_values(ascending=False).index)
        M = np.zeros((len(rows), R))
        for j, r in enumerate(rnds):
            g = Qm[Qm["round"] == r]
            s = g.owner.value_counts(normalize=True) if len(g) else pd.Series(dtype=float)
            M[:, j] = [s.get(p, 0.0) - P[p] for p in rows]
        lo, hi = float(M.min()), float(M.max())
        if lo < 0 < hi:
            from matplotlib.colors import TwoSlopeNorm
            kw = dict(norm=TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi))
        else:
            lim = max(abs(lo), abs(hi)) or 1e-9
            kw = dict(vmin=-lim, vmax=lim)
        im = axes[1].imshow(M, cmap="RdBu_r", aspect="auto",
                            extent=(0.5, R + 0.5, len(rows) - 0.5, -0.5), **kw)
        axes[1].set_yticks(range(len(rows)))
        axes[1].set_yticklabels(rows, fontsize=6)
        axes[1].set_ylabel(f"source (queries came from), sorted by {a.sort}: "
                           f"over-queried at the top", fontsize=9)
        axes[1].set_xlabel("round", fontsize=10)
        axes[1].set_xticks(rnds)
        cb = fig.colorbar(im, ax=axes[1], fraction=0.035, pad=0.02)
        cb.set_label("query share $-$ pool share", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        fig.tight_layout()
        out = figure_path(run, f"al_and_query_share_{m}.png")
        fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"  {m:10s} self-share round1 {100 * ss.iloc[0]:5.1f}% -> "
              f"round{R} {100 * ss.iloc[-1]:5.1f}%   (unbiased {100 * self_ref:.1f}%)   "
              f"-> {os.path.basename(out)}")


if __name__ == "__main__":
    main()
