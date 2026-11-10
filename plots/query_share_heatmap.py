"""Source x target query-share heatmaps, one per acquisition function.

The same construction as Banaware_AL-2's meeting/plot_query_heatmap_cw.py:

    mats[m] = counts[m].div(counts[m].sum(axis=0), axis=1).sub(P, axis=0)

so cell (s, t) is "of all the labels bought for target t, the share that came from source s,
minus source s's share of the candidate pool". Zero means t's model bought from s exactly in
proportion to how much of the pool s is; red means over-queried, blue under-queried. Limits are
symmetric and shared across methods so the panels are comparable.

Unlike the Banaware script this one does NOT replay the acquisition: main.py writes the owner
of every query to queries_split<k>.csv, so the counts are read off the real run. That is also
why it covers EU and eer, which cannot be replayed without refitting the model.

The boxed diagonal is the target's own rows -- the one cell where "source" and "target" are the
same participant, and the one the personalisation question turns on.

    python plots/query_share_heatmap.py
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
from paths import RESULTS_ROOT, analysis_path, figure_path               # noqa: E402
from preprocess_data import canonical_ids, prepare_one                   # noqa: E402

ORDER = ["laplace", "eer", "typiclust", "coreset", "probcover", "random"]
LABEL = {"laplace": "EU"}


def boot_ci(v, n=1000, seed=0, pct=(2.5, 97.5)):
    """95% percentile CI of the mean of ``v``, resampling TARGETS with replacement.

    The target is the resampling unit, matching Banaware_AL-2's walking/query_enrichment.py.
    Splits are NOT resampled: there are only three and they share most of their training data,
    so they are not independent replicates and an interval built on them would be meaningless.
    This interval therefore says "how much would this method's diagonal move if we had drawn a
    different 43 participants", not "how much does it move across splits".
    """
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return tuple(np.percentile(m, pct))


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
    ap.add_argument("--run", default=None, help="run folder the outputs are written into")
    ap.add_argument("--cols", type=int, default=3, help="panel columns")
    ap.add_argument("--sort", default="overquery",
                    choices=["overquery", "overquery_offdiag", "pool", "id"],
                    help="row order. `overquery` (default) puts the most over-queried sources "
                         "first, by that method's own marginal `query share - pool share`. "
                         "`overquery_offdiag` ranks on labels bought for OTHER people only, "
                         "which is the useful one for EU and eer, whose marginal is dominated "
                         "by self-buying and therefore nearly uniform. `pool` orders by size, "
                         "`id` by participant number. Targets take the SAME order as sources "
                         "so the self cells stay on the diagonal.")
    ap.add_argument("--top_sources", type=int, default=0,
                    help="show only the k highest-ranked source rows (0 = all)")
    ap.add_argument("--drop", nargs="*", default=[], metavar="FOLDER:METHOD",
                    help="drop one method's rows from folders matching a substring, e.g. "
                         "`--drop -random_r10x3:random` to ignore the stale random arm that "
                         "lives inside the paired run folders when a corrected standalone "
                         "random run exists. Without this the two copies both survive the "
                         "dedup (their rows genuinely differ) and are silently pooled.")
    ap.add_argument("--out", default="query_share_heatmaps.png")
    ap.add_argument("--boot", type=int, default=1000,
                    help="bootstrap resamples over TARGETS for the per-method error bars")
    a = ap.parse_args()

    root = os.path.join(RESULTS_ROOT, a.target)
    files = sorted(glob.glob(os.path.join(root, f"*{a.contains}*", "queries_split*.csv")))
    if not files:
        raise SystemExit(f"no queries_split*.csv under {root}/*{a.contains}*")
    Q = pd.concat([pd.read_csv(f).assign(_src=f) for f in files], ignore_index=True)
    # random is written into several run folders from the same seed; one copy per query
    Q = Q.drop_duplicates(["method", "split", "target", "round", "row"])
    Q = _drop(Q, a.drop)
    splits = sorted(Q.split.unique())
    ms = [m for m in ORDER if m in set(Q.method)] + \
         [m for m in sorted(set(Q.method)) if m not in ORDER]
    print(f"{len(files)} file(s), {len(Q)} queries, methods {ms}, splits {splits}")

    # pool share per source, rebuilt with the seed main.py used (seed=split_id)
    data, cols = load(get_target(a.target), verbose=True)
    ids = canonical_ids(data)
    shares = []
    for k in splits:
        sp, _, _ = prepare_one(data, cols, seed=k, W=a.block, ids=ids)
        n = pd.Series([sp.ids[u] for u in sp.owner]).value_counts()
        shares.append(n / n.sum())
        print(f"  split {k}: {len(sp.owner)} pool rows", flush=True)
    P = pd.concat(shares, axis=1).mean(axis=1)

    key = lambda s: int(str(s)[1:])                                      # noqa: E731
    pids = sorted(set(P.index) | set(Q.owner), key=key)                  # sources
    targets = sorted(set(Q.target), key=key)
    P = P.reindex(pids).fillna(0.0)

    counts, mats = {}, {}
    for m in ms:
        c = (Q[Q.method == m].pivot_table(index="owner", columns="target", values="row",
                                          aggfunc="size")
             .reindex(index=pids, columns=targets).fillna(0.0))
        counts[m] = c
        mats[m] = c.div(c.sum(axis=0), axis=1).sub(P, axis=0)

    def row_order(m):
        """-> (sources, targets) for one panel. Same order on both axes keeps the diagonal."""
        if a.sort == "id":
            rank = pd.Series(range(len(pids)), index=pids)
        elif a.sort == "pool":
            rank = -P
        elif a.sort == "overquery_offdiag":
            c = counts[m].copy()
            for t in targets:                      # strangers only: blank the self cells
                if t in c.index:
                    c.loc[t, t] = 0.0
            rank = -(c.sum(axis=1) / c.values.sum() - P)
        else:
            sm = counts[m].sum(axis=1)
            rank = -(sm / sm.sum() - P)
        src = list(rank.sort_values(kind="mergesort").index)
        if a.top_sources > 0:
            src = src[:a.top_sources]
        return src, [t for t in src if t in targets] + \
                    [t for t in targets if t not in src]

    run = a.run or sorted(glob.glob(os.path.join(root, f"*laplace*{a.contains}*")))[0]

    tbl = {"pool_share": P}
    for m in ms:
        s = counts[m].sum(axis=1)
        tbl[f"{m}_share"] = s / s.sum()
        tbl[f"{m}_delta"] = tbl[f"{m}_share"] - P
    csv = analysis_path(run, "query_share_by_method.csv")
    pd.DataFrame(tbl).to_csv(csv)
    print(f"saved {csv}")

    # Each panel gets its OWN colour scale and colour bar. A shared scale is set by EU's
    # diagonal (~0.93) and flattens the other five panels to near-white, hiding structure that
    # is real but an order of magnitude smaller. The price is that colour no longer means the
    # same number across panels -- read each bar before comparing panels.
    #
    # The scale is also TWO-SLOPED, not symmetric. `query share - pool share` cannot go below
    # `-pool_share` -- you cannot buy negative labels -- so with 43 participants the floor is
    # about -0.05 while EU's self-buying ceiling is +0.93. On a symmetric scale that floor is
    # 5% of the range and every under-queried cell renders white, which reads as "nothing is
    # under-queried" when in fact almost everything is. Giving each side its own half of the
    # map shows the floor. Colour distance is then NOT comparable either side of zero.
    from matplotlib.colors import TwoSlopeNorm
    ncol = min(a.cols, len(ms))
    nrow = int(np.ceil(len(ms) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 5.9 * nrow), squeeze=False)
    for ax, m in zip(axes.ravel(), ms):
        src, tgts = row_order(m)
        V = mats[m].loc[src, tgts].values
        lo, hi = float(np.nanmin(V)), float(np.nanmax(V))
        if lo < 0 < hi:
            kw = dict(norm=TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi))
        else:
            lim = max(abs(lo), abs(hi)) or 1e-9
            kw = dict(vmin=-lim, vmax=lim)
        im = ax.imshow(V, cmap="RdBu_r", aspect="auto", **kw)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6)
        cb.set_label("query share $-$ pool share", fontsize=7)
        ax.set_xticks(range(len(tgts)))
        ax.set_xticklabels(tgts, rotation=90, fontsize=5)
        ax.set_yticks(range(len(src)))
        ax.set_yticklabels(src, fontsize=5)
        d = np.array([mats[m].loc[t, t] for t in targets if t in pids], float)
        lo, hi = boot_ci(d, a.boot)
        ax.set_title(f"{LABEL.get(m, m)}\ndiagonal {d.mean():+.3f}  95% CI "
                     f"[{lo:+.3f}, {hi:+.3f}]", fontsize=10, fontweight="bold")
        ax.set_xlabel("target", fontsize=9)
        ax.set_ylabel("source (queries came from)", fontsize=9)
        for j, t in enumerate(tgts):                        # the target's own rows
            if t in src:
                ax.add_patch(plt.Rectangle((j - .5, src.index(t) - .5), 1, 1, fill=False,
                                           edgecolor="black", lw=0.6))
    for ax in axes.ravel()[len(ms):]:
        ax.axis("off")
    fig.suptitle(f"{a.target}: per-participant query share minus pool share, "
                 f"by acquisition function ({len(splits)} split(s)) -- rows sorted by "
                 f"{a.sort}, most over-queried first; each panel has its own two-sloped "
                 f"colour scale", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = figure_path(run, a.out)
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    print(f"saved {out}")

    print("\nlargest |deviation| per method:")
    for m in ms:
        v = mats[m].stack().abs().sort_values(ascending=False)
        print(f"  {m:10s} " + ", ".join(f"{s}->{t} {mats[m].loc[s, t]:+.3f}"
                                        for (s, t) in v.index[:3]))
    # ---- per-method error bars: the diagonal, and the off-diagonal spread ------------
    diag = {m: np.array([mats[m].loc[t, t] for t in targets if t in pids], float) for m in ms}
    offd = {m: np.array([mats[m].loc[[s2 for s2 in pids if s2 != t], t].abs().mean()
                         for t in targets if t in pids], float) for m in ms}
    print(f"\ndiagonal (target's own rows), bootstrap over {len(targets)} targets, "
          f"{a.boot} resamples:")
    rows = []
    for m in ms:
        dlo, dhi = boot_ci(diag[m], a.boot)
        olo, ohi = boot_ci(offd[m], a.boot)
        rows.append(dict(method=m, diag=diag[m].mean(), diag_lo=dlo, diag_hi=dhi,
                         offdiag_absmean=offd[m].mean(), off_lo=olo, off_hi=ohi))
        print(f"  {m:10s} {diag[m].mean():+.4f}  [{dlo:+.4f}, {dhi:+.4f}]   "
              f"off-diag |mean| {offd[m].mean():.4f}  [{olo:.4f}, {ohi:.4f}]")
    S = pd.DataFrame(rows).set_index("method")
    scsv = analysis_path(run, "query_share_summary.csv")
    S.to_csv(scsv)
    print(f"saved {scsv}")

    f2, ax2 = plt.subplots(1, 2, figsize=(11, 4.2))
    for axx, (col, lo, hi, ttl) in zip(ax2, [
            ("diag", "diag_lo", "diag_hi", "target's OWN rows (the boxed diagonal)"),
            ("offdiag_absmean", "off_lo", "off_hi", "strangers: mean |query - pool| share")]):
        y = np.arange(len(S))
        err = np.vstack([S[col] - S[lo], S[hi] - S[col]])
        axx.errorbar(S[col], y, xerr=err, fmt="o", color="black", ecolor="dimgrey",
                     capsize=3, ms=5, lw=1.2)
        axx.axvline(0, color="crimson", lw=0.9, ls="--")
        axx.set_yticks(y)
        axx.set_yticklabels([LABEL.get(m, m) for m in S.index], fontsize=9)
        axx.invert_yaxis()
        axx.set_title(ttl, fontsize=10)
        axx.set_xlabel("query share $-$ pool share", fontsize=9)
        axx.grid(axis="x", alpha=0.25, lw=0.6)
    f2.suptitle(f"{a.target}: query share minus pool share, mean over targets "
                f"with 95% bootstrap CI (resampled over the {len(targets)} targets)",
                fontsize=11)
    f2.tight_layout(rect=[0, 0, 1, 0.94])
    out2 = figure_path(run, a.out.replace(".png", "_errorbars.png"))
    f2.savefig(out2, dpi=160, facecolor="white", bbox_inches="tight")
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
