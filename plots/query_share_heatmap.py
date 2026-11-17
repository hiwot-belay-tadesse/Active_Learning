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
from paths import (RESULTS_ROOT, analysis_path, figure_path, run_folders,
                   run_glob)               # noqa: E402
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
    ap.add_argument("--scale", default="shared", choices=["shared", "per_panel"],
                    help="shared (default) puts every panel on ONE colour scale, so a colour "
                         "means the same number everywhere and the panels can be compared -- "
                         "which is the point of showing them side by side. per_panel gives "
                         "each its own scale: prettier, but a mid-blue then means -0.05 in "
                         "one panel and -0.015 in the next.")
    ap.add_argument("--mask_diag", action="store_true",
                    help="blank the diagonal (the target's own rows). On a shared scale the "
                         "diagonal sets the range -- EU's is +0.92 against off-diagonal "
                         "values under 0.05 -- so everything else renders white. Masking it "
                         "hands the whole range to the off-diagonal, which is where the "
                         "'whose labels does it buy' question actually lives. The diagonal "
                         "is in every panel title anyway.")
    ap.add_argument("--split", type=int, default=None, metavar="k",
                    help="use ONE split instead of pooling all of them, and measure against "
                         "that split's own pool share. The denominator per target drops to "
                         "rounds x batch (30 here), so a source's share moves in steps of "
                         "1/30 = 0.033 -- larger than every pool share, which means a cell is "
                         "essentially picked-or-not rather than a graded deviation. Useful to "
                         "see one replicate as it actually ran; not for reading structure.")
    ap.add_argument("--show_mean", action="store_true",
                    help="append a one-column strip on the right of each panel holding each "
                         "source's ROW MEAN -- its deviation averaged over all targets, the "
                         "same number the aggregated bar chart shows. For the geometry "
                         "methods that strip reproduces almost the whole panel (94-99.9%% of "
                         "its variance); for EU it does not (47%%), which is the point of "
                         "showing both side by side. Drawn on the panel's own colour scale, "
                         "separated by a gap so it is not read as another target.")
    ap.add_argument("--min_z", type=float, default=0.0,
                    help="blank every (source, target) cell whose deviation is not this many "
                         "standard errors from zero, the SE taken ACROSS SPLITS. The splits "
                         "are independent replicates of the whole experiment, so this asks "
                         "'would this pair come out over-queried again if the days were "
                         "partitioned differently?' and leaves only the cells that survive. "
                         "2 is the usual choice. Costs a second pass: the matrix is rebuilt "
                         "inside each split.")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="draw only these acquisitions, in the usual panel order. The default "
                         "is every method found in the queries CSVs -- which is what makes the "
                         "figure a comparison, so narrowing it is for looking closely at one "
                         "arm, not for the figure of record.")
    ap.add_argument("--sign", default="query_minus_pool",
                    choices=["query_minus_pool", "pool_minus_query"],
                    help="which way round the deviation is drawn. The default `query share - "
                         "pool share` is positive for an OVER-queried source. "
                         "`pool_minus_query` negates every cell, so positive (red) is a source "
                         "the acquisition bought LESS from than its size in the pool -- the "
                         "same information, read as under-use. The row ORDER does not change, "
                         "so the two figures are cell-for-cell comparable; with this sign the "
                         "leading rows are therefore the most negative ones.")
    ap.add_argument("--out", default="query_share_heatmaps.png")
    ap.add_argument("--boot", type=int, default=1000,
                    help="bootstrap resamples over TARGETS for the per-method error bars")
    a = ap.parse_args()

    root = os.path.join(RESULTS_ROOT, a.target)
    files = run_glob(a.target, f"*{a.contains}*", "queries_split*.csv")
    if not files:
        raise SystemExit(f"no queries_split*.csv under {root}/*{a.contains}*")
    Q = pd.concat([pd.read_csv(f).assign(_src=f) for f in files], ignore_index=True)
    # random is written into several run folders from the same seed; one copy per query
    Q = Q.drop_duplicates(["method", "split", "target", "round", "row"])
    Q = _drop(Q, a.drop)
    if a.split is not None:
        if a.split not in set(Q.split):
            raise SystemExit(f"split {a.split} not in this run; have "
                             f"{sorted(Q.split.unique())}")
        Q = Q[Q.split == a.split]
    splits = sorted(Q.split.unique())
    if a.methods is not None:
        unknown = set(a.methods) - set(Q.method)
        if unknown:
            raise SystemExit(f"unknown method(s) {sorted(unknown)}; this run has "
                             f"{sorted(set(Q.method))}")
        Q = Q[Q.method.isin(a.methods)]
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
    # The sign flip is applied once, here, to the finished matrices: everything downstream
    # (colour scale, masking, the diagonal in the panel title, the mean strip) then follows
    # without a second code path. SE is a standard deviation and is sign-free.
    FLIP = a.sign == "pool_minus_query"
    if FLIP:
        mats = {m: -v for m, v in mats.items()}
    DLAB = "pool share $-$ query share" if FLIP else "query share $-$ pool share"

    # Per-split matrices, for the across-split SE. Built only when --min_z asks for it,
    # because it is 20x the pivoting work and most calls do not need it.
    SE = {}
    if a.min_z > 0:
        for m in ms:
            per = []
            for k in splits:
                qk = Q[(Q.method == m) & (Q.split == k)]
                ck = (qk.pivot_table(index="owner", columns="target", values="row",
                                     aggfunc="size")
                      .reindex(index=pids, columns=targets).fillna(0.0))
                tot = ck.sum(axis=0).replace(0.0, np.nan)
                per.append(ck.div(tot, axis=1).sub(P, axis=0))
            A3 = np.dstack([x.values for x in per])
            SE[m] = pd.DataFrame(np.nanstd(A3, axis=2, ddof=1) / np.sqrt(A3.shape[2]),
                                 index=pids, columns=targets)
        kept = {m: float(np.mean(np.abs(mats[m].values) >= a.min_z * SE[m].values))
                for m in ms}
        print("cells surviving |mean| >= "
              f"{a.min_z:g} SE across splits: "
              + ", ".join(f"{m} {100 * v:.1f}%" for m, v in kept.items()))

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

    run = a.run or run_folders(a.target, f"*laplace*{a.contains}*")[0]

    tbl = {"pool_share": P}
    for m in ms:
        s = counts[m].sum(axis=1)
        tbl[f"{m}_share"] = s / s.sum()
        tbl[f"{m}_delta"] = (P - tbl[f"{m}_share"]) if FLIP else (tbl[f"{m}_share"] - P)
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

    def panel_values(m):
        src, tgts = row_order(m)
        V = mats[m].loc[src, tgts].copy()
        if a.min_z > 0:
            V = V.where(V.abs() >= a.min_z * SE[m].loc[src, tgts])
        if a.mask_diag:
            for t in V.columns:
                if t in V.index:
                    V.loc[t, t] = np.nan
        return src, tgts, V.values

    # On a shared scale the limits come from EVERY panel at once, so one colour is one number
    # throughout the figure. Methods whose deviations are genuinely tiny then look washed out
    # -- that is the comparison, not a defect of the drawing.
    GLO = GHI = None
    if a.scale == "shared":
        allv = np.concatenate([panel_values(m)[2].ravel() for m in ms])
        GLO, GHI = float(np.nanmin(allv)), float(np.nanmax(allv))
        print(f"shared colour scale: [{GLO:+.4f}, {GHI:+.4f}]"
              + ("  (diagonal masked)" if a.mask_diag else ""))
    for ax, m in zip(axes.ravel(), ms):
        src, tgts, V = panel_values(m)
        lo, hi = ((GLO, GHI) if GLO is not None
                  else (float(np.nanmin(V)), float(np.nanmax(V))))
        if lo < 0 < hi:
            kw = dict(norm=TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi))
        else:
            lim = max(abs(lo), abs(hi)) or 1e-9
            kw = dict(vmin=-lim, vmax=lim)
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("0.85")                       # masked diagonal reads as grey, not white
        im = ax.imshow(V, cmap=cmap, aspect="auto", **kw)
        # On a shared scale every panel's bar would carry the SAME ticks, so one bar is drawn
        # for the whole figure after the loop. Per-panel scales still need one bar each --
        # that is the entire point of them.
        if a.scale != "shared":
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=6)
            cb.set_label(DLAB, fontsize=7)
        if a.show_mean:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            axm = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.14)
            axm.imshow(np.nanmean(V, axis=1)[:, None], cmap=cmap, aspect="auto", **kw)
            axm.set_yticks([])
            axm.set_xticks([0])
            axm.set_xticklabels(["mean"], rotation=90, fontsize=6)
            for sp_ in axm.spines.values():
                sp_.set_linewidth(1.2)
        ax.set_xticks(range(len(tgts)))
        ax.set_xticklabels(tgts, rotation=90, fontsize=5)
        ax.set_yticks(range(len(src)))
        ax.set_yticklabels(src, fontsize=5)
        d = np.array([mats[m].loc[t, t] for t in targets if t in pids], float)
        # --boot 0 drops the interval everywhere. Its unit is the TARGET, so it is still
        # computable from a single split -- but then it describes participant sampling only
        # and says nothing about the split, which invites misreading.
        if a.boot > 0:
            lo, hi = boot_ci(d, a.boot)
            ttl = f"diagonal {d.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
        else:
            ttl = f"diagonal {d.mean():+.3f}"
        ax.set_title(f"{LABEL.get(m, m)}\n{ttl}", fontsize=10, fontweight="bold")
        ax.set_xlabel("target", fontsize=9)
        ax.set_ylabel("source (queries came from)", fontsize=9)
        for j, t in enumerate(tgts):                        # the target's own rows
            if t in src:
                ax.add_patch(plt.Rectangle((j - .5, src.index(t) - .5), 1, 1, fill=False,
                                           edgecolor="black", lw=0.6))
    for ax in axes.ravel()[len(ms):]:
        ax.axis("off")
    if a.scale == "shared":
        cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cb.ax.tick_params(labelsize=8)
        cb.set_label(DLAB, fontsize=9)
    what = "pool share minus query share" if FLIP else "query share minus pool share"
    first = " (so most NEGATIVE first, on this sign)" if FLIP else ""
    # bbox_inches="tight" sizes the saved PNG to its widest artist, so an unwrapped one-line
    # suptitle stretches a single-panel figure to several times the panel's own width. Wrap it
    # to roughly the figure's width instead.
    import textwrap
    sub = (f"{a.target}: per-participant {what}, by acquisition function "
           f"({len(splits)} split(s)) -- rows sorted by {a.sort}, most over-queried "
           f"first{first}; each panel has its own two-sloped colour scale")
    fig.suptitle(textwrap.fill(sub, width=max(60, 22 * ncol)), fontsize=13)
    # A figure-level colour bar steals space from every axes at once, which tight_layout
    # cannot account for -- calling it then warns and shifts the panels. Only lay out when
    # each panel owns its bar.
    if a.scale != "shared":
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = figure_path(run, a.out if a.split is None
                      else a.out.replace(".png", f"_split{a.split}.png"))
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
    print(f"\ndiagonal (target's own rows)" + (
        f", bootstrap over {len(targets)} targets, {a.boot} resamples:" if a.boot > 0
        else " -- no bootstrap (--boot 0):"))
    rows = []
    for m in ms:
        r = dict(method=m, diag=diag[m].mean(), offdiag_absmean=offd[m].mean())
        if a.boot > 0:
            dlo, dhi = boot_ci(diag[m], a.boot)
            olo, ohi = boot_ci(offd[m], a.boot)
            r.update(diag_lo=dlo, diag_hi=dhi, off_lo=olo, off_hi=ohi)
            print(f"  {m:10s} {diag[m].mean():+.4f}  [{dlo:+.4f}, {dhi:+.4f}]   "
                  f"off-diag |mean| {offd[m].mean():.4f}  [{olo:.4f}, {ohi:.4f}]")
        else:
            print(f"  {m:10s} {diag[m].mean():+.4f}   "
                  f"off-diag |mean| {offd[m].mean():.4f}")
        rows.append(r)
    S = pd.DataFrame(rows).set_index("method")
    scsv = analysis_path(run, "query_share_summary.csv")
    S.to_csv(scsv)
    print(f"saved {scsv}")

    # The bootstrap CIs are NOT plotted. The same five numbers and their intervals are
    # printed above, written to query_share_summary.csv, and already carried in each panel's
    # title, so a separate two-panel error-bar figure restated them with nothing added.


if __name__ == "__main__":
    main()
