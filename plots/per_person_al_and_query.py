"""Per-participant grid, one figure per acquisition: the curve and who it bought from.

One cell per target, each split into the same two panels as al_and_query_share_<method>.png
but for that participant alone:

Scale and resolution, both of which bite here
---------------------------------------------
One round holds only batch x splits purchases (9 here) over 43 possible sources, so a source
either gets picked -- jumping to >=1/9 -- or sits at exactly `0 - pool_share`, about -0.023.
That floor is the hardest bound in the plot: you cannot buy negative labels. Against a self
band near +0.9 a symmetric colour scale renders every under-queried cell white, which reads as
"nothing is under-queried" when in truth almost everything is. Two things follow:

* the colour norm is TWO-SLOPED by default (`--norm twoslope`): blue spans the real negative
  range and red the real positive one, so the floor is visible. `--norm symmetric` restores a
  single scale if you want colour distances to be comparable either side of zero.
* per-round `share - pool share` is close to vacuous as a statistic -- with 9 draws the
  expected count per source is 0.21, so ~34 of 43 sources are expected to be zero every round
  and "under-queried" is arithmetic rather than behaviour. `--cumulative` instead compares the
  share of everything bought UP TO round r, which by the last round rests on 90 purchases and
  is a claim about the method rather than about the batch size.

  top     that target's mean test AUROC, +-1 SE over splits
  bottom  source x round heatmap of `query share - pool share` for THIS target's purchases.
          Rows are the participants that actually SUPPLIED this target's labels -- the top
          `--top` of them by total contribution, named on the axis, with everyone else summed
          into a final "other (k)" row. The target itself is drawn in the method's colour when
          it appears, so self-buying is still visible, but now every band has a name.
          Red = supplied more of that round's labels than its pool size warrants, blue = less.

          Rows therefore differ from cell to cell: each cell answers "who contributed to THIS
          participant", not "where does this participant sit in a fixed list". Median target
          draws from ~20 distinct sources (EU) to ~38 (random), so the named rows are a tail-
          truncated view, not the whole story -- the "other" row carries what is left.

The colour scale is shared across every target in one figure, so cells are comparable within a
method but not across methods. Each round holds only batch x splits purchases (9 here), so a
cell is coarse by construction -- read bands, not single squares.

Cells sit in the SAME position in every method's figure -- ordered by one method's gap over
random (`--order_by`, default laplace) -- so the figures can be flipped through and compared
target by target.

    python plots/per_person_al_and_query.py
    python plots/per_person_al_and_query.py --methods laplace eer --rows 4
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
from paths import RESULTS_ROOT, figure_path                              # noqa: E402
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
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--order_by", default="self",
                    help="method whose gap over random orders the panels. `self` (default) "
                         "uses the method each figure is about, so a coreset figure is sorted "
                         "by coreset's own gap. Name a method instead (e.g. laplace) to pin "
                         "ONE order across every figure, which is what you want when flipping "
                         "between methods to compare the same target.")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cumulative", action="store_true",
                    help="share of everything bought up to round r, not just in round r")
    ap.add_argument("--picked_counts", action="store_true",
                    help="with --only_picked, colour by the raw number of labels bought "
                         "instead of by `share - pool share`")
    ap.add_argument("--only_picked", action="store_true",
                    help="show only the sources actually bought from in each round, coloured "
                         "by how many labels they supplied; everything else is left blank. "
                         "At this resolution that is all the figure can honestly say: the "
                         "largest pool share is 0.043 and the smallest possible picked share "
                         "is 1/9 = 0.111, so `share - pool` is positive for EVERY pick and "
                         "negative for every non-pick -- the diverging scale is a picked / "
                         "not-picked indicator wearing a continuous disguise.")
    ap.add_argument("--norm", choices=["twoslope", "symmetric"], default="twoslope",
                    help="twoslope (default) gives the negative and positive ranges their own "
                         "half of the colour map, so the -pool_share floor is visible")
    ap.add_argument("--top", type=int, default=0,
                    help="0 (default) = every source, in ONE fixed order (by pool share) so a "
                         "row is the same participant in every cell and the names need "
                         "printing only once per grid row. A positive k instead shows that "
                         "target's top k contributors by volume, named in each cell, with the "
                         "rest pooled into `other` -- legible per cell, but rows then mean "
                         "something different in every cell and cannot be compared across them.")
    ap.add_argument("--run", default=None)
    ap.add_argument("--drop", nargs="*", default=[], metavar="FOLDER:METHOD",
                    help="drop one method's rows from folders matching a substring")
    a = ap.parse_args()

    root = os.path.join(RESULTS_ROOT, a.target)
    qf = sorted(glob.glob(os.path.join(root, f"*{a.contains}*", "queries_split*.csv")))
    rf = sorted(glob.glob(os.path.join(root, f"*{a.contains}*", "per_round_split*.csv")))
    if not qf:
        raise SystemExit(f"no queries_split*.csv under {root}/*{a.contains}*")
    Q = pd.concat([pd.read_csv(f).assign(_src=f) for f in qf], ignore_index=True) \
          .drop_duplicates(["method", "split", "target", "round", "row"])
    Q = _drop(Q, a.drop)
    D = pd.concat([pd.read_csv(f).assign(_src=f) for f in rf], ignore_index=True)
    D = _drop(D, a.drop).drop_duplicates(["method", "split", "target", "round"])
    ms = a.methods or ([m for m in ORDER if m in set(Q.method)] +
                       [m for m in sorted(set(Q.method)) if m not in ORDER])
    R = int(D["round"].max())

    data, cols = load(get_target(a.target), verbose=True)
    ids = canonical_ids(data)
    shares = []
    for k in sorted(Q.split.unique()):
        sp, _, _ = prepare_one(data, cols, seed=k, W=a.block, ids=ids)
        n = pd.Series([sp.ids[u] for u in sp.owner]).value_counts()
        shares.append(n / n.sum())
    P = pd.concat(shares, axis=1).mean(axis=1)           # per-participant pool share

    # panel order: one method's gap over random at the last round, descending
    W = D[D["round"] == R].pivot_table(index=["target", "split"], columns="method",
                                       values="auroc")

    def panel_order(m):
        by = m if a.order_by == "self" else a.order_by
        if by in W and "random" in W:
            g = (W[by] - W["random"]).groupby("target").mean().sort_values(ascending=False)
            return g, list(g.index), by
        return None, sorted(set(D.target), key=lambda z: int(z[1:])), by
    nrow = a.rows
    ncol = int(np.ceil(len(set(D.target)) / nrow))
    run = a.run or sorted(glob.glob(os.path.join(root, f"*laplace*{a.contains}*")))[0]
    print(f"{D.target.nunique()} targets, rounds 0-{R}, panel order by {a.order_by}")

    pids = list(P.sort_values(ascending=False).index)          # sources, biggest pool first
    rnds = np.arange(1, R + 1)
    for m in ms:
        Dm, Qm = D[D.method == m], Q[Q.method == m]
        gap, order, order_by = panel_order(m)

        # per-target source x round deviation
        M, ROWS = {}, {}
        for tgt in order:
            g = Qm[Qm.target == tgt]
            if a.top > 0:                       # per-cell contributors, named in every cell
                tot = g.owner.value_counts()
                named, rest = list(tot.index[:a.top]), list(tot.index[a.top:])
                labels = named + ([f"other ({len(rest)})"] if rest else [])
            else:                               # every source, one fixed order for all cells
                named, rest, labels = pids, [], pids
            A = np.zeros((len(labels), R))
            for j, r in enumerate(rnds):
                gr = g[g["round"] <= r] if a.cumulative else g[g["round"] == r]
                ct = gr.owner.value_counts() if len(gr) else pd.Series(dtype=float)
                if a.only_picked and a.picked_counts:
                    col = [ct.get(pp, 0.0) for pp in named]
                    if rest:
                        col.append(sum(ct.get(pp, 0.0) for pp in rest))
                else:
                    sh = (gr.owner.value_counts(normalize=True) if len(gr)
                          else pd.Series(dtype=float))
                    col = [sh.get(pp, 0.0) - P.get(pp, 0.0) for pp in named]
                    if rest:
                        col.append(sum(sh.get(pp, 0.0) - P.get(pp, 0.0) for pp in rest))
                A[:, j] = col
                if a.only_picked:                  # blank wherever nothing was bought
                    nz = [ct.get(pp, 0.0) for pp in named] + \
                         ([sum(ct.get(pp, 0.0) for pp in rest)] if rest else [])
                    A[np.array(nz) == 0, j] = np.nan
            M[tgt], ROWS[tgt] = A, labels
        hi = max(float(np.nanmax(A)) for A in M.values())
        lo = min(float(np.nanmin(A)) for A in M.values())
        if a.only_picked and not (lo < 0 < hi):
            # every pick is over-queried here: the largest pool share is 0.043 while the
            # smallest possible picked share is 1/9 = 0.111, so `share - pool` cannot be
            # negative for a source that was bought from at all. A sequential ramp says that
            # honestly; the shade grades HOW over-queried, which counts alone cannot show --
            # a small participant picked once is further above its share than a large one.
            cmap, norm, vkw = "YlOrRd", None, dict(vmin=min(lo, 0) if lo < 0 else lo, vmax=hi)
        elif a.norm == "twoslope" and lo < 0 < hi:
            from matplotlib.colors import TwoSlopeNorm
            cmap, norm, vkw = "RdBu_r", TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi), {}
        else:
            lim = max(abs(lo), abs(hi)) or 1e-9
            cmap, norm, vkw = "RdBu_r", None, dict(vmin=-lim, vmax=lim)

        allsrc = a.top <= 0
        cell_h, hr = (5.0, [1.0, 3.4]) if allsrc else (3.25, [1.0, 1.75])
        fig = plt.figure(figsize=(2.45 * ncol, cell_h * nrow))
        outer = fig.add_gridspec(nrow, ncol, hspace=0.30, wspace=0.14 if allsrc else 0.18)
        for i, tgt in enumerate(order):
            cell = outer[i // ncol, i % ncol].subgridspec(2, 1, height_ratios=hr, hspace=0.08)
            ax, axh = fig.add_subplot(cell[0]), fig.add_subplot(cell[1])

            g = Dm[Dm.target == tgt]
            if len(g):
                per = g.pivot_table(index="split", columns="round", values="auroc")
                mu = per.mean()
                se = per.std(ddof=1) / np.sqrt(per.notna().sum())
                ax.plot(mu.index, mu.values, color=COL.get(m, "grey"), lw=1.3)
                ax.fill_between(mu.index, mu - se, mu + se, color=COL.get(m, "grey"),
                                alpha=0.18, lw=0)
            ax.set_xlim(0, R + 1)
            ax.set_ylim(0.15, 1.0)
            ax.set_xticks(np.arange(0, R + 1))
            ax.set_xticklabels([])
            ax.tick_params(labelsize=5)
            ax.grid(alpha=0.2, lw=0.5)
            ax.set_title(tgt if gap is None else f"{tgt}  {gap[tgt]:+.3f}", fontsize=7)
            if i % ncol == 0:
                ax.set_ylabel("AUROC", fontsize=6)

            lab = ROWS[tgt]
            # round r's column spans [r, r+1], so its LEFT EDGE sits on tick r -- the curve
            # above has its round-r point on the same tick. Round 0 is the seed set: the curve
            # starts there and the heatmap has no column before tick 1, which is where the
            # first purchases happen.
            im = axh.imshow(M[tgt], cmap=cmap, norm=norm, aspect="auto",
                            extent=(1, R + 1, len(lab) - 0.5, -0.5), **vkw)
            axh.set_xlim(0, R + 1)
            if allsrc:
                # one fixed order, so the names are printed once per grid row, on the left.
                # Each cell marks its own target's row with a coloured bar at the left edge.
                if i % ncol == 0:
                    axh.set_yticks(range(len(lab)))
                    axh.set_yticklabels(lab, fontsize=3.8)
                else:
                    axh.set_yticks([])
                if tgt in lab:
                    k = lab.index(tgt)
                    axh.add_patch(plt.Rectangle((0.5, k - 0.5), 0.33, 1.0, lw=0,
                                                color=COL.get(m, "black"), clip_on=False))
                    axh.text(0.42, k, tgt, ha="right", va="center", fontsize=4.2,
                             color=COL.get(m, "black"), fontweight="bold", clip_on=False)
            else:
                axh.set_yticks(range(len(lab)))
                axh.set_yticklabels(lab, fontsize=4.5)
                for tick, nm in zip(axh.get_yticklabels(), lab):
                    if nm == tgt:
                        tick.set_color(COL.get(m, "black"))
                        tick.set_fontweight("bold")
                        tick.set_fontsize(5.5)
            axh.tick_params(labelsize=4.5, length=2)
            # identical ticks to the curve above (0..R), so the two panels line up exactly:
            # round 0 is the seed set, where the curve starts and no purchase has happened yet
            axh.set_xticks(np.arange(0, R + 1))
            if i // ncol == nrow - 1 or i + ncol >= len(order):
                axh.set_xticklabels([str(r) for r in range(R + 1)], fontsize=4.5)
                axh.set_xlabel("round", fontsize=6)
            else:
                axh.set_xticklabels([])
            if i % ncol == 0:
                axh.set_ylabel("source", fontsize=6)

        cax = fig.add_axes([0.995, 0.35, 0.006, 0.3])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(("labels bought in the round" if a.picked_counts else
                      "picked sources only: query share $-$ pool share") if a.only_picked else
                     (("cumulative " if a.cumulative else "per-round ")
                      + "query share $-$ pool share"), fontsize=7)
        cb.ax.tick_params(labelsize=6)
        fig.suptitle(f"{LABEL.get(m, m)}: per-participant AUROC (top of each cell) and the "
                     f"source x round provenance of that participant's purchases (bottom). "
                     + (f"Rows = all {len(pids)} sources in one fixed order (by pool share), "
                        f"named on the left of each grid row; the bar marks each cell's own "
                        f"target. " if a.top <= 0 else
                        f"Rows = that target's top {a.top} contributors by volume, named; the "
                        f"target itself in colour. ")
                     + f"Cell title = {LABEL.get(order_by, order_by)}-random gap.",
                     fontsize=11)
        name = f"per_person_al_and_query_{m}{'_cumulative' if a.cumulative else ''}.png"
        fig.savefig(figure_path(run, name), dpi=140, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {name}")


if __name__ == "__main__":
    main()
