"""Which participants did an acquisition buy from, round by round, for each target?

A grid with ONE PANEL PER TARGET. Inside a panel, one stacked bar per AL round: the bar is the
labels bought for that target in that round, split by the participant who OWNED each queried
row. Colour = owner, the same colour for the same person in every panel, so a colour running
across a panel is one donor's contribution over the budget.

The target's OWN segment is outlined in black. Pooling over targets (the earlier version of this
figure) hid exactly that: every participant is the target in 1/43 of the runs, so the self mass
was smeared over all 43 colours and the rounds looked identical. Per target it is one block you
can watch shrink.

Bars are normalised to 100% of that target's round, so a panel is composition, not volume. The
panel title gives the target and its self-share over the whole run.

Read-only: main.py records the owner of every query in queries_split<k>.csv, so nothing is
replayed or refitted.

    python plots/query_hist_by_round.py                        # EU, the 20-split run
    python plots/query_hist_by_round.py --targets M9 M13 M16   # just a few panels
    python plots/query_hist_by_round.py --method typiclust --run <run dir>
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import analysis_path, figure_path                             # noqa: E402

LABEL = {"laplace": "EU"}
DEFAULT_RUN = ("results/computer_work/"
               "computer_work_w15_laplace-random_r10x3_s5_20split")

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=DEFAULT_RUN)
ap.add_argument("--method", default="laplace")
ap.add_argument("--counts", action="store_true", help="y as counts instead of % of round")
ap.add_argument("--targets", nargs="+", default=None, help="subset of targets to draw")
ap.add_argument("--ncol", type=int, default=0, help="panels per row (0 = auto)")
ap.add_argument("--out", default=None)
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.run, "queries_split*.csv")))
if not files:
    raise SystemExit(f"no queries_split*.csv under {a.run}")
Q = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
have = sorted(Q.method.unique())
Q = Q[Q.method == a.method]
if Q.empty:
    raise SystemExit(f"method {a.method!r} not in {a.run} (have: {have})")

people = sorted(set(Q.owner) | set(Q.target), key=lambda s: int(s[1:]))
rounds = sorted(Q["round"].unique())
targets = sorted(set(Q.target), key=lambda s: int(s[1:]))
if a.targets:
    targets = [t for t in targets if t in set(a.targets)]
    if not targets:
        raise SystemExit(f"none of {a.targets} is a target in this run")

from palette import distinct_colors                                     # noqa: E402




cmap = dict(zip(people, distinct_colors(len(people))))

G = Q.groupby(["target", "round", "owner"]).size()
ncol = a.ncol or int(np.ceil(np.sqrt(len(targets) * 1.3)))
nrow = int(np.ceil(len(targets) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.05 * ncol, 2.25 * nrow),
                         sharex=True, sharey=True, squeeze=False)
axes = axes.ravel()

tab = []
for ax, t in zip(axes, targets):
    C = (G.loc[t].unstack(fill_value=0)
         .reindex(index=rounds, columns=people, fill_value=0))
    H = C if a.counts else C.div(C.sum(axis=1), axis=0) * 100.0
    bottom = np.zeros(len(rounds))
    for p in people:
        v = H[p].values
        if not v.any():
            continue
        me = p == t
        ax.bar(rounds, v, bottom=bottom, width=0.80, color=cmap[p], label=p,
               edgecolor="black" if me else "white", linewidth=1.0 if me else 0.25,
               zorder=3 if me else 2)
        bottom += v
    sp_t = Q[Q.target == t].is_self.mean() * 100
    ax.set_title(f"{t} — self {sp_t:.0f}%", fontsize=8.5, pad=2.5)
    ax.tick_params(labelsize=6.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.92", lw=0.5)
    H2 = H.copy()
    H2.insert(0, "target", t)
    tab.append(H2.reset_index())

for ax in axes[len(targets):]:
    ax.set_visible(False)
for ax in axes[max(0, len(targets) - ncol):len(targets)]:
    ax.set_xticks(rounds)
    ax.set_xlabel("round", fontsize=7.5)
for i, ax in enumerate(axes[:len(targets)]):
    if i % ncol == 0:
        ax.set_ylabel("labels" if a.counts else "% of round", fontsize=7.5)

name = LABEL.get(a.method, a.method)
handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[p]) for p in people]
fig.legend(handles, people, title="owner of the queried row", fontsize=6.6,
           title_fontsize=7.6, ncol=1, frameon=False, loc="upper left",
           bbox_to_anchor=(0.908, 0.985), handlelength=0.9, handleheight=0.9,
           labelspacing=0.3)
fig.suptitle(f"{name}: whose labels each target bought, by round "
             f"({Q.split.nunique()} splits, {len(Q)} queries; "
             f"the target's own block is outlined in black)", fontsize=12, y=0.997)
fig.tight_layout(rect=(0, 0, 0.9, 0.986))

out = a.out or f"query_hist_by_round_{a.method}.png"
fig.savefig(figure_path(a.run, out), dpi=160, bbox_inches="tight")
pd.concat(tab, ignore_index=True).round(4).to_csv(
    analysis_path(a.run, out.replace(".png", ".csv")), index=False)
print("figure ->", figure_path(a.run, out))
print("table  ->", analysis_path(a.run, out.replace(".png", ".csv")))
