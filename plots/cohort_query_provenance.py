"""Where did a COHORT run's purchases come from, round by round?

cohort_al.py buys one stream of labels for the whole fleet, so there is no target and the
per-target provenance figures do not apply. The question that replaces "did it buy from the
target?" is "whose labels did the fleet's budget go to, and did that change as the model
learned?"

ONE PANEL PER METHOD, one stacked bar per round: the labels bought in that round, split by the
participant who OWNED each row, normalised to 100% so a panel is composition, not volume.
Colour = owner, the same colour in every panel and the same colour query_hist_by_round.py
gives that person. A colour running the width of a panel is somebody bought from throughout;
a colour that fills the left and vanishes is somebody the criterion exhausted its interest in.

The seed (``n_seed_per_person`` blocks from EVERY participant) is not drawn: it is round 0, it
is identical across methods, and it is not a choice.

``--style shares`` draws the other view instead -- purchases per participant over the whole
run, against the grey step of their pool share, with the top-5 share and Gini of each.

    python plots/cohort_query_provenance.py --run results/computer_work/full_global/<run>
    python plots/cohort_query_provenance.py --run <run> --style shares
"""
import argparse
import glob
import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from palette import bright_colors, distinct_colors, fixed_colors         # noqa: E402
from paths import figure_path                                            # noqa: E402

LABEL = {"laplace_pool": "EU (cohort, per row)", "laplace_people": "EU (cohort, per person)"}
COLOR = {"laplace_pool": "teal", "laplace_people": "saddlebrown", "coreset": "seagreen",
         "typiclust": "purple", "probcover": "dodgerblue", "random": "darkorange"}
ORDER = ["laplace_pool", "laplace_people", "coreset", "typiclust", "probcover", "random"]


def gini(x):
    """-> Gini of a share vector; 0 = everyone equal, -> 1 = one person takes everything."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def pool_shares(run, cfg, splits):
    """-> each participant's mean share of the TRAIN rows over `splits` (the `shares` style)."""
    from load_data import get_target, load
    from preprocess_data import canonical_ids, prepare_one
    data, cols = load(get_target(cfg["target"]), verbose=False)
    ids, out = canonical_ids(data), {}
    for k in splits:
        sp, ids, _ = prepare_one(data, cols, seed=int(k), W=cfg["block"], ids=ids)
        n = pd.Series(sp.owner).map(sp.ids).value_counts()
        out[k] = n / n.sum()
    return pd.DataFrame(out).mean(axis=1)


ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--run", required=True)
ap.add_argument("--methods", nargs="+", default=None)
ap.add_argument("--style", default="rounds", choices=["rounds", "shares"])
ap.add_argument("--top_per_bar", type=int, default=3,
                help="rounds: draw only the N most-bought-from participants OF EACH BAR, named "
                     "on the bar. 0 = draw everyone and rely on --top instead")
ap.add_argument("--top", type=int, default=12,
                help="rounds: draw only the N most-queried participants, in mutually "
                     "distinguishable colours and correspondingly wider bars. 0 = everyone, "
                     "which forces a 43-colour palette that is unavoidably shades")
ap.add_argument("--accum", default="cumulative", choices=["cumulative", "per_round"],
                help="rounds: bar height is the labels bought from that person SO FAR, or only "
                     "in that round")
ap.add_argument("--panel_w", type=float, default=8.2,
                help="inches per panel; the bars split whatever width this leaves")
ap.add_argument("--gap", type=float, default=0.12,
                help="rounds/grouped: blank fraction of a round's slot, so the groups separate")
ap.add_argument("--layout", default="grouped", choices=["grouped", "stacked"],
                help="rounds: one bar per person side by side within the round, or one stack")
ap.add_argument("--y", default="count", choices=["count", "share"],
                help="rounds: bar height is the labels bought, or their share of the bar")
ap.add_argument("--split", type=int, default=None,
                help="rounds: which split to draw (default: the first). Pooling splits stacks "
                     "several draws of the same round into one bar, which is a different "
                     "quantity -- use --all_splits to ask for it on purpose")
ap.add_argument("--all_splits", action="store_true", help="rounds: pool every split")
ap.add_argument("--max_rounds", type=int, default=10, help="rounds: first N rounds (0 = all)")
ap.add_argument("--rounds_per_bar", type=int, default=2,
                help="rounds: consecutive rounds pooled into one bar. 1 = a bar per round; "
                     "more widens every bar, which is what makes 43 participants resolvable")
ap.add_argument("--ncol", type=int, default=3)
ap.add_argument("--out", default=None)
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.run, "queries_split*.csv")))
if not files:
    raise SystemExit(f"no queries_split*.csv under {a.run}")
Q = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
have = sorted(Q.method.unique())
methods = a.methods or [m for m in ORDER if m in have] + [m for m in have if m not in ORDER]
unknown = set(methods) - set(have)
if unknown:
    raise SystemExit(f"not in this run: {sorted(unknown)}; have {have}")
cfg = json.load(open(os.path.join(a.run, "config.json")))
splits = sorted(Q.split.unique())
people = sorted(set(Q.owner), key=lambda s: int(s[1:]))

nrow = int(np.ceil(len(methods) / a.ncol))
fig, axes = plt.subplots(nrow, a.ncol, figsize=(a.panel_w * a.ncol, 3.3 * nrow),
                         sharex=True, sharey=True, squeeze=False)

if a.style == "rounds":
    if not a.all_splits:
        k = splits[0] if a.split is None else a.split
        if k not in splits:
            raise SystemExit(f"split {k} not in this run (have {splits})")
        Q, splits = Q[Q.split == k], [k]
    rounds = sorted(Q["round"].unique())
    if a.max_rounds:
        rounds = rounds[:a.max_rounds]
        Q = Q[Q["round"].isin(rounds)]
    # Optional pooling of consecutive rounds. At batch x splits purchases a single round is
    # largely sampling noise, so a long run is easier to read a few rounds to the bar; the
    # default leaves each round its own bar, which is what a 10-round budget wants.
    per_bin = max(1, a.rounds_per_bar)
    bin_of = {r: i // per_bin for i, r in enumerate(rounds)}
    nbin = max(bin_of.values()) + 1
    Q = Q.assign(bin=Q["round"].map(bin_of))
    ticks = []
    for b in range(nbin):
        lo, hi = rounds[per_bin * b], rounds[min(per_bin * (b + 1), len(rounds)) - 1]
        ticks.append(str(lo) if lo == hi else f"{lo}-{hi}")

    # Panel stats describe the whole cohort, so they are taken before anything is dropped or
    # greyed: "37 people, Gini 0.31" has to describe the run, not the figure.
    stats = {}
    for m in methods:
        c = Q[Q.method == m].owner.value_counts()
        stats[m] = (len(c), gini((c / c.sum()).values))

    # Height matrices first, for every panel, because the colours depend on who turns out to be
    # a leader anywhere -- a per-panel palette would give one colour to several people.
    HS, leaders = {}, {}
    for m in methods:
        C = (Q[Q.method == m].groupby(["bin", "owner"]).size()
             .unstack("owner").reindex(index=range(nbin), columns=people).fillna(0.0))
        if a.accum == "cumulative":
            C = C.cumsum(axis=0)
        HS[m] = C if a.y == "count" else C.div(C.sum(axis=1), axis=0)
        if a.top_per_bar:
            for b in range(nbin):
                v = HS[m].iloc[b]
                leaders[(m, b)] = list(v[v > 0].sort_values(ascending=False)
                                       .head(a.top_per_bar).index)

    if a.top_per_bar:
        # Only the bars that lead some panel in some round need a colour; everyone else is grey,
        # so the palette is asked to separate ~15 people rather than 43 and can stay distinct.
        lead_all = sorted({q for v in leaders.values() for q in v}, key=lambda t: int(t[1:]))
        # Saturated hues only: the point of the colour is to stand out from the grey tail, and
        # the general palette's olives and dusty pinks sit too close to 0.78 grey to do that.
        cmap = dict(zip(lead_all, bright_colors(len(lead_all))))
        covered = len(Q[Q.owner.isin(lead_all)]) / len(Q)
    elif a.top:
        keep = [p for p in people if p in set(Q.owner.value_counts().head(a.top).index)]
        cmap = dict(zip(keep, fixed_colors(len(keep))))
        covered = len(Q[Q.owner.isin(keep)]) / len(Q)
        people = keep
    else:
        cmap = dict(zip(people, distinct_colors(len(people))))
        covered = 1.0

    drawn = set()
    for ax, m in zip(axes.ravel(), methods):
        H = HS[m]
        # Slots only for the people this method ever bought from: a participant it never
        # touched would otherwise hold an empty column in every group and steal width from the
        # bars that have something to show.
        seen = [q for q in people if H[q].max() > 0]
        drawn |= set(seen)
        if a.layout == "stacked":
            bot = np.zeros(nbin)
            for p_ in seen:
                v = H[p_].values
                # a white hairline between segments so one person's block is countable, not a
                # smooth gradient of neighbouring hues
                ax.bar(np.arange(nbin), v, bottom=bot, width=0.86, color=cmap.get(p_, "0.84"),
                       edgecolor="white", linewidth=0.5)
                bot += v
        else:
            # Bars touch inside a round and the slot leaves `--gap` blank around it, so the
            # eye groups by round; separating the bars too would make the two gaps compete.
            slot = max(0.1, 1.0 - a.gap)
            # Cumulative: every participant keeps the SAME x position inside every group, so a
            # person's bar stays put and grows; re-sorting each round would make the growth
            # impossible to follow. Per-round: only the people bought from that round get a
            # slice, and the group's width reads as how far the batch was spread.
            fixed = a.accum == "cumulative"
            for b in range(nbin):
                v = H.iloc[b]
                hit = seen if fixed else [q for q in seen if v[q] > 0]
                if not len(hit):
                    continue
                # the bar's own leaders, recomputed for every bar, so a late riser lights up and
                # an early favourite goes grey again
                lead = set(leaders[(m, b)]) if a.top_per_bar else set(hit)
                w = slot / len(hit)
                xs = b - slot / 2 + w * (np.arange(len(hit)) + 0.5)
                ax.bar(xs, [v[q] for q in hit], width=w,
                       color=[cmap[q] if q in lead else "0.84" for q in hit], linewidth=0)
        npeople, g = stats[m]
        ax.set_title(f"{LABEL.get(m, m)}   {npeople} people, Gini {g:.2f}", fontsize=11)
        ax.set_xlim(-0.6, nbin - 0.4)
        ax.set_xticks(np.arange(nbin))
        ax.set_xticklabels(ticks, fontsize=9)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="0.85", lw=0.6)
    ylab = "labels bought" if a.y == "count" else "share of the bar's purchases"
    if a.y == "share":
        axes[0, 0].set_ylim(0, 1)
    elif a.layout == "grouped":
        _c = Q.groupby(["method", "bin", "owner"]).size()
        if a.accum == "cumulative":
            _c = _c.unstack("bin").fillna(0.0).cumsum(axis=1).stack()
        top = int(max(1, _c.max()))
        axes[0, 0].set_ylim(0, top + 0.4)
        axes[0, 0].set_yticks(np.arange(top + 1))
    batch = int(cfg["batch"]) * len(splits)
    who = f"split {splits[0]}" if len(splits) == 1 else f"{len(splits)} splits pooled"
    sub = (f"{cfg['target']}: who the cohort model bought from, round by round -- "
           f"{batch * per_bin} labels per bar"
           f"{', cumulative' if a.accum == 'cumulative' else ''}, {who}, "
           f"{cfg['n_seed_per_person']}/participant seed (round 0) not shown"
           + (f" -- the top {a.top_per_bar} of each bar in colour ({covered:.0%} of the "
              f"budget), everyone else grey" if a.top_per_bar else
              f" -- the {len(people)} most-queried participants, {covered:.0%} of the budget"
              if a.top else ""))
    people_leg = (lead_all if a.top_per_bar else [p for p in people if p in drawn])
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[p]) for p in people_leg]
    fig.legend(handles, people_leg, loc="lower center",
               ncol=min(22, max(1, len(people_leg))), fontsize=7, frameon=False,
               handlelength=1.0, columnspacing=0.8, handletextpad=0.35,
               bbox_to_anchor=(0.5, 0.0))
    rect = (0, 0.055, 1, 0.945)
else:
    P = pool_shares(a.run, cfg, splits).sort_values(ascending=False)
    people = list(P.index)
    x = np.arange(len(people))
    for ax, m in zip(axes.ravel(), methods):
        cnt = (Q[Q.method == m].groupby(["split", "owner"]).size()
               .unstack("split").reindex(people).fillna(0.0))
        share = (cnt / cnt.sum(axis=0)).mean(axis=1)
        ax.bar(x, share.values, width=0.82, color=COLOR.get(m, "grey"), label="query share")
        ax.step(np.r_[x - 0.5, x[-1] + 0.5], np.r_[P.values, P.values[-1]], where="post",
                color="0.35", lw=1.3, label="pool share")
        ax.set_title(f"{LABEL.get(m, m)}   top-5 "
                     f"{share.sort_values(ascending=False).head(5).sum():.0%}, "
                     f"Gini {gini(share.values):.2f}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(people, rotation=90, fontsize=5.5)
        ax.margins(x=0.01)
    ylab = "share of purchases"
    axes[0, 0].legend(fontsize=9, loc="upper right", frameon=False)
    n_buy = int(Q[Q.method == methods[0]].groupby("split").size().mean())
    sub = (f"{cfg['target']}: who the cohort model bought labels from -- {n_buy} purchases "
           f"per split over {len(splits)} split(s), {cfg['n_seed_per_person']}/participant "
           f"seed excluded (pool-share Gini {gini(P.values):.2f})")
    rect = (0, 0, 1, 0.95)

for ax in axes.ravel()[len(methods):]:
    ax.axis("off")
# A panel sitting above a blank cell keeps its own x axis: sharex hands the tick labels to the
# bottom row, which for that column does not exist.
for j in range(a.ncol):
    col = [i for i in range(nrow) if i * a.ncol + j < len(methods)]
    if col:
        last = axes[col[-1], j]
        last.tick_params(labelbottom=True)
        last.set_xlabel("AL round" if a.style == "rounds" else "")
for ax in axes[:, 0]:
    ax.set_ylabel(ylab)

fig.suptitle(textwrap.fill(sub, width=max(70, 46 * a.ncol)), fontsize=12)
fig.tight_layout(rect=rect)
out = figure_path(a.run, a.out or f"cohort_query_provenance_{a.style}.png")
fig.savefig(out, dpi=170)
print(f"saved {out}")
