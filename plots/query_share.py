"""Under- vs over-querying: whose blocks each acquisition actually buys.

For every participant, compare the share of acquired labels that came from them against their
share of the candidate pool. The pool share is the right baseline because it is what an
unbiased sampler converges to, so

    over/under-query  =  queried share - pool share        (percentage points)

is zero for a method that treats the pool as uniform, positive for a participant the method
favours, negative for one it neglects. Aggregated over every target and split, so a bar reads
"across all 43 personalised models, this method bought from M12 n points more than its size
warrants".

Queries are pooled from every r10x3 run folder, and the pool share is recomputed per split with
the same seed main.py used, so the baseline matches the data the methods actually saw.

    python plots/query_share.py
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
LABEL = {"laplace": "EU", "eer": "eer", "typiclust": "typiclust",
         "coreset": "coreset", "probcover": "probcover", "random": "random"}
POS, NEG = "#2a7f62", "#b5503f"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="computer_work")
    ap.add_argument("--block", type=int, default=15)
    ap.add_argument("--contains", default="r10x3")
    ap.add_argument("--run", default=None, help="run folder the figure is written into")
    ap.add_argument("--out", default="query_share.png")
    a = ap.parse_args()

    root = os.path.join(RESULTS_ROOT, a.target)
    files = sorted(glob.glob(os.path.join(root, f"*{a.contains}*", "queries_split*.csv")))
    if not files:
        raise SystemExit(f"no queries_split*.csv under {root}/*{a.contains}*")
    Q = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # random is written into several run folders by the same seed; keep one copy per
    # (split, target, round, row) so it is not counted once per folder it appears in
    Q = Q.drop_duplicates(["method", "split", "target", "round", "row"])
    splits = sorted(Q.split.unique())
    print(f"{len(files)} file(s), {len(Q)} queries, methods {sorted(Q.method.unique())}")

    # pool composition per split, rebuilt with the seed main.py used (seed=split_id)
    data, cols = load(get_target(a.target), verbose=True)
    ids = canonical_ids(data)
    pool = {}
    for k in splits:
        sp, _, _ = prepare_one(data, cols, seed=k, W=a.block, ids=ids)
        who = pd.Series([sp.ids[u] for u in sp.owner])
        pool[k] = who.value_counts(normalize=True)
        print(f"  split {k}: {len(sp.owner)} pool rows, {who.nunique()} participants")
    P = pd.DataFrame(pool)                       # participant x split, pool share

    # queried share per (method, split, participant), then averaged over splits
    rows = []
    for (m, k), g in Q.groupby(["method", "split"]):
        s = g.owner.value_counts(normalize=True)
        for pid in P.index:
            rows.append(dict(method=m, split=k, pid=pid,
                             delta=100 * (s.get(pid, 0.0) - P.loc[pid, k])))
    G = (pd.DataFrame(rows).pivot_table(index="pid", columns="method", values="delta")
         .reindex(columns=[m for m in ORDER if m in Q.method.unique()]))
    order = P.mean(axis=1).sort_values(ascending=False).index          # biggest pool share first
    G = G.loc[order]

    n = G.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(13, 1.9 * n), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(G))
    for ax, m in zip(axes, G.columns):
        v = G[m].values
        ax.bar(x, v, color=[POS if t > 0 else NEG for t in v], width=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("pp", fontsize=8)
        ax.text(0.004, 0.9, LABEL.get(m, m), transform=ax.transAxes, fontsize=10,
                va="top", fontweight="bold")
        ax.text(0.999, 0.9, f"mean |over/under| {np.abs(v).mean():.2f} pp", ha="right",
                va="top", transform=ax.transAxes, fontsize=8, color="dimgrey")
        ax.tick_params(labelsize=7)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(G.index, rotation=90, fontsize=7)
    axes[-1].set_xlabel("participant, ordered by share of the candidate pool (largest first)",
                        fontsize=9)
    fig.suptitle("queried share - pool share:  over-queried (green) vs under-queried (red), "
                 f"{a.target} W={a.block}, {len(splits)} split(s)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    run = a.run or sorted(glob.glob(os.path.join(root, f"*laplace*{a.contains}*")))[0]
    out = figure_path(run, a.out)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"saved {out}")
    print()
    print("mean |over/under-query| per method, percentage points:")
    print(G.abs().mean().sort_values().round(2).to_string())


if __name__ == "__main__":
    main()
