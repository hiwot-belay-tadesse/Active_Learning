"""Why ProbCover collapses onto a few participants here: the covering radius it actually picks.

ProbCover covers the pool with balls of radius ``delta`` and queries the candidate covering the
most still-uncovered points. ``delta`` is chosen as the largest value in a grid whose balls stay
"pure" with respect to a k-means partition of the pool -- and scikit-activeml's default grid is
``np.arange(0.2, 2.2, 0.2)``, i.e. it tops out at 2.0.

That grid is calibrated for the setting the method was published in: self-supervised *embeddings*,
typically L2-normalised, where the interesting radii are O(1). This pool is 124 standardised
sensor features, where the distance between a block and its 20th nearest neighbour is ~5-6 and
the median pairwise distance is far larger. If the chosen radius is small relative to that scale,
the covering objective stops being "cover the pool" and becomes "find the tightest micro-cluster",
which is a density criterion -- and in a multi-subject pool the densest micro-clusters belong to
whichever participants recorded long stationary stretches. Hence queries concentrated on a few
people, which is what the burden numbers show.

This script measures that, rather than asserting it: the delta ProbCover selects, what fraction of
pairs it covers, how many of a block's neighbours fall inside it, and how concentrated the
resulting coverage counts are by participant.

    python two_stage/probcover_delta.py --splits 0 1 2

Note: it builds the same full (n x n) distance matrix ProbCover builds (~0.9 GB at n = 11 000), so
do not run it beside a full benchmark fan-out.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from load_data import get_target, load                                       # noqa: E402
from preprocess_data import canonical_ids, prepare_one                       # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="computer_work")
    ap.add_argument("--block", type=int, default=15)
    ap.add_argument("--splits", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default=os.path.join(HERE, "diagnosis", "probcover_delta.csv"))
    a = ap.parse_args()

    from skactiveml.pool import ProbCover
    t = get_target(a.target)
    data, cols = load(t, verbose=False)
    ids = canonical_ids(data)
    rows = []
    for k in a.splits:
        sp, _, _ = prepare_one(data, cols, seed=k, W=a.block, ids=ids)
        n = len(sp.y)
        y_al = np.full(n, np.nan)
        y_al[:3] = sp.y[:3]                        # a nominal labelled set; delta does not use it
        qs = ProbCover(n_classes=2, random_state=0)
        qs.query(X=sp.Z, y=y_al, batch_size=1)     # fits delta_max_ and distances_
        delta = float(qs.delta_max_)
        D = qs.distances_
        iu = np.triu_indices(n, 1)
        pair = D[iu]
        covered = pair <= delta
        deg = (D <= delta).sum(axis=1) - 1         # neighbours inside the ball, excluding self
        # how concentrated is "coverage" by participant: each participant's share of the total
        # in-ball degree is what a coverage-maximising greedy picks proportionally to
        owners = np.asarray([ids.get(o, o) for o in sp.owner])
        share = pd.Series(deg).groupby(owners).sum()
        share = share / share.sum()
        knn20 = np.sort(D, axis=1)[:, 1:21].mean(axis=1)
        rows.append(dict(split=k, n=n, delta=delta,
                         grid_max=2.0, delta_is_grid_max=bool(abs(delta - 2.0) < 1e-9),
                         median_pair_dist=float(np.median(pair)),
                         mean_knn20_dist=float(knn20.mean()),
                         delta_over_knn20=float(delta / knn20.mean()),
                         pairs_covered=float(covered.mean()),
                         deg_zero_share=float((deg == 0).mean()),
                         deg_median=float(np.median(deg)), deg_max=int(deg.max()),
                         top1_participant_coverage_share=float(share.max()),
                         top3_participant_coverage_share=float(share.nlargest(3).sum()),
                         n_participants=len(share)))
        print(f"split {k}: delta = {delta:.2f} (grid tops out at 2.0), "
              f"mean 20-NN distance {knn20.mean():.2f}, "
              f"{covered.mean() * 100:.3f}% of pairs within delta, "
              f"{float((deg == 0).mean()) * 100:.1f}% of blocks have an EMPTY ball, "
              f"top-3 participants hold {share.nlargest(3).sum():.1%} of all in-ball coverage",
              flush=True)
        del D, qs
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\n-> {a.out}")
    print(df.mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()
