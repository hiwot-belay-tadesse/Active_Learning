"""Row-level and participant-level attributes of one split's pool -- the vocabulary of the diagnosis.

The diagnosis section needs to say *what kind of blocks* an acquisition buys, which means every
pooled training row has to carry attributes that exist independently of any method. They are
computed once per (target, W, split), cached under ``two_stage/cache/``, and then joined onto
whatever queries a run recorded.

The attributes, and what each is meant to catch
-----------------------------------------------
``knn_dist``        mean Euclidean distance to the row's ``k`` nearest neighbours in the pool.
                    Large = isolated. This is the quantity a k-centre objective (CoreSet)
                    *maximises* by construction, so it is the first place to look for
                    over-querying of anomalies.
``local_purity``    share of those ``k`` neighbours carrying the same label. Low = the row sits
                    where the two classes interleave, i.e. its label is close to unpredictable
                    from its features. Computed with the full labels, so it is a diagnostic and
                    not a signal any learner is allowed to use.
``own_z``           distance to the OWNER's own centroid, in units of that owner's own spread:
                    "how unusual is this block for this participant". This is the
                    subject-specific noise axis -- a block that is ordinary for the cohort but
                    extreme for its owner.
``impurity``        0.5 - |mean(minute labels) - 0.5| for the block, i.e. how mixed the W minutes
                    that were averaged into it were. 0 = a pure block, 0.5 = half the minutes
                    were COMPUTER_WORK and half were not. These are the **phase-shift blocks**:
                    an activity boundary smeared across one 15-minute window, whose features are
                    an average of two regimes and whose majority-vote label is a coin toss.
``is_transition``   the block's label differs from a temporally adjacent block of the same
                    participant on the same day -- the coarser, label-only reading of the same
                    boundary phenomenon.
``pos``             the label itself, so "does this method hunt positives" is answerable.

Participant level
-----------------
``person_isolation``  mean distance from the participant's blocks to the nearest block belonging
                      to *anyone else*. Large = an **outlier individual**: someone whose sensor
                      signature does not resemble the rest of the cohort. A geometric
                      acquisition covering the pool must buy from them repeatedly; whether it
                      should is the equity question.
``pool_share``        the participant's share of the candidate pool -- the baseline a sampler
                      that treats the pool as uniform converges to.

Reconstruction, and why the assert matters
------------------------------------------
``impurity`` needs the minute labels that went into each block, which the ``Split`` object no
longer has. They are recovered by re-running the parent repo's own split machinery on a
row-index proxy, then re-chunking exactly as ``block_rows`` does. That reconstruction is only
valid if it reproduces the block labels the run actually used, so it is asserted against
``sp.y`` row by row -- a silent mis-alignment here would attach every attribute to the wrong
block and the whole diagnosis would be noise with a straight face.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from load_data import get_target, load                                       # noqa: E402
from preprocess_data import (_row_index_proxy, day_group_folds, canonical_ids,  # noqa: E402
                             prepare_one)

CACHE = os.environ.get("TWO_STAGE_CACHE") or os.path.join(HERE, "cache")
__all__ = ["row_attributes", "person_attributes", "load_attributes"]


# --------------------------------------------------------------------------- blocks -> minutes
def _train_minute_index(data, users, seed, test_frac=0.3, min_test=5):
    """-> {uuid: minute-row indices used for TRAINING}, from the parent repo's own split draw."""
    pfolds, pkeep = day_group_folds(_row_index_proxy(data, users), users, 1, test_frac, seed,
                                    min_test)
    _, psp = pfolds[0]
    return {u: psp[u][0].ravel() for u in pkeep}


def _block_label_stats(y_min, d_min, W):
    """-> (block label means, block day) for one participant's training minutes.

    Chunked exactly as ``preprocess_data.block_rows``: days in ascending order, then consecutive
    runs of W rows inside each day, with a short final run allowed.
    """
    means, days = [], []
    for d in np.unique(d_min):
        i = np.where(d_min == d)[0]
        for st in range(0, len(i), W):
            j = i[st:st + W]
            means.append(float(np.mean(y_min[j])))
            days.append(d)
    return np.array(means), np.array(days)


# --------------------------------------------------------------------------- the attributes
def row_attributes(data, cols, split_id, W=15, k=20, verbose=False):
    """-> (DataFrame of per-row attributes, Split) for one split. Row order matches ``sp.Z``."""
    users_all = sorted(data)
    ids = canonical_ids(data)
    sp, ids, (tr_day, _) = prepare_one(data, cols, seed=split_id, W=W, ids=ids)
    tr_idx = _train_minute_index(data, users_all, split_id)

    means, bdays, owners = [], [], []
    for u in sp.users:
        idx = tr_idx[u]
        m, d = _block_label_stats(data[u][1][idx], data[u][2][idx], W)
        means.append(m)
        bdays.append(d)
        owners += [u] * len(m)
    means, bdays = np.concatenate(means), np.concatenate(bdays)
    assert len(means) == len(sp.y), f"reconstructed {len(means)} blocks, split has {len(sp.y)}"
    assert np.array_equal((means >= 0.5).astype(float), sp.y), \
        "minute-label reconstruction does not reproduce the split's block labels"
    assert np.array_equal(np.asarray(owners), sp.owner), "owner order diverged"

    nn = NearestNeighbors(n_neighbors=k + 1).fit(sp.Z)
    dist, nbr = nn.kneighbors(sp.Z)
    knn_dist = dist[:, 1:].mean(axis=1)                       # drop the row itself
    same = (sp.y[nbr[:, 1:]] == sp.y[:, None])
    local_purity = same.mean(axis=1)

    own_z = np.zeros(len(sp.y))
    isolation = {}
    for u in sp.users:
        r = sp.rows_of(u)
        c = sp.Z[r].mean(axis=0)
        d = np.linalg.norm(sp.Z[r] - c, axis=1)
        own_z[r] = (d - d.mean()) / (d.std() + 1e-12)
        other = np.setdiff1d(np.arange(len(sp.y)), r)
        nn_o = NearestNeighbors(n_neighbors=1).fit(sp.Z[other])
        isolation[u] = float(nn_o.kneighbors(sp.Z[r])[0].mean())

    # temporal adjacency within owner+day: the blocks were built in time order, so neighbouring
    # positions in the array are neighbouring windows unless the owner or the day changes
    trans = np.zeros(len(sp.y), bool)
    key = pd.Series([f"{o}|{d}" for o, d in zip(sp.owner, bdays)])
    for _, g in key.groupby(key).groups.items():
        g = np.sort(np.asarray(g))
        if len(g) < 2:
            continue
        yy = sp.y[g]
        diff = np.zeros(len(g), bool)
        diff[:-1] |= yy[:-1] != yy[1:]
        diff[1:] |= yy[1:] != yy[:-1]
        trans[g] = diff

    df = pd.DataFrame(dict(
        split=split_id, row=np.arange(len(sp.y)),
        owner=[ids.get(o, o) for o in sp.owner], pos=sp.y,
        knn_dist=knn_dist, local_purity=local_purity, own_z=own_z,
        impurity=0.5 - np.abs(means - 0.5), is_transition=trans.astype(float),
        day=bdays))
    df["person_isolation"] = [isolation[o] for o in sp.owner]
    if verbose:
        print(f"split {split_id}: {len(df)} blocks, {len(sp.users)} participants, "
              f"{df.pos.mean():.1%} positive, {df.is_transition.mean():.1%} transition blocks, "
              f"mean impurity {df.impurity.mean():.3f}")
    return df, sp


def person_attributes(df):
    """-> per-participant table: pool share, isolation, and mean row attributes."""
    g = df.groupby("owner")
    out = g.agg(n_pool=("row", "size"), pos_rate=("pos", "mean"),
                knn_dist=("knn_dist", "mean"), local_purity=("local_purity", "mean"),
                impurity=("impurity", "mean"), transition=("is_transition", "mean"),
                isolation=("person_isolation", "first")).reset_index()
    out["pool_share"] = out.n_pool / out.n_pool.sum()
    out["split"] = df.split.iloc[0]
    return out


# --------------------------------------------------------------------------- cache
def load_attributes(target="computer_work", splits=(0, 1, 2), W=15, k=20, refresh=False,
                    verbose=True):
    """-> (rows, persons) for several splits, cached as parquet-free CSV.gz under cache/.

    Attributes depend on the target, the block width and the split seed -- not on any run -- so
    one cache serves every method and every run folder that used those settings.
    """
    os.makedirs(CACHE, exist_ok=True)
    t = get_target(target)
    rows, persons, need = [], [], []
    for k_split in splits:
        f = os.path.join(CACHE, f"rows_{t.name}_w{W}_k{k}_split{k_split}.csv.gz")
        if os.path.exists(f) and not refresh:
            df = pd.read_csv(f)
            rows.append(df)
            persons.append(person_attributes(df))
        else:
            need.append((k_split, f))
    if need:
        data, cols = load(t, verbose=verbose)
        for k_split, f in need:
            df, _ = row_attributes(data, cols, k_split, W=W, k=k, verbose=verbose)
            df.to_csv(f, index=False)
            rows.append(df)
            persons.append(person_attributes(df))
    R = pd.concat(rows, ignore_index=True).sort_values(["split", "row"])
    P = pd.concat(persons, ignore_index=True).sort_values(["split", "owner"])
    return R, P


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="computer_work")
    ap.add_argument("--block", type=int, default=15)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--splits", type=int, nargs="+", default=[0])
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    R, P = load_attributes(a.target, a.splits, a.block, a.k, a.refresh)
    print(R.describe().T.to_string())
    print()
    print(P.sort_values("isolation", ascending=False).head(8).to_string(index=False))
