"""Turn loaded ExtraSensory data into the matrices an active-learning run consumes.

Input is whatever ``load_data.load`` returned -- ``{uuid: (X, y, day)}`` plus the column
names. Output is a :class:`Split`: one pooled, standardised training matrix with a row-aligned
owner vector, and per-participant test matrices.

The three things this file does, in the order they must happen:

1. **Blocking** (``block_dataset``). ExtraSensory rows are one minute apart and adjacent minutes
   of the same activity are near-duplicates, so treating them as independent observations
   overstates the amount of information in a label. Averaging consecutive runs of W minutes
   within a day gives blocks that are closer to independent. The label of a block is the
   majority of its minutes. W=15 is the setting the COMPUTER_WORK results use; W=1 leaves the
   data untouched.

2. **Splitting** (``day_group_folds``, ``one_split``). Whole DAYS are held out, never individual
   rows, so adjacent minutes of one activity episode can never straddle the train/test boundary.
   Splitting at row level leaks almost perfectly at this sampling rate. The split is drawn on
   MINUTE rows and each side is blocked afterwards (see the ordering note below).

3. **Standardising** (``build_pool``). Median imputation + z-scoring, fitted on the TRAINING
   rows only and then applied to the test rows. Fitting the scaler on all rows would let test-day
   statistics into the transform.

Ordering matters and is not arbitrary. The split is drawn first, on minute rows, and the train
and test sides are blocked separately afterwards; the scaler is then fitted on the blocked
training rows. Blocking before splitting would be just as leak-free -- blocks never cross a day
boundary -- but it changes the cohort: ``min_test`` would then count BLOCKS rather than minutes,
making the per-participant eligibility test about W times stricter (at W=15 that drops the
COMPUTER_WORK cohort from 43 participants to 38). Keep split-then-block unless you want a
different cohort.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------------- 1. blocking
def block_rows(X, y, day, W):
    """Average consecutive runs of W rows within one day; label = majority, day carried through.

    Runs do not cross a day boundary, and the final run of a day is whatever rows remain, so a
    block can hold fewer than W minutes. Averaging is nan-aware: a feature missing from some
    minutes still yields the mean of the ones present.
    """
    if W <= 1:
        return np.asarray(X, float), np.asarray(y, float), np.asarray(day)
    Xb, yb, db = [], [], []
    for d in np.unique(day):
        i = np.where(day == d)[0]
        for st in range(0, len(i), W):
            j = i[st:st + W]
            with np.errstate(invalid="ignore"):
                Xb.append(np.nanmean(X[j], axis=0))      # all-nan column -> nan, imputed later
            yb.append(float(y[j].mean() >= 0.5))
            db.append(d)
    return np.vstack(Xb), np.array(yb), np.array(db)


def block_dataset(data, W):
    """Apply :func:`block_rows` to every participant. -> {uuid: (X, y, day)} with fewer rows."""
    return {u: block_rows(X, y, d, W) for u, (X, y, d) in data.items()}


# ----------------------------------------------------------------------------- 2. splitting
def day_group_folds(data, users, n_folds=3, test_frac=0.3, seed=0, min_test=5, max_tries=40):
    """Hold out whole days at random. -> ([(fold, {uuid: (Xtr, Xte, ytr, yte)})], kept users).

    A walk-forward split (train on days 1-3, test on day 4) drops most participants for a sparse
    target, because the particular test day often contains no positive minutes at all. Drawing
    the held-out days at random keeps the no-leak property without requiring the activity to fall
    on the days a fixed schedule happens to test. Days are redrawn until both sides carry at
    least `min_test` rows of each class; a participant who never satisfies that in `max_tries`
    draws is dropped from the fold.

    Only participants valid in EVERY fold are kept, so the cohort is identical across folds.
    """
    out = []
    for f in range(n_folds):
        splits = {}
        for ui, u in enumerate(users):
            X, y, d = data[u]
            days = np.unique(d)
            if len(days) < 3:
                continue
            rng = np.random.default_rng([seed, f, ui, 17])
            n_te = max(1, int(round(test_frac * len(days))))
            for _ in range(max_tries):
                te_d = rng.choice(days, min(n_te, len(days) - 1), replace=False)
                te = np.isin(d, te_d)
                tr = ~te
                if (np.bincount(y[te].astype(int), minlength=2).min() >= min_test
                        and np.bincount(y[tr].astype(int), minlength=2).min() >= min_test):
                    splits[u] = (X[tr], X[te], y[tr], y[te])
                    break
        out.append((f, splits))
    keep = [u for u in users if all(u in sp for _, sp in out)]
    return [(f, {u: sp[u] for u in keep}) for f, sp in out], keep


def _row_index_proxy(data, users):
    """The same data with the row INDEX as the only feature, so a split returns its indices."""
    return {u: (np.arange(len(data[u][1]))[:, None], data[u][1], data[u][2]) for u in users}


def split_with_days(data, users, n_folds=3, test_frac=0.3, seed=0, min_test=5):
    """day_group_folds, plus which day each train/test row came from.

    day_group_folds returns arrays rather than indices, so the day of a row cannot be recovered
    from its output. Running the identical split on a matrix whose single feature is the row
    index returns exactly the indices the real split used -- cheaper and safer than
    re-implementing the draw. The assert guards against the two ever diverging.
    """
    folds, keep = day_group_folds(data, users, n_folds, test_frac, seed, min_test)
    pfolds, pkeep = day_group_folds(_row_index_proxy(data, users), users, n_folds, test_frac,
                                    seed, min_test)
    assert pkeep == keep, "row-index proxy split diverged from the real split"
    tr_day, te_day = {}, {}
    for (f, sp), (_, psp) in zip(folds, pfolds):
        tr_day[f] = {u: data[u][2][psp[u][0].ravel()] for u in keep}
        te_day[f] = {u: data[u][2][psp[u][1].ravel()] for u in keep}
    return folds, keep, tr_day, te_day


def one_split(data, users, seed, test_frac=0.3, min_test=5):
    """A single day split keyed by one seed -- the unit a multi-split run iterates over.

    Same machinery as `split_with_days` with n_folds=1, so the seed alone decides which days are
    held out. One number then indexes everything a learning curve has to be averaged over.
    """
    folds, keep, tr_day, te_day = split_with_days(data, users, 1, test_frac, seed, min_test)
    return folds[0][1], keep, tr_day[0], te_day[0]


def canonical_ids(data, n_folds=3, seed=0, test_frac=0.3, min_test=5, prefix="M"):
    """Fixed ``uuid -> M1..Mn`` map over the participants kept in ALL folds of one canonical call.

    Numbering must not depend on which participants a given split happens to keep, or M21 would
    be a different person in every run and no two figures would be comparable. Build this map
    once from a canonical (seed, n_folds) and pass it everywhere.
    """
    _, keep = day_group_folds(data, sorted(data), n_folds, test_frac, seed, min_test)
    return {u: f"{prefix}{i + 1}" for i, u in enumerate(keep)}


# ----------------------------------------------------------------------------- 3. standardising
def make_prep():
    """Median imputation then z-scoring. Fit on training rows only -- never on the test rows."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler())


class Split:
    """One split's matrices, ready for an acquisition function and a model.

    Attributes
    ----------
    Z        (n_train, n_feat) standardised pooled training rows of every participant
    y        (n_train,)        their labels
    owner    (n_train,)        uuid of each training row -- the provenance an AL run records
    pid      (n_train,)        short id (M1..Mn) of each training row, when `ids` was given
    Zte      {uuid: array}     that participant's standardised test rows
    yte      {uuid: array}     their test labels
    prep                       the fitted imputer+scaler, for transforming anything else
    users    list[str]         participants in this split, in a fixed order
    cols     list[str]         feature names, aligned with Z's columns
    """

    def __init__(self, Z, y, owner, Zte, yte, prep, users, cols, ids=None):
        self.Z, self.y, self.owner = Z, y, owner
        self.Zte, self.yte, self.prep = Zte, yte, prep
        self.users, self.cols, self.ids = list(users), list(cols), ids or {}
        self.pid = np.array([self.ids.get(o, o) for o in owner])

    def __repr__(self):
        return (f"<Split {len(self.users)} users, {self.Z.shape[0]} train x {self.Z.shape[1]} "
                f"feat, {self.y.mean():.1%} positive, "
                f"{sum(len(v) for v in self.yte.values())} test rows>")

    def rows_of(self, user):
        """Index of the training rows belonging to one participant."""
        return np.where(self.owner == user)[0]

    def test_of(self, user):
        return self.Zte[user], self.yte[user]


def build_pool(split, users, cols, ids=None):
    """Pool one fold's participants into a standardised training matrix. -> :class:`Split`.

    `split` is the ``{uuid: (Xtr, Xte, ytr, yte)}`` mapping from `day_group_folds`. Every
    participant's training rows are stacked into one pool -- that pool is what a global
    acquisition function queries from, and `owner` is what makes it possible to say afterwards
    whose labels were bought.

    The imputer and scaler are fitted on the pooled TRAINING rows and then applied to each
    participant's test rows, so no test-day statistic enters the transform.
    """
    users = [u for u in users if u in split]
    Xtr = np.vstack([split[u][0] for u in users])
    ytr = np.concatenate([split[u][2] for u in users]).astype(float)
    owner = np.concatenate([[u] * len(split[u][2]) for u in users])
    prep = make_prep().fit(Xtr)
    Z = prep.transform(Xtr)
    Zte = {u: prep.transform(split[u][1]) for u in users}
    yte = {u: np.asarray(split[u][3], float) for u in users}
    return Split(Z, ytr, owner, Zte, yte, prep, users, cols, ids)


# ----------------------------------------------------------------------------- end to end
def _block_split(sp, tr_day, te_day, W):
    """Block each side of one fold's split separately, using that side's own day vector."""
    if W <= 1:
        return sp
    out = {}
    for u, (Xtr, Xte, ytr, yte) in sp.items():
        Xtr_b, ytr_b, _ = block_rows(Xtr, ytr, tr_day[u], W)
        Xte_b, yte_b, _ = block_rows(Xte, yte, te_day[u], W)
        out[u] = (Xtr_b, Xte_b, ytr_b, yte_b)
    return out


def prepare(data, cols, W=15, n_folds=3, seed=0, test_frac=0.3, min_test=5, ids=None,
            verbose=False):
    """Day splits -> per-side blocking -> pooled standardised matrices, for every fold.

    -> (list[Split], ids, days) where `days` is ``{fold: (train_days, test_days)}`` for anything
    that needs to know which day a row came from (seed sets spread over days, day-level
    diagnostics).

    `ids` defaults to the canonical M-id map over the participants kept in every fold at this
    (seed, n_folds). Pass an existing map to keep participant numbering identical to an earlier
    run -- the numbering is positional, so a different cohort renumbers everyone.
    """
    users_all = sorted(data)
    folds, keep, tr_day, te_day = split_with_days(data, users_all, n_folds, test_frac, seed,
                                                  min_test)
    ids = ids or {u: f"M{i + 1}" for i, u in enumerate(keep)}
    splits = [build_pool(_block_split(sp, tr_day[f], te_day[f], W), keep, cols, ids)
              for f, sp in folds]
    if verbose:
        n0 = sum(len(v[1]) for v in data.values())
        n1 = sum(s.Z.shape[0] + sum(len(v) for v in s.yte.values()) for s in splits[:1])
        print(f"W={W}: {n0} minute rows -> {n1} blocks in fold 0; {len(keep)} of "
              f"{len(users_all)} participants kept in all {n_folds} folds")
        for s in splits:
            print(" ", s)
    return splits, ids, {f: (tr_day[f], te_day[f]) for f in tr_day}


def prepare_one(data, cols, seed, W=15, test_frac=0.3, min_test=5, ids=None):
    """The single-split form of :func:`prepare`, keyed by one seed. -> (Split, ids, days)."""
    users_all = sorted(data)
    sp, keep, tr_day, te_day = one_split(data, users_all, seed, test_frac, min_test)
    ids = ids or {u: f"M{i + 1}" for i, u in enumerate(keep)}
    return build_pool(_block_split(sp, tr_day, te_day, W), keep, cols, ids), ids, (tr_day, te_day)


def block_summary(data, blocked):
    """Per-participant row and block counts -- a quick check that blocking did what you meant."""
    rows = []
    for u in sorted(blocked):
        X0, y0, d0 = data[u]
        X1, y1, d1 = blocked[u]
        rows.append(dict(uuid=u, rows=len(y0), blocks=len(y1),
                         pos_rows=int((y0 == 1).sum()), pos_blocks=int((y1 == 1).sum()),
                         pos_rate_rows=float(np.mean(y0)), pos_rate_blocks=float(np.mean(y1)),
                         days=len(np.unique(d1))))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    from load_data import load

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="computer_work")
    ap.add_argument("--block", type=int, default=15, help="W, minutes per block (1 = no blocking)")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    data, cols = load(a.target, verbose=True)
    splits, ids, days = prepare(data, cols, W=a.block, n_folds=a.folds, seed=a.seed, verbose=True)
    print(block_summary(data, block_dataset(data, a.block)).head(10).to_string(index=False))
