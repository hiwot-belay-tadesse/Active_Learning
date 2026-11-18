"""Per-participant active learning with plain logistic regression -- everything is the target's.

The pooled experiment in ``main.py`` asks whose labels a personalised model should buy, and lets
it buy from anyone. This script is the opposite corner of that design and the baseline it should
be read against: one participant, one model, one pool.

    seed      ``n_seed`` of the TARGET's own labels (``n_seed_pos`` positive), spread over days
    pool      the TARGET's own unlabelled training blocks -- no other participant is a candidate
    model     sklearn LogisticRegression on that participant's labelled rows alone
    test      the TARGET's held-out rows, the same ones main.py scores on

so there is no provenance question to ask: every bought label is the target's own, and the
``owner`` column of ``queries_split<k>.csv`` is constant. What the figures can still show is the
learning curve per participant, and how it compares with the pooled arms at the same budget.

Acquisitions (``--methods``), all restricted to the target's own rows:

    random        uniform over the unlabelled blocks -- the control
    uncertainty   scikit-activeml's UncertaintySampling over the current fit
                  (``--uncertainty_method``, default least_confident; with two classes
                  least_confident, margin_sampling and entropy induce the SAME ordering, so
                  the choice only matters if the task becomes multi-class)
    coreset       k-centre greedy (Sener & Savarese 2018) over the target's own blocks
    typiclust     dense-region typicality (Hacohen et al. 2022), k nearest neighbours
    probcover     probability cover, delta from clustering purity (n_classes=2)

The three representation strategies are label-free: they read the geometry of the pool, never
the model. Here that pool is ONE participant's blocks -- a few hundred rows rather than ~11000 --
so their clusterings describe that person's own feature space. That is the intended contrast
with the pooled run, where the same strategies cluster everybody at once, but it also means
their neighbourhood parameters (TypiClust's k, ProbCover's delta) are being applied at a very
different scale; a null result there is as likely to be the scale as the method.

Two deliberate differences from ``main.py``:

* **No person column and no random effects.** ``main.py`` hands the classifier an owner-coded
  design matrix; here the model never sees anyone else, so the person code would be a constant.
* **No second standardisation.** ``prepare_one`` already imputes and standardises against the
  pooled training rows, and refitting a scaler on 3-30 labelled rows would rescale each round
  from a handful of points. The features go in as the split produced them.

Written to the same schema as ``main.py`` -- ``per_round_split<k>.csv`` and
``queries_split<k>.csv`` with identical columns -- so ``plots/per_person_grid.py`` can draw it.
Pass ``--no_siblings`` there: the configuration gate compares mixed-effects settings this run
does not have, so the merge with pooled runs is not meaningful, only the side-by-side reading.

    python personal_al.py --methods uncertainty random --splits 3
    python personal_al.py --methods uncertainty random --first_split 0 --splits 1 --resume
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from load_data import get_target, load
from main import build_strategy
from paths import done_splits, run_dir, split_files, update_index, write_config
from preprocess_data import canonical_ids, prepare_one
from utility import block_days, scores, seed_set

# Representation strategies come from scikit-activeml, built by main.build_strategy so the two
# experiments cannot drift apart: `coreset` here is the same CoreSet with the same defaults as
# the pooled run, only the pool it sees is different.
METHODS = ["uncertainty", "coreset", "typiclust", "probcover", "random"]
# Every arm but `random` is a scikit-activeml strategy. `uncertainty` additionally needs the
# classifier, wrapped in SklearnClassifier so the library can read the NaN-coded label vector
# and refit on the labelled rows itself.
SKACTIVEML = {"coreset", "typiclust", "probcover"}
NEEDS_CLF = {"uncertainty"}


def fit_lr(Z, y, args):
    """-> (decision_function callable, fit_failed).

    A labelled set with one class cannot train a classifier. Rather than abort the target -- the
    seed is built to hold a positive, but a participant with too few positives is flagged, not
    skipped -- the round scores a constant predictor, which AUROC reads as 0.5 and AUPRC as the
    test prevalence. ``fit_failed`` marks those rows so they can be excluded later.
    """
    if len(np.unique(y)) < 2:
        return (lambda X: np.zeros(len(X))), True
    clf = LogisticRegression(C=args.C, max_iter=args.max_iter,
                             class_weight=args.class_weight or None)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(Z, y)
    return clf.decision_function, False


def pick(method, rng, Z, own, y_own, cand, batch, qs, skclf):
    """-> the rows to buy this round, as indices into the full matrix.

    Every strategy is confined to the target: it is handed ``Z[own]`` and the label vector over
    those rows only, so the clustering, the covering radius, the "already covered" set and the
    classifier the uncertainty rule reads are all computed inside this participant's data.
    Indices come back relative to ``own`` and are mapped out again here.

    `uncertainty` passes ``fit_clf=True``: the strategy refits the wrapped classifier on the
    labelled rows itself, so the ranking is never taken from a stale model. A labelled set with
    one class leaves SklearnClassifier predicting the prior, which makes every candidate equally
    uncertain and the pick a tie-break -- the library's own behaviour, not a special case here.
    """
    batch = min(batch, len(cand))
    if method == "random":
        return rng.choice(cand, size=batch, replace=False)
    if method in NEEDS_CLF:
        loc = qs.query(X=Z[own], y=y_own, clf=skclf, fit_clf=True, batch_size=batch)
    elif method in SKACTIVEML:
        loc = qs.query(X=Z[own], y=y_own, batch_size=batch)
    else:
        raise SystemExit(f"unknown method {method!r}; choose from {METHODS}")
    return own[np.asarray(loc, dtype=int)]


def build_query(method, seed, args):
    """-> (query strategy, wrapped classifier or None) for one (target, method).

    Built per target rather than once per run: CoreSet and ProbCover carry state about what is
    already covered, so one object shared across participants would leak one person's coverage
    into the next.
    """
    if method == "random":
        return None, None
    if method in NEEDS_CLF:
        from skactiveml.classifier import SklearnClassifier         # noqa: PLC0415
        from skactiveml.pool import UncertaintySampling             # noqa: PLC0415
        skclf = SklearnClassifier(
            LogisticRegression(C=args.C, max_iter=args.max_iter,
                               class_weight=args.class_weight or None),
            classes=[0.0, 1.0], missing_label=np.nan, random_state=seed)
        return UncertaintySampling(method=args.uncertainty_method, random_state=seed), skclf
    return build_strategy(method, seed), None


def run_target(sp, uuid, tr_day, args, split_id):
    """One participant's learning curves, every method from the same seed. -> (rows, queries)."""
    pid = sp.ids[uuid]
    own = sp.rows_of(uuid)
    rng = np.random.default_rng([args.seed, split_id, int(pid[1:]), 11])
    day = block_days(tr_day[uuid], args.block)
    assert len(day) == len(own), f"{len(day)} block days for {len(own)} blocks of {pid}"
    loc, flagged = seed_set(sp.y[own], day, rng, args.n_seed, args.n_seed_pos)
    seed_rows = own[loc]

    Zte, yte = sp.test_of(uuid)
    n_pos_te = int(np.sum(yte == 1))
    # Keyed on the target as well as the split, so `random` does not replay one stream for
    # every participant, and not drawn from `rng` above, so a method's position in --methods
    # cannot change what it buys.
    strat_seed = int(np.random.default_rng(
        [args.seed, split_id, int(pid[1:]), 7]).integers(2 ** 31))

    rows, queries = [], []
    for method in args.methods:
        qrng = np.random.default_rng(strat_seed)
        # One strategy object per (target, method): CoreSet and ProbCover keep state about what
        # is already covered, so sharing one across targets would leak one participant's
        # coverage into the next.
        qs, skclf = build_query(method, strat_seed, args)
        labelled = list(seed_rows)
        bought = []
        decision, failed = fit_lr(sp.Z[labelled], sp.y[labelled], args)
        for rnd in range(args.rounds + 1):
            if rnd > 0:
                cand = np.setdiff1d(own, labelled)
                if len(cand) == 0:                  # this participant is exhausted
                    break
                # the label vector the skactiveml strategies see: this participant's rows only,
                # NaN where unlabelled
                y_own = np.where(np.isin(own, labelled), sp.y[own], np.nan)
                idx = pick(method, qrng, sp.Z, own, y_own, cand, args.batch, qs, skclf)
                labelled += [int(i) for i in idx]
                bought += [int(i) for i in idx]
                for i in idx:
                    queries.append(dict(split=split_id, target=pid, method=method, round=rnd,
                                        row=int(i), owner=pid, is_self=True))
                decision, failed = fit_lr(sp.Z[labelled], sp.y[labelled], args)
            auroc, auprc = scores(yte, decision(Zte))
            rows.append(dict(split=split_id, target=pid, method=method, round=rnd,
                             n_labeled=len(labelled), acquired=len(labelled) - len(seed_rows),
                             auroc=auroc, auprc=auprc, n_pos_test=n_pos_te,
                             self_share=1.0 if bought else np.nan,
                             fit_failed=bool(failed), seed_flagged=bool(flagged)))
    return rows, queries


def run_split(data, cols, split_id, args, ids=None):
    t0 = time.time()
    sp, ids, (tr_day, _) = prepare_one(data, cols, seed=split_id, W=args.block, ids=ids)
    targets = [u for u in sp.users if np.sum(sp.yte[u] == 1) >= args.min_test_pos]
    print(f"  split {split_id}: {sp}, {len(targets)} targets", flush=True)

    rows, queries = [], []
    for n, u in enumerate(targets, 1):
        r, q = run_target(sp, u, tr_day, args, split_id)
        rows += r
        queries += q
        if n % 10 == 0 or n == len(targets):
            print(f"    {n}/{len(targets)} targets, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(queries), ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="computer_work")
    ap.add_argument("--block", type=int, default=15)
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--first_split", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--batch", type=int, default=3, help="labels bought per round")
    ap.add_argument("--n_seed", type=int, default=3, help="the target's own initial labels")
    ap.add_argument("--n_seed_pos", type=int, default=1, help="how many of them are positive")
    ap.add_argument("--uncertainty_method", default="least_confident",
                    choices=["least_confident", "margin_sampling", "entropy"],
                    help="scikit-activeml's UncertaintySampling rule. All three agree on a "
                         "binary task; the option exists so the choice is recorded in "
                         "config.json rather than implied.")
    ap.add_argument("--C", type=float, default=1.0, help="inverse L2 strength")
    ap.add_argument("--max_iter", type=int, default=1000)
    ap.add_argument("--class_weight", default=None, choices=[None, "balanced"],
                    help="'balanced' reweights by class frequency; off by default, since it "
                         "changes the operating point and AUROC is threshold-free anyway")
    ap.add_argument("--min_test_pos", type=int, default=1,
                    help="skip a target with fewer positive test rows than this")
    ap.add_argument("--max_targets", type=int, default=0, help="debug: stop after N targets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="suffix for the folder name")
    ap.add_argument("--name", default=None,
                    help="the folder name outright, replacing <target>_<splits>_splits")
    ap.add_argument("--resume", action="store_true", help="skip splits already written")
    ap.add_argument("--no_index", action="store_true")
    ap.add_argument("--out", default=None, help="override the derived folder")
    args = ap.parse_args()

    t = get_target(args.target)
    args.target = t.name
    # A short, readable folder name instead of the derived one. The personal experiments all
    # share a model and a pool, so the acquisition list and the block/round/slope settings do
    # not need to be in the name to tell them apart -- config.json carries all of it anyway.
    #
    # The price, and it is a real one: the split count IS in the name, while the derived names
    # deliberately leave it out because splits accumulate inside a run. So `--splits 3` then
    # `--splits 20` writes TWO folders rather than resuming one, and two different settings run
    # with the same --splits land in the same folder and merge. Pass --name (or --tag) when
    # that matters.
    name = args.name or f"{t.name}_{args.splits}_splits" + (f"_{args.tag}" if args.tag else "")
    out = args.out or run_dir(t.name, args.block, args.methods, args.rounds, args.batch,
                              0, scope="personal", name=name)
    os.makedirs(out, exist_ok=True)
    cfg = write_config(out, args, extra=dict(model="sklearn_logistic", pool="target_only",
                                             seed_source="self"))
    print(f"{t.name}: personal LR, {' vs '.join(args.methods)}, W={args.block}, "
          f"{args.n_seed} own seed + {args.rounds} x {args.batch} = "
          f"{args.rounds * args.batch} acquired, {args.splits} split(s)")
    print(f"  -> {out}", flush=True)

    data, cols = load(t, verbose=True)
    have = done_splits(out) if args.resume else []
    if have:
        print(f"  resuming: splits {have} already written", flush=True)
    ids = canonical_ids(data)
    for k in range(args.first_split, args.first_split + args.splits):
        if k in have:
            continue
        R, Q, ids = run_split(data, cols, k, args, ids)
        if args.max_targets:
            keep = list(dict.fromkeys(R.target))[:args.max_targets]
            R, Q = R[R.target.isin(keep)], Q[Q.target.isin(keep)]
        f_round, f_query = split_files(out, k)
        R.to_csv(f_round, index=False)
        Q.to_csv(f_query, index=False)
        if not args.no_index:
            update_index(out, cfg)

        last = R[R["round"] == R["round"].max()]
        w = last.pivot_table(index="target", columns="method", values="auroc")
        alt = next((m for m in args.methods if m != "random"), None)
        if alt and "random" in w and alt in w:
            gap = (w[alt] - w["random"]).dropna()
            print(f"  split {k}: {alt}-random {gap.mean():+.4f} over {len(gap)} targets",
                  flush=True)
        else:
            print(f"  split {k}: written", flush=True)


if __name__ == "__main__":
    main()
