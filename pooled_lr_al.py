"""Per-target active learning with plain logistic regression over the WHOLE cohort's pool.

The missing cell of the 2x2. The three existing designs vary model class and pool together, so
no two of them isolate either one:

    personal_al.py   sklearn LR          pool = the target's own rows
    main.py          mixed effects       pool = everyone, one model per target
    cohort_al.py     mixed effects       pool = everyone, ONE model for the cohort

This script is sklearn LR with main.py's pool, which makes `pooled_lr_al.py` vs `main.py` a
clean model-class contrast and `pooled_lr_al.py` vs `personal_al.py` a clean POOL contrast:
same model class, same seed, same budget, and the only thing left differing is who the
acquisition is allowed to buy from.

    seed      ``n_seed`` of the TARGET's own labels (``n_seed_pos`` positive), spread over days
    pool      every participant's unlabelled training blocks -- the target included
    model     one sklearn LogisticRegression per target, on the labelled rows whoever owns them
    test      the target's held-out days, the same rows main.py and personal_al.py score on

Seeding is personal_al.py's, row for row: `utility.seed_set` over the target's own rows, same
defaults, same RNG recipe (``[seed, split, pid, 11]``), against the same
``prepare_one(seed=split)`` division. For a given (split, target) the two runs therefore start
from the SAME row indices, so their difference is paired per target rather than merely drawn
from the same distribution. Seeding from the whole cohort instead -- which this script used to
do -- handed it 172 labels where personal_al.py gets 3, and since the figures plot against
labels ACQUIRED, that head start sat invisibly beneath every point of the curve.

The model has no person column and no random effects, so nothing in it is personalised: two
targets differ only in which labels their acquisition chose to buy. That is the point -- it
measures what the pool alone is worth, with personalisation held at zero.

Acquisitions (``--methods``):

    random        uniform over the unlabelled pool -- the control
    laplace       A-optimality: the candidate whose label most reduces the total predictive
                  variance on the TARGET's unlabelled rows, under the Laplace approximation of
                  the logistic likelihood. The analogue of main.py's `laplace`, re-derived for
                  a plain LR, and the only arm here that knows which target it is working for.
    uncertainty   scikit-activeml's UncertaintySampling over the current fit, whole pool
    coreset       k-centre greedy over everybody's blocks
    typiclust     dense-region typicality, k nearest neighbours
    probcover     probability cover, delta from clustering purity

The three representation strategies are label-free and target-blind: handed the same pool, they
return the same batch whatever the target. Their curves therefore differ across targets only
through the test set, which makes them a useful null -- any spread they show is the spread a
target-blind acquisition produces for free.

Each split also writes ``analysis/reference_auroc_split<k>.csv``: `pooled_full`, this same LR
on 100% of the labels, scored per target -- the matched ceiling. ``--no_refs`` skips it.

Written to main.py's schema -- ``per_round_split<k>.csv`` and ``queries_split<k>.csv`` with the
same columns, including ``owner`` and ``is_self`` -- so the provenance figures work unchanged.
Results land under ``results/<target>/global_lr/``, a scope of their own, which also stops
plots/per_person_grid.py merging them with the mixed-effects runs: its sibling search never
leaves the scope folder.

    python pooled_lr_al.py --methods laplace random --splits 3
    python pooled_lr_al.py --methods laplace random --first_split 0 --splits 1 --resume
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from load_data import get_target, load
from main import build_strategy
from paths import (analysis_path, done_splits, run_dir, split_files, update_index,
                   write_config)
from preprocess_data import canonical_ids, prepare_one
from utility import block_days, scores, seed_set

METHODS = ["laplace", "uncertainty", "coreset", "typiclust", "probcover", "random"]
SKACTIVEML = {"coreset", "typiclust", "probcover"}
NEEDS_CLF = {"uncertainty"}


def fit_lr(Z, y, args):
    """-> (fitted estimator or None, fit_failed). One class cannot train a classifier."""
    if len(np.unique(y)) < 2:
        return None, True
    clf = LogisticRegression(C=args.C, max_iter=args.max_iter,
                             class_weight=args.class_weight or None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(Z, y)
    return clf, False


def decide(clf, Z):
    """-> decision values, or zeros when no model could be fitted."""
    return np.zeros(len(Z)) if clf is None else clf.decision_function(Z)


def full_reference(sp, targets, args, split_id):
    """-> one row per target: `fit_lr` on 100% of the labels, scored on the same test rows.

    Pooled LR has nothing target-specific, so full labels give ONE fit, one number per target.
    """
    clf, failed = fit_lr(sp.Z, sp.y, args)
    rows = []
    for u in targets:
        Zte, yte = sp.test_of(u)
        auroc, auprc = scores(yte, decide(clf, Zte))
        rows.append(dict(split=split_id, target=sp.ids[u], model="pooled_full",
                         auroc=auroc, auprc=auprc, n_test=len(yte),
                         n_pos_test=int(np.sum(yte == 1)), n_train=int(len(sp.y)),
                         C=args.C, fit_failed=bool(failed)))
    return pd.DataFrame(rows)


def refs_for_split(data, cols, split_id, args, ids=None):
    """-> (reference frame, ids) for one split, fitting nothing else.

    What --refs_only uses to add the reference to a run whose curves are already on disk.
    """
    sp, ids, _ = prepare_one(data, cols, seed=split_id, W=args.block, ids=ids)
    targets = [u for u in sp.users if np.sum(sp.yte[u] == 1) >= args.min_test_pos]
    return full_reference(sp, targets, args, split_id), ids


def laplace_pick(clf, Z, labelled, cand, eval_rows, batch, args):
    """-> `batch` candidate rows, greedily minimising predictive variance on `eval_rows`.

    A-optimality under the Laplace approximation. The fitted LR gives H = Z_L' S Z_L + I/C with
    S = diag(p(1-p)); buying candidate j adds s_j z_j z_j', and Sherman-Morrison turns the
    resulting drop in trace(Z_E H^-1 Z_E') into a closed form,

        gain(j) = s_j * ||Z_E H^-1 z_j||^2 / (1 + s_j z_j' H^-1 z_j)

    The inverse is updated after each pick rather than once per batch, so a batch cannot be
    `batch` copies of one neighbourhood -- the failure mode `laplace_var` has in main.py.
    """
    if clf is None or not len(cand) or not len(eval_rows):
        return None
    one = np.ones((len(Z), 1))
    A = np.hstack([Z, one])                                  # intercept as a column
    p = clf.predict_proba(Z[labelled])[:, 1]
    s = np.clip(p * (1 - p), 1e-6, None)
    H = (A[labelled] * s[:, None]).T @ A[labelled]
    H[:-1, :-1] += np.eye(A.shape[1] - 1) / max(args.C, 1e-12)   # no prior on the intercept
    M = np.linalg.pinv(H)

    pc = clf.predict_proba(Z[cand])[:, 1]
    sc = np.clip(pc * (1 - pc), 1e-6, None)
    E, C = A[eval_rows], A[cand]
    alive = np.ones(len(cand), bool)
    out = []
    for _ in range(min(batch, len(cand))):
        G = (E @ M) @ C.T                                    # (n_eval, n_cand)
        num = sc * np.einsum("ij,ij->j", G, G)
        den = 1.0 + sc * np.einsum("ij,ij->i", C @ M, C)
        gain = np.where(alive, num / den, -np.inf)
        j = int(np.argmax(gain))
        out.append(int(cand[j]))
        alive[j] = False
        v = M @ C[j]
        M = M - np.outer(v, v) * sc[j] / (1.0 + sc[j] * C[j] @ v)
    return np.array(out, dtype=int)


def pick(method, rng, sp, clf, y_al, labelled, cand, eval_rows, batch, qs, skclf, args):
    """-> the rows to buy this round. Candidates are the whole cohort's unlabelled blocks."""
    batch = min(batch, len(cand))
    if method == "random":
        return rng.choice(cand, size=batch, replace=False)
    if method == "laplace":
        got = laplace_pick(clf, sp.Z, labelled, cand, eval_rows, batch, args)
        return rng.choice(cand, size=batch, replace=False) if got is None else got
    if method in NEEDS_CLF:
        loc = qs.query(X=sp.Z, y=y_al, clf=skclf, fit_clf=True, batch_size=batch)
    elif method in SKACTIVEML:
        loc = qs.query(X=sp.Z, y=y_al, batch_size=batch)
    else:
        raise SystemExit(f"unknown method {method!r}; choose from {METHODS}")
    return np.asarray(loc, dtype=int)


def build_query(method, seed, args):
    """-> (query strategy, wrapped classifier or None). One object per (target, method)."""
    if method in ("random", "laplace"):
        return None, None
    if method in NEEDS_CLF:
        from skactiveml.classifier import SklearnClassifier         # noqa: PLC0415
        from skactiveml.pool import UncertaintySampling             # noqa: PLC0415
        skclf = SklearnClassifier(
            LogisticRegression(C=args.C, max_iter=args.max_iter,
                               class_weight=args.class_weight or None),
            classes=[0.0, 1.0], missing_label=np.nan, random_state=seed)
        return UncertaintySampling(method=args.uncertainty_method, random_state=seed), skclf
    return build_strategy(method, seed, args.typiclust_k), None


def run_target(sp, tgt_uuid, tr_day, args, split_id):
    """One target's curves, every method from THAT TARGET's own seed.

    -> (rows, queries, seed was short of positives). The seed is personal_al.py's row for row,
    which is what makes the pool the only difference between the two runs.
    """
    pid = sp.ids[tgt_uuid]
    own = sp.rows_of(tgt_uuid)
    rng = np.random.default_rng([args.seed, split_id, int(pid[1:]), 11])
    day = block_days(tr_day[tgt_uuid], args.block)
    assert len(day) == len(own), f"{len(day)} block days for {len(own)} blocks of {pid}"
    loc, flagged = seed_set(sp.y[own], day, rng, args.n_seed, args.n_seed_pos)
    seed_rows = own[loc]

    Zte, yte = sp.test_of(tgt_uuid)
    n_pos_te = int(np.sum(yte == 1))
    # Keyed on the target as well as the split, and NOT drawn from `rng` above, so a method's
    # position in --methods cannot change what it buys.
    strat_seed = int(np.random.default_rng(
        [args.seed, split_id, int(pid[1:]), 7]).integers(2 ** 31))

    rows, queries = [], []
    for method in args.methods:
        qrng = np.random.default_rng(strat_seed)
        qs, skclf = build_query(method, strat_seed, args)
        y_al = np.full(len(sp.y), np.nan)
        y_al[seed_rows] = sp.y[seed_rows]
        labelled, bought = list(seed_rows), []
        clf, failed = fit_lr(sp.Z[labelled], sp.y[labelled], args)
        for rnd in range(args.rounds + 1):
            if rnd > 0:
                cand = np.flatnonzero(np.isnan(y_al))
                if not len(cand):
                    break
                own_unl = own[np.isnan(y_al[own])]
                idx = pick(method, qrng, sp, clf, y_al, labelled, cand, own_unl,
                           args.batch, qs, skclf, args)
                y_al[idx] = sp.y[idx]
                labelled += [int(i) for i in idx]
                bought += [int(i) for i in idx]
                for i in idx:
                    queries.append(dict(split=split_id, target=pid, method=method, round=rnd,
                                        row=int(i),
                                        owner=sp.ids.get(sp.owner[i], sp.owner[i]),
                                        is_self=bool(sp.owner[i] == tgt_uuid)))
                clf, failed = fit_lr(sp.Z[labelled], sp.y[labelled], args)
            auroc, auprc = scores(yte, decide(clf, Zte))
            rows.append(dict(split=split_id, target=pid, method=method, round=rnd,
                             n_labeled=len(labelled), acquired=len(labelled) - len(seed_rows),
                             auroc=auroc, auprc=auprc, n_pos_test=n_pos_te,
                             self_share=(float(np.mean([sp.owner[i] == tgt_uuid
                                                        for i in bought]))
                                         if bought else np.nan),
                             fit_failed=bool(failed), seed_flagged=bool(flagged)))
    return rows, queries, flagged


def run_split(data, cols, split_id, args, ids=None):
    t0 = time.time()
    sp, ids, (tr_day, _) = prepare_one(data, cols, seed=split_id, W=args.block, ids=ids)
    targets = [u for u in sp.users if np.sum(sp.yte[u] == 1) >= args.min_test_pos]
    print(f"  split {split_id}: {sp}, {len(targets)} targets, {args.n_seed} own seed labels "
          f"each ({args.n_seed_pos} positive)", flush=True)

    F = pd.DataFrame()
    if not args.no_refs:
        F = full_reference(sp, targets, args, split_id)
        print(f"    reference (100% of {len(sp.y)} labels): AUROC {F.auroc.mean():.4f}, "
              f"AUPRC {F.auprc.mean():.4f} over {len(F)} targets", flush=True)

    rows, queries, short = [], [], []
    for n, u in enumerate(targets, 1):
        r, q, flagged = run_target(sp, u, tr_day, args, split_id)
        rows += r
        queries += q
        if flagged:
            short.append(sp.ids[u])
        if n % 10 == 0 or n == len(targets):
            print(f"    {n}/{len(targets)} targets, {time.time() - t0:.0f}s", flush=True)
    if short:
        print(f"    too few positives for a full seed: {', '.join(short)}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(queries), F, ids


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
    ap.add_argument("--n_seed", type=int, default=3,
                    help="the TARGET's own initial labels. personal_al.py's default; it has to "
                         "match there, or the pool contrast is not paired")
    ap.add_argument("--n_seed_pos", type=int, default=1, help="how many of them are positive")
    ap.add_argument("--typiclust_k", type=int, default=20)
    ap.add_argument("--uncertainty_method", default="least_confident",
                    choices=["least_confident", "margin_sampling", "entropy"])
    ap.add_argument("--C", type=float, default=1.0,
                    help="inverse L2 strength; 1.0 matches the mixed model's fe_p=1.0")
    ap.add_argument("--max_iter", type=int, default=1000)
    ap.add_argument("--class_weight", default=None, choices=[None, "balanced"])
    ap.add_argument("--min_test_pos", type=int, default=1,
                    help="skip a target with fewer positive test rows than this")
    ap.add_argument("--max_targets", type=int, default=0, help="debug: stop after N targets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="suffix for the folder name")
    ap.add_argument("--name", default=None, help="the folder name outright")
    ap.add_argument("--resume", action="store_true", help="skip splits already written")
    ap.add_argument("--no_index", action="store_true")
    ap.add_argument("--no_refs", action="store_true",
                    help="skip the 100%%-label reference. One extra LR per split, so it is on "
                         "by default and reference_models.py is not needed for this scope.")
    ap.add_argument("--refs_only", action="store_true",
                    help="write ONLY the reference for an existing run and exit, leaving its "
                         "curves alone. Needs --name/--out, and block and C must match it.")
    ap.add_argument("--out", default=None, help="override the derived folder")
    args = ap.parse_args()

    t = get_target(args.target)
    args.target = t.name
    out = args.out or run_dir(t.name, args.block, args.methods, args.rounds, args.batch,
                              0, tag=args.tag, scope="global_lr", name=args.name,
                              create=not args.refs_only)
    # --refs_only writes no config: the folder already has one, and rewriting it from these
    # args would change the keys per_person_grid.py gates sibling folders on.
    cfg = None
    if args.refs_only:
        if not os.path.exists(os.path.join(out, "config.json")):
            raise SystemExit(f"--refs_only needs a finished run; no config.json in {out}\n"
                             f"  name it: --name <folder under results/<target>/global_lr/>")
    else:
        os.makedirs(out, exist_ok=True)
        cfg = write_config(out, args, extra=dict(model="sklearn_logistic", pool="cohort",
                                                 seed_source="self",
                                                 seed_rule="own_rows_day_spread"))
    print(f"{t.name}: pooled LR, {' vs '.join(args.methods)}, W={args.block}, "
          f"{args.n_seed} own seed + {args.rounds} x {args.batch} "
          f"= {args.rounds * args.batch} acquired, {args.splits} split(s)")
    print(f"  -> {out}", flush=True)

    data, cols = load(t, verbose=True)
    have = done_splits(out) if args.resume else []
    if have:
        print(f"  resuming: splits {have} already written", flush=True)
    ids = canonical_ids(data)
    if args.refs_only:
        for k in range(args.first_split, args.first_split + args.splits):
            dst = analysis_path(out, f"reference_auroc_split{k}.csv")
            if args.resume and os.path.exists(dst):
                continue
            F, ids = refs_for_split(data, cols, k, args, ids)
            F.to_csv(dst, index=False)
            print(f"  split {k}: reference AUROC {F.auroc.mean():.4f}, "
                  f"AUPRC {F.auprc.mean():.4f} over {len(F)} targets", flush=True)
        return

    for k in range(args.first_split, args.first_split + args.splits):
        if k in have:
            continue
        R, Q, F, ids = run_split(data, cols, k, args, ids)
        if args.max_targets:
            keep = list(dict.fromkeys(R.target))[:args.max_targets]
            R, Q = R[R.target.isin(keep)], Q[Q.target.isin(keep)]
            if len(F):
                F = F[F.target.isin(keep)]
        f_round, f_query = split_files(out, k)
        R.to_csv(f_round, index=False)
        Q.to_csv(f_query, index=False)
        # One file per split, never a shared one: concurrent jobs would race on it.
        # per_person_grid.py merges reference_auroc_split*.csv itself.
        if len(F):
            F.to_csv(analysis_path(out, f"reference_auroc_split{k}.csv"), index=False)
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
