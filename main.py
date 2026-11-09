"""Personalised pool-based active learning on ExtraSensory, with scikit-activeml acquisitions.

    python main.py                                   # COMPUTER_WORK, W=15, typiclust vs random
    python main.py --target walking --block 1        # a different target, no blocking
    python main.py --methods coreset random --splits 20

The question the run is built to answer: when a model is personalised to participant *p* but may
buy labels from ANYONE in the pool, does a smarter acquisition beat random -- and whose labels
does it buy? Each round queries from the pooled training blocks of every participant, the
mixed-effects model is refitted on everything labelled so far, and the curve is scored on *p*'s
held-out days only. Every query's owner is written out, so "which participant supplied the
labels" is answerable afterwards.

Pieces, each in its own file:

    load_data.py        target -> {uuid: (X, y, day)}          (what "positive" means per target)
    preprocess_data.py  that -> blocked, day-split, standardised matrices
    utility.py        the mixed-effects model + its scikit-activeml wrapper
    main.py           this file: the loop, the logging

Acquisitions come from scikit-activeml (``skactiveml.pool``). RandomSampling, CoreSet and
TypiClust need no classifier -- they work on the geometry of the pool -- so the mixed-effects
model is the thing being *evaluated*, not the thing driving the queries. Uncertainty-style
strategies would use the same wrapper through ``clf=``; see ``build_strategy``.

Outputs go to one deterministic folder per configuration (see paths.py), so the same settings
always land in the same place and ``results/index.csv`` lists every run that exists:

    results/<target>/<target>_w15_random-typiclust_r10x3_s5/
        config.json, per_round_split<k>.csv, queries_split<k>.csv, analysis/, figures/
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from load_data import get_target, load
from paths import done_splits, run_dir, split_files, update_index, write_config
from preprocess_data import canonical_ids, prepare_one
from utility import (MixedEffectsClassifier, attach_person, scores, seed_set, select_slopes)

# ----------------------------------------------------------------------------- defaults
TARGET = "computer_work"      # change here, or pass --target
BLOCK_W = 15                  # minutes per block; 1 = no blocking
METHODS = ("typiclust", "random")
N_SEED, N_SEED_POS = 3, 1     # initial labels per target, and how many must be positive
BATCH, ROUNDS = 3, 10         # labels bought per round, and how many rounds
N_SLOPES = 5                  # features given a per-participant random slope
TYPICLUST_K = 20              # kNN used for typicality (scikit-activeml's default is 5)


def build_strategy(name, seed, typiclust_k=TYPICLUST_K, eer_candidates=50):
    """-> a scikit-activeml pool strategy. All three ignore the classifier and use the pool only.

    Note the defaults differ from a hand-rolled implementation: TypiClust's `k` is 5 upstream,
    while the original experiments used 20 nearest neighbours for typicality, so it is set
    explicitly here. CoreSet is the k-centre greedy of Sener & Savarese (2018); TypiClust is
    Hacohen, Dekel & Weinshall (2022).
    """
    from skactiveml.pool import (CoreSet, MonteCarloEER, ProbCover, RandomSampling,
                                 SubSamplingWrapper, TypiClust)
    if name == "random":
        return RandomSampling(random_state=seed)
    if name == "coreset":
        return CoreSet(random_state=seed)
    if name == "typiclust":
        return TypiClust(random_state=seed, k=typiclust_k)
    if name == "probcover":
        # delta (the covering radius) is estimated from clustering purity, which needs the
        # class count; with one delta for a 16%-positive target it can come out large enough
        # to cover nearly everything, at which point the picks are close to random -- worth
        # checking the chosen delta before reading much into a null result
        return ProbCover(n_classes=2, random_state=seed)
    if name == "eer":
        # Expected error reduction refits the model twice per candidate (once per possible
        # label), so the full pool would be ~22k refits per round. SubSamplingWrapper draws
        # max_candidates of them at random first, which is the only thing that makes it
        # runnable; the result approximates the full criterion on that subsample.
        return SubSamplingWrapper(
            query_strategy=MonteCarloEER(method="misclassification_loss", random_state=seed),
            max_candidates=eer_candidates, random_state=seed)
    if name in ("laplace", "laplace_var"):
        # Our own strategy: variance reduction under the mixed model's Laplace posterior. The
        # library has nothing posterior-aware for this estimator (EpistemicUncertaintySampling
        # is isinstance-locked to ParzenWindowClassifier / LogisticRegression). Label-free and
        # closed-form, so unlike eer it scores the WHOLE pool -- no SubSamplingWrapper.
        from laplace_variance import LaplaceVarianceReduction
        return LaplaceVarianceReduction(
            criterion="variance" if name == "laplace_var" else "reduction", random_state=seed)
    raise ValueError(f"unknown acquisition {name!r}")


def run_target(sp, tgt_uuid, tr_day, slope_idx, args, split_id):
    """One participant's learning curves: every method, from the same seed set. -> (rows, queries).

    The seed set is the target's OWN labels, identical across methods so the curves start from
    the same model; everything after round 0 is the acquisition's doing.
    """
    pid = sp.ids[tgt_uuid]
    own = sp.rows_of(tgt_uuid)
    rng = np.random.default_rng([args.seed, split_id, int(pid[1:]), 11])
    loc, flagged = seed_set(sp.y[own], tr_day[tgt_uuid], rng, args.n_seed, args.n_seed_pos)
    seed_rows = own[loc]

    # The acquisition's RNG, keyed on the TARGET as well as the split. It used to be
    # `args.seed + split_id`, one integer shared by all 43 targets, which made RandomSampling
    # replay the identical stream for every one of them: M1 and M2 bought the same 30 rows, so
    # 43 "replicates" of the random arm were really one draw and its per-target error bars were
    # meaningless. Its own generator rather than a draw from `rng` above, because `rng` has
    # already been consumed by seed_set and would make the value depend on the method's
    # position in args.methods -- then `--methods random` and `--methods laplace random` would
    # disagree, breaking the promise cluster/run.py relies on when it submits them separately.
    # 7 just separates this stream from the 11 used for the seed set.
    strat_seed = int(np.random.default_rng(
        [args.seed, split_id, int(pid[1:]), 7]).integers(2 ** 31))

    Xclf = attach_person(sp.Z, sp.owner, sp.users)                 # person code as last column
    Zte, yte = sp.test_of(tgt_uuid)
    Xte = attach_person(Zte, [tgt_uuid] * len(yte), sp.users)
    n_pos_te = int(np.sum(yte == 1))

    CLF_STRATEGIES = {"eer"}
    # Laplace variance reduction is model-based too, but takes the OWNER-coded matrix, not the
    # target-coded one: its criterion reads the candidate's own random-effect columns, so
    # recoding every candidate as the target (what eer needs) would erase exactly the
    # provenance term that makes it interesting. See laplace_variance.py, `Candidate coding`.
    LAPLACE_STRATEGIES = {"laplace", "laplace_var"}
    rows, queries = [], []
    for method in args.methods:
        qs = build_strategy(method, strat_seed, args.typiclust_k, args.eer_candidates)
        y_al = np.full(len(sp.y), np.nan)
        y_al[seed_rows] = sp.y[seed_rows]
        bought = []
        clf = MixedEffectsClassifier(persons=sp.users, slope_idx=slope_idx, fe_p=args.fe_p,
                                     vcp_p=args.vcp_p, classes=[0.0, 1.0], missing_label=np.nan)
        for rnd in range(args.rounds + 1):
            if rnd > 0:
                # scikit-activeml queries the pool by index; the FEATURES go in, never the
                # person column -- a participant code is not a coordinate to measure distance in
                if method in LAPLACE_STRATEGIES:
                    own_unl = own[np.isnan(y_al[own])]
                    # fit_clf=False: `clf` below was already fitted on (Xclf, y_al) at the end
                    # of the previous round, so refitting here would just repeat it.
                    idx = qs.query(X=Xclf, y=y_al, clf=clf, fit_clf=False,
                                   X_eval=Xclf[own_unl] if len(own_unl) else None,
                                   batch_size=args.batch)
                elif method in CLF_STRATEGIES:
                    # the model reads the participant code from the last column, so a
                    # model-based strategy must be given that matrix -- and every candidate is
                    # coded as the TARGET, i.e. "how much would this label reduce error for
                    # THIS participant", not for whoever happens to own the block.
                    Xq = attach_person(sp.Z, [tgt_uuid] * len(sp.Z), sp.users)
                    own_unl = own[np.isnan(y_al[own])]          # the target's own unlabelled rows
                    # fit_clf=True is required: skactiveml refuses a pre-fitted model it
                    # cannot attribute to known rows ("unknown where it has been fitted on"),
                    # so it refits on (Xq, y_al) itself before simulating each candidate label
                    idx = qs.query(X=Xq, y=y_al, clf=clf, fit_clf=True,
                                   ignore_partial_fit=True,
                                   X_eval=Xq[own_unl] if len(own_unl) else None,
                                   batch_size=args.batch)
                else:
                    idx = qs.query(X=sp.Z, y=y_al, batch_size=args.batch)
                y_al[idx] = sp.y[idx]
                bought.extend(int(i) for i in idx)
                for i in idx:
                    queries.append(dict(split=split_id, target=pid, method=method, round=rnd,
                                        row=int(i), owner=sp.ids.get(sp.owner[i], sp.owner[i]),
                                        is_self=bool(sp.owner[i] == tgt_uuid)))
            clf = MixedEffectsClassifier(persons=sp.users, slope_idx=slope_idx,
                                         fe_p=args.fe_p, vcp_p=args.vcp_p,
                                         classes=[0.0, 1.0], missing_label=np.nan)
            clf.fit(Xclf, y_al)
            auroc, auprc = scores(yte, clf.decision_function(Xte))
            lab = int(np.sum(~np.isnan(y_al)))
            rows.append(dict(split=split_id, target=pid, method=method, round=rnd,
                             n_labeled=lab, acquired=lab - len(seed_rows),
                             auroc=auroc, auprc=auprc, n_pos_test=n_pos_te,
                             self_share=float(np.mean([sp.owner[i] == tgt_uuid for i in bought]))
                             if bought else np.nan,
                             fit_failed=bool(clf.fit_failed_), seed_flagged=bool(flagged)))
    return rows, queries


def run_split(data, cols, split_id, args, ids=None):
    """Prepare one split and run every eligible target through it."""
    t0 = time.time()
    sp, ids, (tr_day, _) = prepare_one(data, cols, seed=split_id, W=args.block, ids=ids)
    slope_idx = select_slopes(sp.Z, sp.y, sp.pid, args.n_slopes, cols, verbose=True)
    targets = [u for u in sp.users if np.sum(sp.yte[u] == 1) >= args.min_test_pos]
    print(f"  split {split_id}: {sp}, {len(targets)} targets", flush=True)

    rows, queries = [], []
    for n, u in enumerate(targets, 1):
        r, q = run_target(sp, u, tr_day, slope_idx, args, split_id)
        rows += r
        queries += q
        if n % 5 == 0 or n == len(targets):
            print(f"    {n}/{len(targets)} targets, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(queries), ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default=TARGET, help="load_data.TARGETS key, or label:NAME")
    ap.add_argument("--block", type=int, default=BLOCK_W, help="W: minutes per block (1 = none)")
    ap.add_argument("--methods", nargs="+", default=list(METHODS),
                    choices=["typiclust", "coreset", "probcover", "eer", "laplace", "laplace_var",
                             "random"])
    ap.add_argument("--splits", type=int, default=1, help="independent day splits to average over")
    ap.add_argument("--first_split", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--batch", type=int, default=BATCH, help="labels bought per round")
    ap.add_argument("--n_seed", type=int, default=N_SEED)
    ap.add_argument("--n_seed_pos", type=int, default=N_SEED_POS)
    ap.add_argument("--n_slopes", type=int, default=N_SLOPES)
    ap.add_argument("--typiclust_k", type=int, default=TYPICLUST_K)
    ap.add_argument("--eer_candidates", type=int, default=50,
                    help="candidates subsampled per round for --methods eer (2 refits each)")
    ap.add_argument("--fe_p", type=float, default=1.0, help="fixed-effect prior SD")
    ap.add_argument("--vcp_p", type=float, default=0.5, help="prior SD on log random-effect SD")
    ap.add_argument("--min_test_pos", type=int, default=1,
                    help="a target needs this many positive test blocks for AUROC to exist")
    ap.add_argument("--max_targets", type=int, default=0, help="debug: stop after N targets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="keep a variant of the same settings separate")
    ap.add_argument("--reserve", action="store_true",
                    help="never write into a folder that already holds results")
    ap.add_argument("--resume", action="store_true", help="skip splits already written")
    ap.add_argument("--no_index", action="store_true",
                    help="do not touch results/index.csv -- for parallel jobs, which would "
                         "otherwise race on one file; run `python paths.py --reindex` after")
    ap.add_argument("--out", default=None, help="override the derived folder (rarely wanted)")
    args = ap.parse_args()

    t = get_target(args.target)
    args.target = t.name                                   # the registry name, not the alias
    out = args.out or run_dir(t.name, args.block, args.methods, args.rounds, args.batch,
                              args.n_slopes, args.tag, reserve=args.reserve)
    os.makedirs(out, exist_ok=True)
    cfg = write_config(out, args)
    print(f"{t.name}: {' vs '.join(args.methods)}, W={args.block}, "
          f"{args.n_seed} seed + {args.rounds} x {args.batch} = "
          f"{args.n_seed + args.rounds * args.batch} labels, {args.splits} split(s)")
    print(f"  -> {out}", flush=True)

    data, cols = load(t, verbose=True)
    have = done_splits(out) if args.resume else []
    if have:
        print(f"  resuming: splits {have} already written", flush=True)
    # Fixed uuid -> M1..Mn map, built from the dataset rather than from whichever split runs
    # first. Threading `ids` through the loop (the old behaviour) gave stable numbering only
    # because split `first_split` happened to run first and define it; a job that runs ONE split
    # in its own process -- which is what the cluster fan-out does -- would renumber from its own
    # keep list, and M21 would name a different person in different jobs. Verified to reproduce
    # the numbering every existing run already uses, so no finished result changes.
    ids = canonical_ids(data)
    for k in range(args.first_split, args.first_split + args.splits):
        if k in have:
            continue
        R, Q, ids = run_split(data, cols, k, args, ids)
        if args.max_targets:                       # debug runs keep only the first N targets
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
        if alt and "random" in w:
            gap = (w[alt] - w["random"]).dropna()
            print(f"  split {k}: {alt}-random {gap.mean():+.4f} "
                  f"(SE {gap.std(ddof=1) / np.sqrt(len(gap)):.4f}), "
                  f"{alt} better on {int((gap > 0).sum())}/{len(gap)}", flush=True)
    if not args.no_index:          # the per-split call above honours this; so must the final one
        update_index(out, cfg)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
