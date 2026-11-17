"""The benchmark: the two-phase framework against CoreSet, ProbCover, BALD and random.

    python two_stage/run_experiment.py --first_split 0 --splits 1              # everything
    python two_stage/run_experiment.py --methods twostage random --max_targets 3 --rounds 2

One process runs ONE split through EVERY method, which matters for two reasons. The expensive
part of a split is shared (61 CSVs parsed, day split drawn, pool standardised, random slopes
selected on the full training labels), and -- more importantly -- every arm then starts from
*byte-identical* seed sets and sees the identical pool, so a difference between two curves is the
acquisition and nothing else. Parallelism is across splits, as in the parent repo's cluster
fan-out.

What is measured, per (split, target, method, round)
---------------------------------------------------
Axis 1, on the target's held-out DAYS: ``macro_prauc`` (the headline), ``macro_rocauc``,
``auroc``, ``auprc``. The first is the proposal's primary endpoint because a 16%-positive target
makes positive-class PR-AUC alone jumpy at a 33-label budget; the parent repo's two columns are
kept so the curves can be laid beside existing runs.

Axis 2, on the queries: every purchased label's owner is written out, so cohort burden (Gini
coefficient, max-to-mean ratio) is computed afterwards from the record rather than estimated.
The per-round CSV also carries the within-target burden -- how concentrated ONE personalised
model's 30 purchases are across the cohort -- because the two are different claims and both get
made in the write-up.

The arms
--------
``twostage``      the framework: Phase 1 D-optimality on b_T for ``--n_calib`` labels, then TW-FEG
``twostage_kmed`` same schedule, Phase 1 replaced by target-restricted k-medoids (is the
                  posterior needed to calibrate, or would representative points do?)
``twfeg``         Phase 2 from the first round (is the calibration phase needed?)
``twfeg_own``     TW-FEG with the evaluation measure put uniformly on the target's OWN blocks
                  instead of the density ratio (is the ratio doing anything?)
``calib``         Phase 1 for the whole budget (is Phase 2 needed, or is self-labelling enough?)
``bald``          standard BALD over the posterior committee (``bald.py``)
``coreset``       k-centre greedy, from scikit-activeml, on the pool's geometry
``probcover``     ProbCover, from scikit-activeml, on the pool's geometry
``random``        equal-budget uniform sampling over the pool

Shared with the parent repo on purpose: the data loading, blocking, day splits, standardisation,
random-slope selection, seed sets, the model and its priors. The RNG recipes for the seed set and
for the strategy are copied verbatim from ``main.py`` -- same seed, same split, same participant
-> same seed set -- so a run here is directly comparable to one there.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from acquisitions import TWFEG, TargetCalibration, TwoStage, chol_psd      # noqa: E402
from bald import SampledMixedEffectsClassifier, build_bald                 # noqa: E402
from density_ratio import eval_weights, second_moment                      # noqa: E402
from load_data import get_target, load                                     # noqa: E402
from main import build_strategy as parent_strategy                         # noqa: E402
from metrics import burden, target_scores                                  # noqa: E402
from paths import done_splits, run_dir, split_files, write_config          # noqa: E402
from preprocess_data import canonical_ids, prepare_one                     # noqa: E402
from utility import attach_person, seed_set, select_slopes                 # noqa: E402

RESULTS = os.environ.get("TWO_STAGE_RESULTS_DIR") or os.path.join(HERE, "results")

DEFAULT_METHODS = ("twostage", "twostage_kmed", "twfeg", "twfeg_own", "calib", "bald",
                   "coreset", "probcover", "random")
# `laplace` is the parent repo's own posterior criterion (A-optimality on the target's FULL
# predictive variance, random effects included). It is the strongest baseline in the repo and the
# closest relative of Phase 2, so it belongs in the comparison -- but it is kept out of the
# default list on purpose: the run folder is named after the method list, and adding an arm to
# the default would move every future default run to a new folder and orphan the finished one.
# Ask for it explicitly: --methods laplace.
# `twfeg_marg` / `twostage_marg` are the refinement: Phase 2 with a stranger's FIRST label
# discounted by the exact factor for marginalising that stranger's own random effect out of the
# observation (acquisitions.singleton_attenuation). Also opt-in, for the same folder-naming
# reason, and so that the criterion as proposed is what the headline table reports.
# `self_random` is the control that separates the two claims Phase 1 makes at once: that the
# TARGET's blocks are the right ones to buy, and that D-optimality picks the right blocks among
# them. It is uniform sampling restricted to the target's own unlabelled rows, so `calib` minus
# `self_random` is the value of the design criterion and `self_random` minus `random` is the
# value of asking the target at all.
ALL_METHODS = DEFAULT_METHODS + ("laplace", "twfeg_marg", "twostage_marg", "self_random")
GEOMETRIC = {"coreset", "probcover", "random"}      # pool geometry only, no model, no person code
SCHEDULED = {"twostage", "twostage_kmed", "twostage_marg"}       # Phase 1 then Phase 2
PHASE2 = {"twfeg", "twfeg_own", "twfeg_marg"}                    # Phase 2 from round 1
POSTERIOR = SCHEDULED | PHASE2 | {"calib", "bald", "laplace"}
NEEDS_A = SCHEDULED | PHASE2
MEASURE = {"twfeg_own": "own"}                      # everything else uses the density ratio


def new_clf(sp, slope_idx, args, seed):
    """One estimator spec, used for every arm -- the thing being evaluated must not differ."""
    return SampledMixedEffectsClassifier(
        persons=sp.users, slope_idx=slope_idx, fe_p=args.fe_p, vcp_p=args.vcp_p,
        classes=[0.0, 1.0], missing_label=np.nan, random_state=seed, n_post=args.n_post)


def eval_factor(sp, uuid, measure, args, cache):
    """-> Cholesky factor of the target-weighted second moment A_T, cached per (target, measure).

    Round-independent (it is a property of the features and of whose density is being targeted),
    so it is built once per target and shared by every round and every arm that uses the same
    measure.
    """
    key = (uuid, measure)
    if key not in cache:
        pi = eval_weights(sp.Z, sp.owner, uuid, measure=measure, C=args.dr_C, cv=args.dr_cv,
                          seed=args.seed, clip_q=args.dr_clip)
        cache[key] = chol_psd(second_moment(sp.Z, pi))
    return cache[key]


def build(method, seed, args):
    """-> the query strategy for one arm."""
    if method in GEOMETRIC or method == "laplace":
        return parent_strategy(method, seed)                 # identical to the parent repo's
    if method == "self_random":
        return parent_strategy("random", seed)               # the same sampler, restricted below
    if method == "bald":
        return build_bald(seed, n_post=args.n_post, greedy=args.bald_greedy)
    if method == "twostage":
        return TwoStage(n_calib=args.n_calib, calib="dopt", random_state=seed)
    if method == "twostage_kmed":
        return TwoStage(n_calib=args.n_calib, calib="kmedoids", random_state=seed)
    if method == "twostage_marg":
        return TwoStage(n_calib=args.n_calib, calib="dopt", marginalise=True, random_state=seed)
    if method in ("twfeg", "twfeg_own"):
        return TWFEG(random_state=seed)
    if method == "twfeg_marg":
        return TWFEG(marginalise=True, random_state=seed)
    if method == "calib":
        return TargetCalibration(criterion="dopt", random_state=seed)
    raise ValueError(f"unknown method {method!r}")


def ask(method, qs, sp, uuid, own, y_al, clf, Xclf, A_chol, n_bought, args):
    """One round's query, routed to the interface each strategy actually has. -> row indices."""
    unl = np.where(np.isnan(y_al))[0]
    own_unl = np.intersect1d(own, unl)
    bs = args.batch
    if method in GEOMETRIC:
        # the person code is NOT a coordinate: geometric strategies see the features only
        return qs.query(X=sp.Z, y=y_al, batch_size=bs)
    if method == "self_random":
        if len(own_unl) == 0:                       # the target has no blocks left to label
            return qs.query(X=sp.Z, y=y_al, batch_size=bs)
        return qs.query(X=sp.Z, y=y_al, candidates=own_unl,
                        batch_size=min(bs, len(own_unl)))
    if method == "bald":
        return qs.query(X=Xclf, y=y_al, ensemble=clf, fit_ensemble=False, batch_size=bs)
    if method == "laplace":
        # the parent repo's call, verbatim: owner-coded pool, the target's own unlabelled rows as
        # the evaluation set, and no refit (clf was fitted at the end of the previous round)
        return qs.query(X=Xclf, y=y_al, clf=clf, fit_clf=False,
                        X_eval=Xclf[own_unl] if len(own_unl) else None, batch_size=bs)
    if method in SCHEDULED:
        return qs.query(X=Xclf, y=y_al, clf=clf, target=uuid, own_candidates=own_unl,
                        A_chol=A_chol, n_acquired=n_bought, candidates=unl, batch_size=bs)
    if method in PHASE2:
        return qs.query(X=Xclf, y=y_al, clf=clf, A_chol=A_chol, batch_size=bs)
    if method == "calib":
        if len(own_unl) < bs:            # the target has run out of own blocks to label
            return qs.query(X=Xclf, y=y_al, clf=clf, target=uuid,
                            candidates=own_unl if len(own_unl) else unl,
                            batch_size=min(bs, max(len(own_unl), 1)))
        return qs.query(X=Xclf, y=y_al, clf=clf, target=uuid, candidates=own_unl, batch_size=bs)
    raise ValueError(method)


def phase_of(method, n_before, j, args):
    """Which phase bought the j-th label of this round -- written into the queries CSV."""
    if method in SCHEDULED:
        return "calib" if (n_before + j) < args.n_calib else "twfeg"
    if method == "calib":
        return "calib"
    return method


def run_target(sp, uuid, tr_day, slope_idx, args, split_id, cache):
    """Every arm's learning curve for one target participant. -> (rows, queries)."""
    pid = sp.ids[uuid]
    own = sp.rows_of(uuid)
    # seed set and strategy seed: the recipes from main.py, verbatim, so the arms here start
    # from the same labels as the parent repo's runs on the same (seed, split, participant)
    rng = np.random.default_rng([args.seed, split_id, int(pid[1:]), 11])
    loc, flagged = seed_set(sp.y[own], tr_day[uuid], rng, args.n_seed, args.n_seed_pos)
    seed_rows = own[loc]
    strat_seed = int(np.random.default_rng(
        [args.seed, split_id, int(pid[1:]), 7]).integers(2 ** 31))

    Xclf = attach_person(sp.Z, sp.owner, sp.users)
    Zte, yte = sp.test_of(uuid)
    Xte = attach_person(Zte, [uuid] * len(yte), sp.users)
    n_pos_te = int(np.sum(yte == 1))

    rows, queries = [], []
    for method in args.methods:
        t0 = time.time()
        A_chol = (eval_factor(sp, uuid, MEASURE.get(method, "ratio"), args, cache)
                  if method in NEEDS_A else None)
        qs = build(method, strat_seed, args)
        y_al = np.full(len(sp.y), np.nan)
        y_al[seed_rows] = sp.y[seed_rows]
        bought, owners = [], []
        clf = new_clf(sp, slope_idx, args, strat_seed).fit(Xclf, y_al)
        for rnd in range(args.rounds + 1):
            if rnd > 0:
                n_before = len(bought)
                idx = np.asarray(ask(method, qs, sp, uuid, own, y_al, clf, Xclf, A_chol,
                                     n_before, args), dtype=int)
                idx = np.asarray([i for i in idx if np.isnan(y_al[i])], dtype=int)
                y_al[idx] = sp.y[idx]
                for j, i in enumerate(idx):
                    o = sp.ids.get(sp.owner[i], sp.owner[i])
                    bought.append(int(i))
                    owners.append(o)
                    queries.append(dict(split=split_id, target=pid, method=method, round=rnd,
                                        row=int(i), owner=o, is_self=bool(sp.owner[i] == uuid),
                                        phase=phase_of(method, n_before, j, args),
                                        y=float(sp.y[i])))
            clf = new_clf(sp, slope_idx, args, strat_seed).fit(Xclf, y_al)
            sc = target_scores(yte, clf.decision_function(Xte))
            b = burden([sp.ids[u] for u in sp.users], owners) if owners else {}
            lab = int(np.sum(~np.isnan(y_al)))
            rows.append(dict(split=split_id, target=pid, method=method, round=rnd,
                             n_labeled=lab, acquired=lab - len(seed_rows), **sc,
                             n_pos_test=n_pos_te,
                             self_share=float(np.mean(np.array(owners) == pid)) if owners
                             else np.nan,
                             within_gini=b.get("gini", np.nan),
                             within_max_mean=b.get("max_to_mean", np.nan),
                             n_owners=b.get("n_touched", 0),
                             pos_bought=float(np.mean([sp.y[i] for i in bought])) if bought
                             else np.nan,
                             fit_failed=bool(clf.fit_failed_), seed_flagged=bool(flagged),
                             secs=round(time.time() - t0, 1)))
    return rows, queries


def run_split(data, cols, split_id, args, ids):
    """Prepare one split and run every eligible target through every arm."""
    t0 = time.time()
    sp, ids, (tr_day, _) = prepare_one(data, cols, seed=split_id, W=args.block, ids=ids)
    slope_idx = select_slopes(sp.Z, sp.y, sp.pid, args.n_slopes, cols, verbose=True)
    targets = [u for u in sp.users if np.sum(sp.yte[u] == 1) >= args.min_test_pos]
    if args.max_targets:
        targets = targets[:args.max_targets]
    print(f"  split {split_id}: {sp}, {len(targets)} targets, "
          f"{len(args.methods)} methods", flush=True)
    rows, queries, cache = [], [], {}
    for n, u in enumerate(targets, 1):
        r, q = run_target(sp, u, tr_day, slope_idx, args, split_id, cache)
        rows += r
        queries += q
        if n % 5 == 0 or n == len(targets):
            print(f"    {n}/{len(targets)} targets, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(queries), ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="computer_work")
    ap.add_argument("--block", type=int, default=15)
    ap.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), choices=ALL_METHODS)
    ap.add_argument("--splits", type=int, default=1)
    ap.add_argument("--first_split", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--n_seed", type=int, default=3)
    ap.add_argument("--n_seed_pos", type=int, default=1)
    ap.add_argument("--n_slopes", type=int, default=5)
    ap.add_argument("--n_calib", type=int, default=6,
                    help="labels spent on Phase 1 before switching to TW-FEG (0 disables it)")
    ap.add_argument("--n_post", type=int, default=20, help="posterior draws for BALD")
    ap.add_argument("--bald_greedy", type=int, default=1,
                    help="1 = GreedyBALD (top-k), 0 = BatchBALD's joint selection")
    ap.add_argument("--dr_C", type=float, default=0.01, help="density-ratio discriminator L2")
    ap.add_argument("--dr_cv", type=int, default=3, help="folds for out-of-fold ratio estimates")
    ap.add_argument("--dr_clip", type=float, default=0.99, help="ratio clipping quantile")
    ap.add_argument("--fe_p", type=float, default=1.0)
    ap.add_argument("--vcp_p", type=float, default=0.5)
    ap.add_argument("--min_test_pos", type=int, default=1)
    ap.add_argument("--max_targets", type=int, default=0, help="debug: first N targets only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--resume", action="store_true", help="skip splits already written")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.bald_greedy = bool(args.bald_greedy)

    t = get_target(args.target)
    args.target = t.name
    out = args.out or run_dir(t.name, args.block, args.methods, args.rounds, args.batch,
                              args.n_slopes, args.tag, root=RESULTS)
    os.makedirs(out, exist_ok=True)
    write_config(out, args, extra=dict(_results_root=RESULTS, _n_calib=args.n_calib))
    print(f"{t.name}: {len(args.methods)} arms, W={args.block}, "
          f"{args.n_seed} seed + {args.rounds}x{args.batch} = "
          f"{args.n_seed + args.rounds * args.batch} labels, Phase 1 = {args.n_calib}")
    print(f"  -> {out}", flush=True)

    data, cols = load(t, verbose=True)
    ids = canonical_ids(data)
    have = done_splits(out) if args.resume else []
    for k in range(args.first_split, args.first_split + args.splits):
        if k in have:
            print(f"  split {k} already written, skipping", flush=True)
            continue
        R, Q, ids = run_split(data, cols, k, args, ids)
        f_round, f_query = split_files(out, k)
        R.to_csv(f_round, index=False)
        Q.to_csv(f_query, index=False)
        last = R[R["round"] == R["round"].max()]
        w = last.pivot_table(index="target", columns="method", values="macro_prauc")
        if "random" in w:
            for m in [c for c in w.columns if c != "random"]:
                gap = (w[m] - w["random"]).dropna()
                print(f"  split {k}: {m:14s} macro-PR-AUC vs random {gap.mean():+.4f} "
                      f"(SE {gap.std(ddof=1) / max(np.sqrt(len(gap)), 1):.4f}), "
                      f"better on {int((gap > 0).sum())}/{len(gap)}", flush=True)
        print(f"  split {k} -> {f_round}", flush=True)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
