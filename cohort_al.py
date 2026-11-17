"""ONE mixed-effects model for the whole cohort, seeded with a pair of blocks per participant.

Unlike main.py, which fits one personalised model per target, this fits a single shared model:
the seed is --n_seed_per_person blocks from everyone, the pool is every unlabelled block, and
one acquisition stream serves the whole cohort. Each participant is still scored on their own
held-out rows under their own random effects, so the per-person plots read it unchanged.

Methods: laplace_pool (cohort A-optimality, eval_weighting="rows") plus the label-free
baselines. The target-specific `laplace` arm has no meaning here and is refused.

Writes main.py's schema. self_share is NaN -- with no target it is undefined.

    python cohort_al.py --methods laplace_pool random --splits 3
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from load_data import get_target, load
from main import build_strategy, initial_labels
from paths import done_splits, run_dir, split_files, update_index, write_config
from preprocess_data import canonical_ids, prepare_one
from utility import MixedEffectsClassifier, attach_person, scores, select_slopes

METHODS = ["laplace_pool", "coreset", "typiclust", "probcover", "random"]
LAPLACE = {"laplace_pool"}


def run_split(data, cols, split_id, args, ids=None):
    """One split: seed, then `rounds` x `batch` purchases per method. -> (rows, queries)."""
    t0 = time.time()
    sp, ids, (tr_day, _) = prepare_one(data, cols, seed=split_id, W=args.block, ids=ids)
    slope_idx = select_slopes(sp.Z, sp.y, sp.pid, args.n_slopes, cols, verbose=True)
    scored = [u for u in sp.users if np.sum(sp.yte[u] == 1) >= args.min_test_pos]
    print(f"  split {split_id}: {sp}, {len(scored)} participants scored", flush=True)

    Xclf = attach_person(sp.Z, sp.owner, sp.users)
    # each participant's rows under their OWN code: what personalises a shared model
    Xte, yte, n_pos = {}, {}, {}
    for u in scored:
        Z_u, y_u = sp.test_of(u)
        Xte[u] = attach_person(Z_u, [u] * len(y_u), sp.users)
        yte[u] = y_u
        n_pos[u] = int(np.sum(y_u == 1))

    # initial_labels' cohort branch ignores the participant argument
    rng = np.random.default_rng([args.seed, split_id, 11])
    seed_args = argparse.Namespace(seed_source="cohort", block=args.block,
                                   n_seed_per_person=args.n_seed_per_person,
                                   n_seed=args.n_seed_per_person, n_seed_pos=0)
    u0 = sp.users[0]
    seed_rows, flagged = initial_labels(sp, u0, sp.rows_of(u0), tr_day, rng, seed_args)
    print(f"    seed: {len(seed_rows)} blocks from {len(set(sp.owner[seed_rows]))} "
          f"participants ({sp.y[seed_rows].mean():.1%} positive)", flush=True)

    rows, queries = [], []
    for method in args.methods:
        if method not in METHODS:
            raise SystemExit(f"unknown method {method!r}; choose from {METHODS}")
        # per (split, method), so position in --methods cannot change what it buys
        strat_seed = int(np.random.default_rng(
            [args.seed, split_id, abs(hash(method)) % 2 ** 31, 7]).integers(2 ** 31))
        qs = build_strategy(method, strat_seed, args.typiclust_k)
        y_al = np.full(len(sp.y), np.nan)
        y_al[seed_rows] = sp.y[seed_rows]

        clf = MixedEffectsClassifier(persons=sp.users, slope_idx=slope_idx, fe_p=args.fe_p,
                                     vcp_p=args.vcp_p, classes=[0.0, 1.0],
                                     missing_label=np.nan)
        clf.fit(Xclf, y_al)
        for rnd in range(args.rounds + 1):
            if rnd > 0:
                unl = np.flatnonzero(np.isnan(y_al))
                if not len(unl):
                    break
                if method in LAPLACE:
                    # X_eval: whole unlabelled pool, each row under its own owner's code
                    idx = qs.query(X=Xclf, y=y_al, clf=clf, fit_clf=False,
                                   X_eval=Xclf[unl], batch_size=args.batch)
                else:
                    idx = qs.query(X=sp.Z, y=y_al, batch_size=args.batch)
                y_al[idx] = sp.y[idx]
                for i in idx:
                    queries.append(dict(split=split_id, method=method, round=rnd, row=int(i),
                                        owner=sp.ids.get(sp.owner[i], sp.owner[i]),
                                        target="COHORT", is_self=False))
                clf = MixedEffectsClassifier(persons=sp.users, slope_idx=slope_idx,
                                             fe_p=args.fe_p, vcp_p=args.vcp_p,
                                             classes=[0.0, 1.0], missing_label=np.nan)
                clf.fit(Xclf, y_al)

            lab = int(np.sum(~np.isnan(y_al)))
            for u in scored:
                auroc, auprc = scores(yte[u], clf.decision_function(Xte[u]))
                rows.append(dict(split=split_id, target=sp.ids[u], method=method, round=rnd,
                                 n_labeled=lab, acquired=lab - len(seed_rows),
                                 auroc=auroc, auprc=auprc, n_pos_test=n_pos[u],
                                 self_share=np.nan, fit_failed=bool(clf.fit_failed_),
                                 seed_flagged=bool(flagged)))
            if rnd % 5 == 0 or rnd == args.rounds:
                mu = np.nanmean([r["auroc"] for r in rows[-len(scored):]])
                print(f"    {method} round {rnd:3d}: {lab} labels, mean AUROC {mu:.4f}, "
                      f"{time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(queries), ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="computer_work")
    ap.add_argument("--block", type=int, default=15)
    ap.add_argument("--methods", nargs="+", default=["laplace_pool", "random"], choices=METHODS)
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--first_split", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--batch", type=int, default=3, help="labels bought per round, cohort-wide")
    ap.add_argument("--n_seed_per_person", type=int, default=2,
                    help="blocks taken from EACH participant for the initial labelled set")
    ap.add_argument("--n_slopes", type=int, default=5)
    ap.add_argument("--fe_p", type=float, default=1.0)
    ap.add_argument("--vcp_p", type=float, default=0.5)
    ap.add_argument("--typiclust_k", type=int, default=20)
    ap.add_argument("--min_test_pos", type=int, default=1,
                    help="skip scoring a participant with fewer positive test rows than this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--name", default=None, help="folder name, replacing the derived one")
    ap.add_argument("--resume", action="store_true", help="skip splits already written")
    ap.add_argument("--no_index", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t = get_target(args.target)
    args.target = t.name
    name = args.name or (f"{t.name}_cohort_model_{args.n_seed_per_person}per_"
                         f"r{args.rounds}x{args.batch}" + (f"_{args.tag}" if args.tag else ""))
    out = args.out or run_dir(t.name, args.block, args.methods, args.rounds, args.batch,
                              args.n_slopes, scope="full_global", name=name)
    os.makedirs(out, exist_ok=True)
    cfg = write_config(out, args, extra=dict(model="one_mixed_effects_cohort",
                                             pool="whole_cohort", seed_source="cohort"))
    print(f"{t.name}: ONE cohort model, {' vs '.join(args.methods)}, W={args.block}, "
          f"{args.n_seed_per_person}/participant seed + {args.rounds} x {args.batch} = "
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
        f_round, f_query = split_files(out, k)
        R.to_csv(f_round, index=False)
        Q.to_csv(f_query, index=False)
        if not args.no_index:
            update_index(out, cfg)
        last = R[R["round"] == R["round"].max()]
        w = last.pivot_table(index="target", columns="method", values="auroc")
        alt = next((m for m in args.methods if m != "random"), None)
        if alt and "random" in w and alt in w:
            g = (w[alt] - w["random"]).dropna()
            print(f"  split {k}: {alt}-random {g.mean():+.4f} over {len(g)} participants",
                  flush=True)
        else:
            print(f"  split {k}: written", flush=True)


if __name__ == "__main__":
    main()
