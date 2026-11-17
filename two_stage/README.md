# Two-stage personalised active learning for GLMMs on multi-user sensor streams

A self-contained experiment inside the `Active_Learning` repo. It proposes and evaluates a
two-phase acquisition strategy for a mixed-effects (GLMM) classifier on
[ExtraSensory](http://extrasensory.ucsd.edu/), target label **COMPUTER_WORK**, and benchmarks it
against CoreSet, ProbCover, BALD and equal-budget random sampling on two axes: predictive
performance for the target participant, and how fairly the labelling burden falls on the cohort.

The headline benchmark is **`twostage` against equal-budget `random`** (`analyse.py --methods
twostage`). The other arms were run on the identical splits, seeds and cells and stay on disk:
they supply the diagnosis in §2 of the write-up and the ablations, and the full tables sit beside
the focused ones in each run's `analysis/`.

**[PROPOSAL.md](PROPOSAL.md) is the write-up** — the diagnosis, the method, the protocol and the
results. This file is just how to run it.

Nothing in the parent repo is modified. The data loading, blocking, day splits, standardisation,
random-slope selection, seed sets, the mixed-effects model and its Laplace covariance algebra are
all imported from it (`load_data.py`, `preprocess_data.py`, `utility.py`, `laplace_variance.py`,
`paths.py`), so an arm here is comparable to a run there by construction.

## Run

```bash
# the benchmark: 10 day splits x 43 targets x 9 arms, 3 seed + 10 rounds x 3 labels
SPLITS=10 PAR=6 bash two_stage/run_all.sh

# one split, two arms, three targets -- the smoke test
./.venv/bin/python two_stage/run_experiment.py --methods twostage random \
    --max_targets 3 --rounds 3 --tag smoke

# the diagnosis (section 2 of the proposal): what each method buys, and from whom
./.venv/bin/python two_stage/diagnose.py

# the benchmark read-out (sections 5-6): curves, paired tests, burden, the equity plane
./.venv/bin/python two_stage/analyse.py --methods twostage     # the headline: vs random
./.venv/bin/python two_stage/analyse.py                        # every arm on disk

# the sensitivity of the one free knob: how many labels Phase 1 should get
bash two_stage/run_ncalib.sh

# the algebra, checked against explicit re-inversion -- run this after touching acquisitions.py
./.venv/bin/python two_stage/selftest.py
./.venv/bin/python two_stage/bald.py
./.venv/bin/python two_stage/density_ratio.py
./.venv/bin/python two_stage/metrics.py
```

**Memory.** `PAR` is limited by RAM, not by cores. The `probcover` arm builds the full
(n x n) distance matrix of the pool for every target -- ~0.95 GB at n = 11 000 -- on top of the
~0.7 GB each process holds for the dataset, so `PAR=6` peaks near 10 GB and swaps on a 16 GB
machine (measured: throughput per target fell from 21 s to ~30 s and worse once swap filled).
Use `PAR=3` on 16 GB if the `probcover` arm is in the method list; runs without it are light.

`ES_DATA_DIR` points at the 61 `*.features_labels.csv` files (the parent repo falls back to a
sibling path). `TWO_STAGE_RESULTS_DIR` moves the results tree; `TWO_STAGE_CACHE` moves the
attribute cache.

## The arms

| arm | what it is |
|---|---|
| `twostage` | **the proposal**: Phase 1 D-optimality on the target's random effects `b_T` for `--n_calib` labels, then Target-Weighted Fixed-Effect Gain |
| `twostage_kmed` | same schedule, Phase 1 replaced by target-restricted k-medoids (is the posterior needed to calibrate?) |
| `twfeg` | Phase 2 from the first round (is Phase 1 needed?) |
| `twfeg_own` | Phase 2 with the evaluation measure uniform on the target's own blocks instead of the density ratio (is the ratio needed?) |
| `calib` | Phase 1 for the whole budget (is Phase 2 needed?) |
| `bald` | standard BALD over a committee sampled from the Laplace posterior |
| `coreset` | k-centre greedy (Sener & Savarese, 2018), from scikit-activeml |
| `probcover` | ProbCover (Yehuda et al., 2022), from scikit-activeml |
| `random` | equal-budget uniform sampling over the pool — **the benchmark's reference arm** |
| `self_random` | uniform sampling restricted to the target's own blocks: the control that separates "ask the target" from "ask the *right* blocks of the target" |
| `laplace` | the parent repo's posterior criterion: A-optimality on the target's *full* predictive variance, random effects included. The strongest baseline in the repo and Phase 2's closest relative |
| `twfeg_marg` | TW-FEG with a stranger's *first* label discounted by the exact factor for marginalising that stranger's own random effect out of the observation |
| `twostage_marg` | the schedule with that refinement in Phase 2 |

The last three are opt-in (`--methods laplace ...`): the run folder is named after the method
list, so adding an arm to the default would move every future default run to a new folder.

## Files

| file | role |
|---|---|
| `acquisitions.py` | the two phases and the schedule: `TargetCalibration`, `TWFEG`, `TwoStage`. The derivations are in the module docstring |
| `density_ratio.py` | `p_T(x)/p_pool(x)` by probabilistic classification, and the target-weighted second moment `A_T` Phase 2 integrates against |
| `bald.py` | `sample_proba` on the Laplace posterior, which is what lets scikit-activeml's `GreedyBALD` take this model |
| `metrics.py` | Macro PR-AUC / ROC-AUC, and the burden statistics (Gini, max-to-mean) |
| `vc_diagnostic.py` | what a 33-label mixed fit actually identifies: the fitted variance components, the target's random effects, and the size of the singleton discount |
| `probcover_delta.py` | the covering radius ProbCover selects here, against the pool's own distance scale |
| `manifold_check.py` | mechanism check: what percentile of the target's density ratio each method's purchases sit at |
| `who_benefits.py` | per-participant gaps: whether the variation is real (split-half reliability), what predicts it, and which mechanisms it is not |
| `run_experiment.py` | the AL loop over every arm; one process per split |
| `run_all.sh`, `run_ncalib.sh` | local parallel drivers |
| `pool_attributes.py` | method-independent row and participant attributes (isolation, phase-shift impurity, owner-extremity, outlier individuals), cached per split |
| `diagnose.py` | what each acquisition buys, from whom, and at what cost in equity |
| `analyse.py` | learning curves (pooled and per participant), paired tests, label efficiency, phase read-out, the equity-performance plane |
| `selftest.py` | both criteria against explicit re-inversion, and the schedule's routing |

## Output layout

```
two_stage/results/computer_work/<run>/   per_round_split<k>.csv, queries_split<k>.csv,
                                         config.json, analysis/ (tables + figures)
two_stage/diagnosis/                     row_profile.csv, person_profile.csv, burden_curve.csv,
                                         fig_row_profile.png, fig_burden.png, fig_lorenz.png,
                                         fig_isolation.png
two_stage/cache/                         per-split row attributes (recomputable, safe to delete)
two_stage/logs/                          one log per split
```

`per_round_split<k>.csv` keeps the parent repo's columns (`auroc`, `auprc`, `self_share`, ...) and
adds `macro_prauc`, `macro_rocauc`, the within-target burden and timing.
`queries_split<k>.csv` adds `phase` (which phase bought the label) and `y` (what it turned out to
be) to the parent's `row` / `owner` / `is_self` record.
