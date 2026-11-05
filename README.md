# Active_Learning

Personalised pool-based active learning on [ExtraSensory](http://extrasensory.ucsd.edu/), with
acquisitions from [scikit-activeml](https://scikit-activeml.github.io/) and a mixed-effects
logistic regression as the classifier.

Each target participant gets a personalised model, but labels may be bought from **anyone** in
the pool. The questions: does a smarter acquisition beat random, and whose labels does it buy?

## Setup

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
export ES_DATA_DIR=/path/to/ExtraSensory      # the 61 *.features_labels.csv files
```

Data is never copied into the repo. scikit-activeml needs scikit-learn >= 1.6, so use a
dedicated venv rather than an existing scientific environment.

## Run

```bash
./.venv/bin/python main.py                                   # computer_work, W=15, typiclust vs random
./.venv/bin/python main.py --methods coreset probcover random --splits 3
./.venv/bin/python reference_models.py --contains r10x3      # 100%-label references
./.venv/bin/python plots/per_person_grid.py --contains r10x3 # the per-participant grid
./.venv/bin/python paths.py                                  # list every run
```

## Files

| file | role |
|---|---|
| `load_data.py` | target -> `{uuid: (X, y, day)}`. A registry defines what "positive" means: `computer_work` / `in_a_meeting` are label columns over 124 phone+watch features; `walking` contrasts five named activities over 46 watch features. |
| `preprocess_data.py` | blocking (W minutes, majority label) -> day-held-out splits -> pooled, standardised matrices. Splits are drawn on minute rows and each side is blocked after; reversing that changes the cohort. |
| `utility.py` | the mixed-effects model (`BinomialBayesMixedGLM`, MAP) and its `SkactivemlClassifier` wrapper, plus seed sets, random-slope selection, scoring. |
| `main.py` | the AL loop. `TARGET` and `BLOCK_W` at the top; `--target` / `--block` override. |
| `reference_models.py` | `mixed_full` (personalised ceiling) and `global_prior` (pooled model, same prior) on 100% of the labels. |
| `slope_sweep.py` | the personalised ceiling as a function of random-slope count. |
| `paths.py` | the results layout. Every run folder is derived from its configuration. |
| `plots/` | figures, written into the run they came from. |

## Results layout

```
results/<target>/<target>_w15_random-typiclust_r10x3_s5/
    config.json              every argument + library versions
    per_round_split<k>.csv   split, target, method, round, n_labeled, auroc, ...
    queries_split<k>.csv     split, target, method, round, row, owner, is_self
    analysis/  figures/
results/index.csv            one row per run
```

The folder name encodes the configuration — `w15` block minutes, methods, `r10x3` rounds x
batch, `s5` random slopes — so identical settings reuse one folder (`--resume` continues it,
`--tag` keeps a variant apart). `results/` is gitignored.

## Method

1. Seed: 3 of the target's own blocks, 1 positive and 2 negative, spread over days. Shared by
   every method, so all curves start from the same model.
2. Each round: `strategy.query(X=Z, y=y_al, batch_size=3)` over the **whole pooled** training
   set, labels revealed, mixed model refitted, scored on the target's held-out days.
3. 10 rounds x 3 labels = 33 labels total.

`RandomSampling`, `CoreSet`, `TypiClust` and `ProbCover` need no classifier — they read the
pool's geometry — so the model is evaluated, not consulted. Uncertainty-style strategies take
`clf=`; the wrapper carries the participant index as the last column of `X`, which must be kept
out of any distance-based acquisition.
