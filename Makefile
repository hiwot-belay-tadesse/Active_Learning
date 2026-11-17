# Active_Learning -- common commands.
#
#   make run METHOD=coreset SPLITS=20     one acquisition locally, no editing of cluster/run.py
#   make submit                           the whole QUEUE to SLURM
#   make submit ONLY=eu_split0            one job from the QUEUE, to test the setup
#   make submit_cohort                    cohort_al.py to SLURM, one array task per split
#   make dry_cohort                       print that sbatch line, submit nothing
#   make local                            the whole QUEUE in this shell (DRYRUN path)
#   make dry                              print what would be submitted, submit nothing
#   make refs RUN=<folder>                100%-label reference models for a finished run
#   make grid                             the per-participant grid
#   make reindex                          rebuild results/index.csv from the folders on disk
#   make selftest                         verify the Laplace covariance algebra
#   make help                             this list
#
# Every variable below is overridable on the command line, e.g.
#   make run METHOD="laplace random" SPLITS=5 ROUNDS=4 TAG=quick
#
# `run` calls main.py directly, which is the quick path for one-off work: it takes flags, so
# nothing in cluster/run.py has to be commented out. Use `submit`/`local` when you want the
# whole queue, which is what the cluster array runs.

# The local .venv when there is one, else whatever python is active -- so an activated mamba
# env is used as-is on a cluster checkout, which has no .venv. Nothing here activates anything.
# `make python` prints the choice; override with PY=... .
PY      ?= $(shell [ -x ./.venv/bin/python ] && echo ./.venv/bin/python || command -v python)
TARGET  ?= computer_work
BLOCK   ?= 15
METHOD  ?= laplace random
SPLITS  ?= 3
FIRST   ?= 0
ROUNDS  ?= 10
BATCH   ?= 3
NSEED   ?= 3
NSEEDPOS?= 1
NSLOPES ?= 5
TAG     ?=
RUN     ?=
EXTRA   ?=
# One job instead of the whole queue: ONLY is matched against the job name `dry` prints.
#   make submit ONLY=eu_split0      one job
#   make submit ONLY=_split0        one split of every queue entry
ONLY    ?=

# cohort_al.py -- ONE mixed-effects model for the whole cohort. It is a separate entry point
# with no target, so cluster/run.py's QUEUE (which expands a grid and calls main.py) cannot
# carry it; `submit_cohort` sbatches it directly instead. One array task per SPLIT, because a
# 20-split run does not fit in one allocation: at batch 10 the labelled set reaches ~1086 rows
# by round 100 and the Laplace refit cost grows with it, so a split is ~4 h and 20 are ~80 h.
CSPLITS ?= 20
CROUNDS ?= 100
CBATCH  ?= 10
CSEEDPP ?= 2
CMETHOD ?= laplace_pool coreset typiclust probcover random
CTIME   ?= 0-12:00
CPART   ?= shared
CMEM    ?= 16000
# Task ids. Override to re-run specific splits (ARRAY=3,7,11) or to cap concurrency (0-19%5).
ARRAY   ?= 0-$(shell echo $$(( $(CSPLITS) - 1 )))
# Activate your env and export ES_DATA_DIR before submitting: sbatch passes the submitting
# environment to the job, so nothing here activates anything.

TAGFLAG := $(if $(TAG),--tag $(TAG),)
ONLYFLAG := $(if $(ONLY),--only $(ONLY),)
RUNFLAG := $(if $(RUN),--run $(RUN),--contains r$(ROUNDS)x$(BATCH))

.PHONY: help run submit submit_cohort dry_cohort local dry refs python sweep grid bar reindex selftest clean-logs

python:
	@echo $(PY)

help:
	@sed -n '2,21p' Makefile | sed 's/^# \{0,1\}//'

# -------------------------------------------------------------------- running
run:
	$(PY) main.py --target $(TARGET) --block $(BLOCK) --methods $(METHOD) \
	  --first_split $(FIRST) --splits $(SPLITS) --rounds $(ROUNDS) --batch $(BATCH) \
	  --n_seed $(NSEED) --n_seed_pos $(NSEEDPOS) --n_slopes $(NSLOPES) \
	  --resume $(TAGFLAG) $(EXTRA)

# The queue in cluster/run.py. `submit` needs DRYRUN=False there, `local` needs DRYRUN=True --
# the Makefile does not flip it for you, because which one you want is not something a target
# name should decide silently.
submit:
	$(PY) cluster/submit_batch.py $(ONLYFLAG)

local:
	$(PY) cluster/submit_batch.py $(ONLYFLAG)

dry:
	$(PY) cluster/submit_batch.py --dry $(ONLYFLAG)

# Split in two so both targets share one definition and `dry_cohort` can echo the body
# verbatim: the body deliberately contains no single quotes, so '$(COHORT_WRAP)' is safe to
# quote in the recipe and to print. BLAS is pinned to one thread (the job holds one core and
# would otherwise spawn a thread per visible CPU) and --no_index keeps concurrent tasks off
# results/index.csv -- run `make reindex` once the array has drained.
COHORT_SB = -J cohort_al --array=$(ARRAY) -n 1 -N 1 -t $(CTIME) -p $(CPART) --mem=$(CMEM) \
	    -o cluster/logs/cohort/out_%A_%a.txt -e cluster/logs/cohort/err_%A_%a.txt
COHORT_WRAP = set -e; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; \
	$(PY) -u cohort_al.py --target $(TARGET) --methods $(CMETHOD) --rounds $(CROUNDS) \
	--batch $(CBATCH) --n_seed_per_person $(CSEEDPP) --splits 1 \
	--first_split $$SLURM_ARRAY_TASK_ID --resume --no_index

submit_cohort:
	@mkdir -p cluster/logs/cohort
	@test -n "$$ES_DATA_DIR" || echo 'WARNING: ES_DATA_DIR is unset -- the job will not find the CSVs'
	@echo "submitting with $(PY)"
	sbatch $(COHORT_SB) --wrap '$(COHORT_WRAP)'

dry_cohort:
	@echo 'sbatch $(COHORT_SB) --wrap:'
	@echo '    $(COHORT_WRAP)'
	@echo
	@echo '  -> results/$(TARGET)/full_global/$(TARGET)_cohort_model_$(CSEEDPP)per_r$(CROUNDS)x$(CBATCH)/'
	@echo '  -> after the array drains: make reindex'

# -------------------------------------------------------------- after a run
# Jobs run with --no_index to keep concurrent writers off results/index.csv; this rebuilds it
# and drops rows for runs that no longer exist on disk.
reindex:
	$(PY) paths.py --reindex

refs:
	$(PY) reference_models.py --target $(TARGET) $(RUNFLAG) $(EXTRA)

sweep:
	$(PY) slope_sweep.py --target $(TARGET) $(RUNFLAG) $(EXTRA)

grid:
	$(PY) plots/per_person_grid.py --target $(TARGET) $(RUNFLAG) $(EXTRA)

bar:
	$(PY) plots/personalisation_gain_bar.py --target $(TARGET) $(RUNFLAG) $(EXTRA)

# -------------------------------------------------------------------- checks
selftest:
	$(PY) laplace_variance.py --self-test

clean-logs:
	rm -rf cluster/logs cluster/tmp
