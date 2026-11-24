# Active_Learning -- common commands.
#
#   make run METHOD=coreset SPLITS=20     one acquisition locally, no editing of cluster/run.py
#   make submit                           the whole QUEUE to SLURM
#   make submit ONLY=eu_split0            one job from the QUEUE, to test the setup
#   make submit_cohort                    cohort_al.py to SLURM, one array task per split
#   make dry_cohort                       print that sbatch line, submit nothing
#   make submit_frozen                    frozen_mixed_al.py to SLURM, split x method array
#   make dry_frozen                       print those sbatch lines, submit nothing
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

# AL_PYTHON wins (export it once on the cluster: `export AL_PYTHON=$$(which python)`), else the
# local .venv, else whatever python is active. cluster/run.py reads AL_PYTHON the same way.
# Nothing here activates anything. `make python` prints the choice; override with PY=... .
PY      ?= $(shell if [ -n "$$AL_PYTHON" ]; then echo "$$AL_PYTHON"; \
	           elif [ -x ./.venv/bin/python ]; then echo ./.venv/bin/python; \
	           else command -v python; fi)
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
# One sbatch per method, each an array over the splits: `squeue` then shows the methods
# separately and a slow one can be cancelled or resubmitted on its own.
NMETH   := $(words $(CMETHOD))
ARRAY   ?= 0-$(shell echo $$(( $(CSPLITS) - 1 )))
CNAME   ?= $(TARGET)_cohort_model_$(CSEEDPP)per_r$(CROUNDS)x$(CBATCH)
# Activate your env and export ES_DATA_DIR before submitting: sbatch passes the submitting
# environment to the job, so nothing here activates anything.

TAGFLAG := $(if $(TAG),--tag $(TAG),)
ONLYFLAG := $(if $(ONLY),--only $(ONLY),)
RUNFLAG := $(if $(RUN),--run $(RUN),--contains r$(ROUNDS)x$(BATCH))

.PHONY: help run submit submit_cohort dry_cohort submit_frozen dry_frozen local dry refs python sweep grid bar reindex selftest clean-logs

python:
	@echo $(PY)

help:
	@sed -n '2,23p' Makefile | sed 's/^# \{0,1\}//'

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

# Defined once so `submit_cohort` and `dry_cohort` cannot drift and the dry run echoes the
# body verbatim. The recipes DOUBLE-quote it -- `$$M` has to expand to the loop's method -- so
# the body contains no double quotes, and SLURM's own variable is escaped (`\$$`) to survive
# into --wrap and be expanded by the job instead. BLAS is pinned to one thread (the job holds
# one core and would otherwise spawn a thread per visible CPU) and --no_index keeps concurrent
# tasks off results/index.csv -- run `make reindex` once the arrays have drained.
COHORT_WRAP = set -e; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; \
	$(PY) -u cohort_al.py --target $(TARGET) --methods $$M --rounds $(CROUNDS) \
	--batch $(CBATCH) --n_seed_per_person $(CSEEDPP) --splits 1 \
	--first_split \$$SLURM_ARRAY_TASK_ID --name $(CNAME)_$$M --resume --no_index

submit_cohort:
	@test -n "$$ES_DATA_DIR" || echo 'WARNING: ES_DATA_DIR is unset -- the job will not find the CSVs'
	@echo "submitting with $(PY)"
	@for M in $(CMETHOD); do \
	  mkdir -p cluster/logs/cohort/$$M; \
	  sbatch -J cohort_$$M --array=$(ARRAY) -n 1 -N 1 -t $(CTIME) -p $(CPART) \
	    --mem=$(CMEM) -o cluster/logs/cohort/$$M/out_%A_%a.txt \
	    -e cluster/logs/cohort/$$M/err_%A_%a.txt \
	    --wrap "$(COHORT_WRAP)"; \
	done

dry_cohort:
	@for M in $(CMETHOD); do \
	  echo "sbatch -J cohort_$$M --array=$(ARRAY) -t $(CTIME) -p $(CPART) --mem=$(CMEM) --wrap:"; \
	  echo "    $(COHORT_WRAP)"; \
	done
	@echo
	@echo '  -> $(NMETH) submissions x $(CSPLITS) splits'
	@echo '  -> results/$(TARGET)/full_global/$(CNAME)_<method>/   (one folder per method)'
	@echo '  -> after the arrays drain: make reindex'

# frozen_mixed_al.py -- mixed effects with the POPULATION FROZEN after the seed fit. Like
# cohort_al.py it is a separate entry point, so cluster/run.py's QUEUE -- which expands a grid
# and calls main.py -- cannot carry it, and `submit_frozen` sbatches it directly instead.
#
# Two dimensions, split differently on purpose: one array task per SPLIT (the unit of
# parallelism everywhere in this repo, so no two jobs touch one file), and one SUBMISSION per
# method, because run_id names the folder after the sorted method list -- so a per-method
# submission puts each arm in its own folder, which is what pooled_lr_al.py's runs already do
# and what the plotting scripts glob. It also lets one slow arm be resubmitted alone.
#
# Cost: one task is 43 targets x 11 rounds of (43 ridge GLMs x <=50 EM iterations), measured at
# ~3 min locally plus the data load, so 0-01:00 is already generous. --n_slopes 0 (every feature
# gets a random slope, the literal specification) is minutes PER TARGET in step 1 alone -- raise
# FTIME to 0-12:00 before trying that.
FSPLITS ?= 20
FROUNDS ?= 10
FBATCH  ?= 3
FSEEDPC ?= 1
FSLOPES ?= 5
FMETHOD ?= coreset typiclust probcover uncertainty random
FTIME   ?= 0-01:00
FPART   ?= shared
FMEM    ?= 8000
FARRAY  ?= 0-$(shell echo $$(( $(FSPLITS) - 1 )))
FEXTRA  ?=
NFMETH  := $(words $(FMETHOD))

# Defined once so `submit_frozen` and `dry_frozen` cannot drift, and quoted on the same terms as
# COHORT_WRAP: the recipes double-quote it, so the body holds no double quotes, and SLURM's own
# variable is escaped (\$$) to survive into --wrap and be expanded by the job.
FROZEN_WRAP = set -e; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; \
	$(PY) -u frozen_mixed_al.py --target $(TARGET) --block $(BLOCK) --methods $$M \
	--rounds $(FROUNDS) --batch $(FBATCH) --n_seed_per_class $(FSEEDPC) \
	--n_slopes $(FSLOPES) --splits 1 --first_split \$$SLURM_ARRAY_TASK_ID --resume --no_index \
	$(FEXTRA)

submit_frozen:
	@test -n "$$ES_DATA_DIR" || echo 'WARNING: ES_DATA_DIR is unset -- the job will not find the CSVs'
	@echo "submitting with $(PY)"
	@for M in $(FMETHOD); do \
	  mkdir -p cluster/logs/frozen/$$M; \
	  sbatch -J frozen_$$M --array=$(FARRAY) -n 1 -N 1 -t $(FTIME) -p $(FPART) \
	    --mem=$(FMEM) -o cluster/logs/frozen/$$M/out_%A_%a.txt \
	    -e cluster/logs/frozen/$$M/err_%A_%a.txt \
	    --wrap "$(FROZEN_WRAP)"; \
	done

dry_frozen:
	@for M in $(FMETHOD); do \
	  echo "sbatch -J frozen_$$M --array=$(FARRAY) -t $(FTIME) -p $(FPART) --mem=$(FMEM) --wrap:"; \
	  echo "    $(FROZEN_WRAP)"; \
	done
	@echo
	@echo '  -> $(NFMETH) submissions x $(FSPLITS) splits = $(shell echo $$(( $(NFMETH) * $(FSPLITS) ))) tasks'
	@echo '  -> results/$(TARGET)/$(TARGET)_w$(BLOCK)_<method>_r$(FROUNDS)x$(FBATCH)_s$(FSLOPES)_frozenmixed/'
	@echo '  -> after the arrays drain: make reindex'

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
