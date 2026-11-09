# Active_Learning -- common commands.
#
#   make run METHOD=coreset SPLITS=20     one acquisition locally, no editing of cluster/run.py
#   make submit                           the whole QUEUE to SLURM
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

PY      ?= ./.venv/bin/python
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

TAGFLAG := $(if $(TAG),--tag $(TAG),)
RUNFLAG := $(if $(RUN),--run $(RUN),--contains r$(ROUNDS)x$(BATCH))

.PHONY: help run submit local dry refs sweep grid bar reindex selftest clean-logs

help:
	@sed -n '2,19p' Makefile | sed 's/^# \{0,1\}//'

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
	$(PY) cluster/submit_batch.py

local:
	$(PY) cluster/submit_batch.py

dry:
	$(PY) cluster/submit_batch.py --dry

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
