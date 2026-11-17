#!/usr/bin/env bash
# Run the benchmark over several day splits in parallel on one machine.
#
#   bash two_stage/run_all.sh                 # 10 splits, 6 at a time, every arm
#   SPLITS=20 PAR=5 bash two_stage/run_all.sh
#   METHODS="twostage random" bash two_stage/run_all.sh
#
# One process per split, every arm inside it (see run_experiment.py for why). Splits never touch
# each other's files -- per_round_split<k>.csv / queries_split<k>.csv -- so no locking is needed,
# and --resume makes a re-run skip the splits already on disk. Logs go to two_stage/logs/.
#
# The first job is launched alone and allowed to write config.json before the rest start: every
# process writes the same configuration to the same path, and two of them interleaving would
# leave truncated JSON behind.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-./.venv/bin/python}
SPLITS=${SPLITS:-10}
FIRST=${FIRST:-0}
PAR=${PAR:-6}
ROUNDS=${ROUNDS:-10}
BATCH=${BATCH:-3}
NCALIB=${NCALIB:-6}
METHODS=${METHODS:-}
EXTRA=${EXTRA:-}
LOGS=two_stage/logs
# Different method sets write different runs; without a tag they would overwrite each other's
# logs (the results themselves are keyed by run folder and were never at risk).
LOGTAG=${LOGTAG:-$(echo "${METHODS:-default}" | tr -c 'A-Za-z0-9' '_' | cut -c1-24)_}
mkdir -p "$LOGS"

# single-threaded BLAS: each job is one statsmodels fit loop, and a thread per core per job
# oversubscribes the machine and slows every job down
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

ARGS="--rounds $ROUNDS --batch $BATCH --n_calib $NCALIB --resume $EXTRA"
[ -n "$METHODS" ] && ARGS="$ARGS --methods $METHODS"

echo "splits $FIRST..$((FIRST+SPLITS-1)), $PAR at a time, rounds=$ROUNDS batch=$BATCH n_calib=$NCALIB"
"$PY" two_stage/run_experiment.py --first_split "$FIRST" --splits 1 $ARGS \
    > "$LOGS/${LOGTAG}split$FIRST.log" 2>&1
echo "split $FIRST done (config.json written)"

# -S 65536: macOS xargs caps the command it assembles for -I at 255 BYTES and then fails with
# "command line cannot be assembled, too long" -- per job, silently, before anything runs. With
# METHODS set the line crosses that limit, so the jobs never launched and only the sequential
# first split appeared. GNU xargs has no such default. Keep -S whenever -I is used here.
seq $((FIRST+1)) $((FIRST+SPLITS-1)) | xargs -P "$PAR" -S 65536 -I{} sh -c \
    "\"$PY\" two_stage/run_experiment.py --first_split {} --splits 1 $ARGS \
        > $LOGS/${LOGTAG}split{}.log 2>&1 && echo 'split {} done' || echo 'split {} FAILED'"
echo "all splits finished"
grep -h "macro-PR-AUC vs random" "$LOGS"/${LOGTAG}split*.log | sort | tail -40
