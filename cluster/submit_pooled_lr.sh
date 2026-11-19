#!/bin/bash
# Submit pooled_lr_al.py, one SLURM job per acquisition method. Run from the REPO ROOT.
#
#   ./cluster/submit_pooled_lr.sh                 # one job per method, all splits inside it
#   ./cluster/submit_pooled_lr.sh --by-split      # job array: one task per split, per method
#   DRY=1 ./cluster/submit_pooled_lr.sh           # print the commands, submit nothing
#   METHODS="coreset random" ./cluster/submit_pooled_lr.sh
#
# With no method argument it submits itself once per method; SLURM then re-enters it with the
# method as $1. Every method gets its own run folder named <PREFIX>-<method>, so no two jobs
# share a CSV and `*${PREFIX}*` globs the whole experiment.
#
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-12:00
#SBATCH -p shared
#SBATCH --mem=16000
#SBATCH -o cluster/logs/pooled_lr/%x_%j.out
#SBATCH -e cluster/logs/pooled_lr/%x_%j.err

set -euo pipefail

# ------------------------------------------------------------------ the experiment
METHODS=${METHODS:-"coreset typiclust probcover random"}
N_SPLITS=${N_SPLITS:-20}
PREFIX=${PREFIX:-pooled}          # bump for a new configuration, or old and new splits merge
TARGET=${TARGET:-computer_work}
BLOCK=${BLOCK:-15}
ROUNDS=${ROUNDS:-10}
BATCH=${BATCH:-3}
N_SEED_POS=${N_SEED_POS:-2}
N_SEED_NEG=${N_SEED_NEG:-2}
C=${C:-1.0}
SEED=${SEED:-0}

[ -f pooled_lr_al.py ] || { echo "run this from the repo root (no pooled_lr_al.py here)" >&2;
                            exit 1; }
: "${AL_PYTHON:?set it to your python, e.g. export AL_PYTHON=\$HOME/.conda/envs/AL2/bin/python}"

# ------------------------------------------------------------------ submit, or run as the job
BY_SPLIT=0
case "${1:-}" in
    --by-split) BY_SPLIT=1; shift ;;
esac

# No method left on the command line: this is the login-node pass, so submit one job per method.
if [ $# -eq 0 ]; then
    mkdir -p cluster/logs/pooled_lr
    # A string, not an array: "${a[@]-}" on an empty array is one EMPTY argument, which sbatch
    # reads as the script name. Unquoted and empty, this expands to nothing.
    ARRAY=""
    [ "$BY_SPLIT" = "1" ] && ARRAY="--array=0-$((N_SPLITS - 1))"
    for m in $METHODS; do
        echo "sbatch -J plr_${m} ${ARRAY} $0 ${m}" \
            "-> results/${TARGET}/global_lr/${TARGET}_w${BLOCK}_${PREFIX}-${m}_r${ROUNDS}x${BATCH}_s0"
        [ "${DRY:-0}" = "1" ] || sbatch -J "plr_${m}" $ARRAY "$0" "${m}"
    done
    echo "when the queue drains:  \$AL_PYTHON paths.py --reindex"
    exit 0
fi

METHOD=$1
FOLDER="${TARGET}_w${BLOCK}_${PREFIX}-${METHOD}_r${ROUNDS}x${BATCH}_s0"

# one core per job; BLAS would otherwise spawn a thread per visible CPU
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# The ExtraSensory CSVs are not in the repo; point this at them on the cluster.
# export ES_DATA_DIR=/n/holylabs/.../ExtraSensory
# Unset writes under the repo; set it to move the results tree onto scratch.
# export AL_RESULTS_DIR=/n/holyscratch01/.../Active_Learning_results

# An array task owns one split; a plain job walks every split itself.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    FIRST=$SLURM_ARRAY_TASK_ID
    HOWMANY=1
else
    FIRST=0
    HOWMANY=$N_SPLITS
fi

exec "$AL_PYTHON" -u pooled_lr_al.py \
    --methods "$METHOD" \
    --name "$FOLDER" \
    --first_split "$FIRST" --splits "$HOWMANY" \
    --target "$TARGET" --block "$BLOCK" \
    --rounds "$ROUNDS" --batch "$BATCH" \
    --n_seed_pos "$N_SEED_POS" --n_seed_neg "$N_SEED_NEG" \
    --C "$C" --seed "$SEED" \
    --resume \
    --no_index
