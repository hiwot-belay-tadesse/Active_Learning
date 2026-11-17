#!/usr/bin/env bash
# Sensitivity of the framework's one free knob: how many labels Phase 1 gets.
#
#   bash two_stage/run_ncalib.sh                  # n_calib in {3, 9, 12, 18, 24}, 10 splits
#   NCALIBS="3 6 9" SPLITS=5 bash two_stage/run_ncalib.sh
#
# n_calib = 0 and n_calib = the whole budget (30) are already arms of the main benchmark
# (`twfeg` and `calib`), so only the interior values need running. Together they trace the
# schedule's full dose-response curve, 0 -> 30 labels of Phase 1, which is the one figure that
# says where the phase boundary actually belongs. Each value gets its own run folder via
# --tag, because the folder name is derived from the configuration and n_calib is not part of it.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-./.venv/bin/python}
SPLITS=${SPLITS:-10}
PAR=${PAR:-6}
ROUNDS=${ROUNDS:-10}
NCALIBS=${NCALIBS:-"3 9 12 18 24"}
LOGS=two_stage/logs
mkdir -p "$LOGS"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

for NC in $NCALIBS; do
  echo "=== n_calib = $NC ==="
  "$PY" two_stage/run_experiment.py --methods twostage --n_calib "$NC" --tag "nc$NC" \
      --rounds "$ROUNDS" --first_split 0 --splits 1 --resume > "$LOGS/nc${NC}_split0.log" 2>&1
  # -S 65536: see the note in run_all.sh -- macOS xargs -I caps the assembled command at 255
  # bytes and fails the job before it starts; this command line is well over that.
  seq 1 $((SPLITS-1)) | xargs -P "$PAR" -S 65536 -I{} sh -c \
      "\"$PY\" two_stage/run_experiment.py --methods twostage --n_calib $NC --tag nc$NC \
          --rounds $ROUNDS --first_split {} --splits 1 --resume \
          > $LOGS/nc${NC}_split{}.log 2>&1 || echo 'nc$NC split {} FAILED'"
  echo "n_calib = $NC done"
done
