#!/bin/bash
#SBATCH -J EXPNAME
#SBATCH -n 1                     # single-threaded statsmodels fit loop
#SBATCH -N 1
#SBATCH -t 0-12:00
#SBATCH -p shared                # not `test`: that partition caps at ~1 h
#SBATCH --mem=16000
#SBATCH -o EXPDIR/out_%j.txt
#SBATCH -e EXPDIR/err_%j.txt

# one core per job; BLAS would otherwise spawn a thread per visible CPU
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# The ExtraSensory CSVs are not in the repo; point this at them on the cluster.
# export ES_DATA_DIR=/n/holyscratch01/.../ExtraSensory

# Unset writes under the repo; set it to move the results tree onto scratch.
# export AL_RESULTS_DIR=/n/holyscratch01/.../Active_Learning_results

# Quoted for paths with spaces -- but #SBATCH -o/-e above cannot be, so check the repo out
# somewhere space-free on the cluster. submit_batch.py warns if it is not.
"PYTHONBIN" -u "RUNPY" "EXPDIR" EXPNAME KWARGS
