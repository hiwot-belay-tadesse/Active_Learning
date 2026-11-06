#!/bin/bash
#SBATCH -J coreset_split16
#SBATCH -n 1                     # one core: the work is a single-threaded statsmodels fit loop
#SBATCH -N 1
#SBATCH -t 0-04:00               # EER is the slow one at ~1.2 h/split; 4 h leaves headroom
#SBATCH -p shared                # NOT `test` -- that partition caps at ~1 h and is for debugging
#SBATCH --mem=16000              # 748 MB of CSV read per job, then ~11k x 124 float64 in memory
#SBATCH -o /Users/hiwotbelaytadesse/Harvard University Dropbox/Hiwot Belay Tadesse/Active_Learning/cluster/logs/coreset/out_%j.txt
#SBATCH -e /Users/hiwotbelaytadesse/Harvard University Dropbox/Hiwot Belay Tadesse/Active_Learning/cluster/logs/coreset/err_%j.txt

# Keep the numeric libraries single-threaded. Each job asks for one core, and BLAS otherwise
# spawns a thread per visible CPU -- oversubscribing the node and slowing every job down.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# The ExtraSensory CSVs are not in the repo. Point this at them on the cluster filesystem;
# load_data.py falls back to a sibling-repo path that will not exist there.
# export ES_DATA_DIR=/n/holyscratch01/.../ExtraSensory

# Results tree. Leave unset to write under the repo; set it to move the whole tree onto scratch.
# export AL_RESULTS_DIR=/n/holyscratch01/.../Active_Learning_results

# Quoted: a repo path containing spaces would otherwise be split into separate arguments.
# NOTE the #SBATCH -o/-e lines above CANNOT be quoted -- SLURM parses those itself and does not
# honour shell quoting -- so a path with spaces in it breaks the log files regardless. Check out
# the repo somewhere space-free on the cluster; submit_batch.py warns if it is not.
"/Users/hiwotbelaytadesse/Harvard University Dropbox/Hiwot Belay Tadesse/Active_Learning/.venv/bin/python" -u "/Users/hiwotbelaytadesse/Harvard University Dropbox/Hiwot Belay Tadesse/Active_Learning/cluster/run.py" "/Users/hiwotbelaytadesse/Harvard University Dropbox/Hiwot Belay Tadesse/Active_Learning/cluster/logs/coreset" coreset_split16 '{"split": 16, "methods": "coreset"}'
