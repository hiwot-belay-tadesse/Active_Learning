"""What each cluster job runs, and the queue of jobs to submit.

Adapted from odyssey-tools/automatic-submission. Two roles in one file, as in the original:
``submit_batch.py`` imports ``QUEUE``/``DRYRUN``/``LOG_DIR`` from here to decide what to submit,
and each SLURM job re-enters through ``main()`` to do the work.

The unit of parallelism is ONE SPLIT. ``main.py --first_split k --splits 1`` runs exactly split
k and writes ``per_round_split<k>.csv`` + ``queries_split<k>.csv``; different k never touch the
same file, so jobs need no locking and no merge step. All jobs of one queue entry land in the
same run folder, because ``paths.run_dir()`` derives it from the configuration rather than from
anything job-specific.

    python cluster/submit_batch.py            # from the REPO ROOT, not from cluster/


Editing the queue
-----------------
Each entry is ``(name, dict-of-lists)`` and expands to one job per combination of the lists, so
``dict(split=range(20), methods=["eer random"])`` is 20 jobs. ``name`` only labels the job and
its log directory; it does not decide where results go.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# False submits to SLURM. True runs the queue one entry after another in the current shell --
# the local path, and the reason this file replaces typing `python main.py ...` by hand. To run
# a single acquisition locally, set this True and comment out the QUEUE entries you do not want.
DRYRUN = False

# SLURM stdout/stderr only. Results go wherever paths.run_dir() puts them (results/<target>/...),
# so this directory holds logs and nothing else.
LOG_DIR = os.path.join(REPO, "cluster", "logs")

# The interpreter the job uses. The repo's own .venv by default, because skactiveml needs
# sklearn >= 1.6 and the shared `Ban` env has an older stack; override on the cluster with
# AL_PYTHON=$(which python) after `micromamba activate`.
PYTHON = os.environ.get("AL_PYTHON") or os.path.join(REPO, ".venv", "bin", "python")

N_SPLITS = 20

# Every acquisition run so far, each paired with `random` in the same job. The pairing is not
# cosmetic: both methods share one seed set per target, so the gap is computed within a job and
# a method is never compared against a random baseline drawn on a different split.
#
# coreset and probcover share an entry because that is how they were run at 3 splits, and the
# existing results are reused by folder name -- splitting them now would make two new folders
# and forfeit the finished splits. The rest keep their existing pairings for the same reason.
#
# Cost per split, measured on 43 targets x 10 rounds (so 20 splits is ~20x this, in parallel):
#   eer      ~4300 s   (refits 2x per candidate; the only one that needs a long walltime)
#   laplace   ~166 s
#   typiclust / coreset / probcover: seconds -- geometry only, no model fits
# Entry names match the series labels on the per-person grid, so a job name, a log directory and
# a curve on the plot are all called the same thing. `eu` is the acquisition the plot labels EU;
# the METHOD flag is still `laplace`, because that string is written into the finished CSVs and
# the run folder name and renaming it there would orphan them.
QUEUE = [
    # ("smoke", dict(split=[0], methods=["random"],
    #             rounds=[1], max_targets=[3], tag=["smoke"])),
    # `refs=1` only here: the 100%-label reference models depend on the split and the priors,
    # not on the acquisition, so one entry computes them for everyone. They land in THIS run's
    # analysis/ and the grid reads them from there with `--refs`.
    ("random",    dict(split=list(range(N_SPLITS)), methods=["random"], refs=[1])),
    ("eu",        dict(split=list(range(N_SPLITS)), methods=["laplace"])),
    # ("eer",       dict(split=list(range(N_SPLITS)), methods=["eer"])),
    ("typiclust", dict(split=list(range(N_SPLITS)), methods=["typiclust"])),
    ("coreset",   dict(split=list(range(N_SPLITS)), methods=["coreset"])),
    ("probcover", dict(split=list(range(N_SPLITS)), methods=["probcover"])),
]

# Random is its own entry rather than riding along in every job. That is safe because the seed
# set is drawn from default_rng([seed, split_id, participant, 11]) (main.py) and does NOT depend
# on the method list, so random computed alone gets byte-identical seed sets and the gap stays a
# paired comparison. Two consequences:
#
#   * run_dir() derives the folder from the method list, so these are NEW folders -- e.g.
#     computer_work_w15_laplace_r10x3_s5, not ..._laplace-random_... . The finished 3-split runs
#     are not resumed into and stay on disk untouched; the array recomputes all 20 splits.
#   * main.py's per-split summary line prints a gap only when `random` is in the same job, so
#     these jobs print the run folder and nothing else. Read the gaps off the grid instead,
#     passing the random run through --also:
#         python plots/per_person_grid.py --run <eu run> \
#             --also <random run> <eer run> <typiclust run> <coreset run> <probcover run>

# Settings shared by every job. These must match across the queue entries you intend to compare,
# because the grid overlays curves from different runs on the assumption that split k means the
# same split everywhere -- which holds only if block/n_seed/seed/rounds/batch agree.
COMMON = dict(target="computer_work", block=15, rounds=100, batch=3,
              n_seed=3, n_seed_pos=1, n_slopes=5, seed=0)


def command(exp_kwargs):
    """-> the argv for one job."""
    kw = dict(COMMON)
    kw.update({k: v for k, v in exp_kwargs.items() if k != "methods"})
    argv = [PYTHON, os.path.join(REPO, "main.py"),
            "--first_split", str(exp_kwargs["split"]), "--splits", "1",
            "--methods", *str(exp_kwargs["methods"]).split(),
            # --resume makes re-submission idempotent: a job whose split is already written
            # exits without redoing it, so a partially failed array can simply be resubmitted.
            "--resume",
            # --no_index keeps 20 concurrent jobs off results/index.csv, which is a
            # read-modify-write of a single file. Rebuild it once afterwards with
            #     python paths.py --reindex
            "--no_index"]
    # main.py declares these with action="store_true", so they take NO value: emitting
    # `--refs 1` makes argparse fail with "unrecognized arguments: 1". A falsy value omits
    # the flag entirely rather than passing --refs 0, which would also switch it on.
    FLAGS = {"refs", "resume", "no_index", "reserve"}
    for k, v in kw.items():
        if k == "split":
            continue
        if k in FLAGS:
            if v:
                argv += [f"--{k}"]
            continue
        argv += [f"--{k}", str(v)]
    return argv


def run(exp_dir, exp_name, exp_kwargs):
    """Execute one job. Called in-process when DRYRUN, or by SLURM through main()."""
    argv = command(exp_kwargs)
    print(f"[{exp_name}] {' '.join(argv)}", flush=True)
    r = subprocess.run(argv, cwd=REPO)
    if r.returncode != 0:
        raise SystemExit(f"[{exp_name}] exited {r.returncode}")


def main():
    assert len(sys.argv) > 3, "usage: run.py <exp_dir> <exp_name> <json kwargs>"
    run(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]))


if __name__ == "__main__":
    main()
