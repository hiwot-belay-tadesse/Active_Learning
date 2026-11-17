"""What each cluster job runs, and the queue of jobs to submit.

submit_batch.py imports QUEUE/DRYRUN/LOG_DIR from here; each SLURM job re-enters through main().
The unit of parallelism is one split, so jobs never touch the same file and need no merge step.

    python cluster/submit_batch.py            # from the REPO ROOT
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# True runs the queue in this shell instead of submitting
DRYRUN = False

# SLURM stdout/stderr only; results go where paths.run_dir() puts them
LOG_DIR = os.path.join(REPO, "cluster", "logs")

# .venv by default (skactiveml needs sklearn >= 1.6); on the cluster set
# AL_PYTHON=$(which python) after activating the env
PYTHON = os.environ.get("AL_PYTHON") or os.path.join(REPO, ".venv", "bin", "python")

N_SPLITS = 20

# Entry names match the grid's series labels. `eu` runs METHOD=laplace: that string is in the
# finished CSVs and folder names, so renaming it would orphan them.
# Cost per split (43 targets x 10 rounds): eer ~4300 s, laplace ~166 s, the geometry-only
# methods seconds.
QUEUE = [
    # ("smoke", dict(split=[0], methods=["random"],
    #             rounds=[1], max_targets=[3], tag=["smoke"])),
    # refs=1 only here: references are acquisition-independent, so one entry covers everyone
    ("random",    dict(split=list(range(N_SPLITS)), methods=["random"], refs=[1])),
    ("eu",        dict(split=list(range(N_SPLITS)), methods=["laplace"])),
    # laplace_var: ranks a candidate by its own variance, not the reduction it buys. Static
    # top-b, no Sherman-Morrison between picks, so a batch can be adjacent blocks of one person.
    # ("laplace_var", dict(split=list(range(N_SPLITS)), methods=["laplace_var"])),
    # ("eer",       dict(split=list(range(N_SPLITS)), methods=["eer"])),
    ("typiclust", dict(split=list(range(N_SPLITS)), methods=["typiclust"])),
    ("coreset",   dict(split=list(range(N_SPLITS)), methods=["coreset"])),
    ("probcover", dict(split=list(range(N_SPLITS)), methods=["probcover"])),
]

# Random is its own entry: the seed set does not depend on the method list, so the gap is still
# paired. Each method therefore gets its own run folder, and a job prints no gap -- read those
# off the grid, passing the other runs through --also.

# Shared settings. These must agree across entries you intend to compare on one grid.
# seed_source="cohort" seeds from every participant; n_seed/n_seed_pos are then unused but kept
# so switching back needs no edit.
COMMON = dict(target="computer_work", block=15, rounds=10, batch=3,
              seed_source="cohort", n_seed_per_person=2,
              n_seed=3, n_seed_pos=1, n_slopes=5, seed=0)


def command(exp_kwargs):
    """-> the argv for one job."""
    kw = dict(COMMON)
    kw.update({k: v for k, v in exp_kwargs.items() if k != "methods"})
    argv = [PYTHON, os.path.join(REPO, "main.py"),
            "--first_split", str(exp_kwargs["split"]), "--splits", "1",
            "--methods", *str(exp_kwargs["methods"]).split(),
            "--resume",          # re-submission is idempotent
            "--no_index"]        # concurrent jobs would race on index.csv; `paths.py --reindex` after
    # store_true flags: emit the bare flag, or nothing at all
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
