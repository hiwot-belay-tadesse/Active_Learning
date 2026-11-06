"""Expand cluster/run.py's QUEUE into one SLURM job per parameter combination and submit them.

    python cluster/submit_batch.py            # from the repo root
    python cluster/submit_batch.py --dry      # print the jobs, submit nothing

Adapted from odyssey-tools/automatic-submission/submit_batch.py, with three changes made
because the original's assumptions do not hold here:

* The generated script is named after the FULL parameter set, not just the queue entry. Upstream
  writes ``tmp/slurm_<exp_name>.sh`` for every combination, so 20 splits overwrite one file --
  harmless only because sbatch reads it before the next iteration rewrites it, and unrecoverable
  afterwards when you want to know what was actually submitted.
* ``exp_dir`` holds SLURM logs only. Upstream puts results there and tells you in a docstring to
  make the directory a function of the kwargs; here ``paths.run_dir()`` already owns the results
  layout, and every split of one queue entry belongs in the SAME run folder.
* Jobs are submitted with ``--no_index``; call ``python paths.py --reindex`` once the array has
  drained. Otherwise 20 jobs race on results/index.csv.
"""
import argparse
import copy
import shutil
import itertools
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run import DRYRUN, LOG_DIR, PYTHON, QUEUE, command, run    # noqa: E402

TMP_DIR = os.path.join(HERE, "tmp")
TEMPLATE = os.path.join(HERE, "template.sh")


def job_name(exp_name, exp_kwargs):
    """A name unique to this parameter combination -- so nothing overwrites anything."""
    bits = [exp_name] + [f"{k}{str(v).replace(' ', '-')}" for k, v in sorted(exp_kwargs.items())
                         if k != "methods"]
    return "_".join(bits)


def submit_job(template, exp_name, exp_kwargs, dry=False):
    name = job_name(exp_name, exp_kwargs)
    exp_dir = os.path.join(LOG_DIR, exp_name)          # logs only; results live under results/
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    script = copy.deepcopy(template)
    script = script.replace("EXPNAME", name)
    script = script.replace("EXPDIR", exp_dir)
    script = script.replace("PYTHONBIN", PYTHON)
    script = script.replace("RUNPY", os.path.join(HERE, "run.py"))
    script = script.replace("KWARGS", "'{}'".format(json.dumps(exp_kwargs)))

    sname = os.path.join(TMP_DIR, f"slurm_{name}.sh")
    with open(sname, "w") as fh:
        fh.write(script)

    if dry:
        print(f"  would submit {name}: {' '.join(command(exp_kwargs))}")
        return
    if not DRYRUN:
        # subprocess raises FileNotFoundError when sbatch is absent, which reads like a bug in
        # this script rather than "you are not on the cluster". Say the useful thing instead.
        if shutil.which("sbatch") is None:
            raise SystemExit(
                "sbatch not found -- this machine has no SLURM.\n"
                "  To run the queue here instead, set DRYRUN = True in cluster/run.py\n"
                "  (or `make run METHOD=... SPLITS=...` for a single configuration).")
        ret = subprocess.call(["sbatch", sname])
        if ret != 0:
            print(f"error {ret} submitting {sname}")
    else:
        run(exp_dir, name, exp_kwargs)                 # in-process, one after another


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry", action="store_true", help="print the jobs, submit nothing")
    a = ap.parse_args()

    for label, path in (("repo", HERE), ("python", PYTHON)):
        if " " in path:
            print(f"WARNING: the {label} path contains a space ({path}).\n"
                  f"         SLURM does not honour quoting in #SBATCH -o/-e, so job logs will\n"
                  f"         break. Check the repo out somewhere without spaces on the cluster.")

    with open(TEMPLATE) as fh:
        template = fh.read()

    n = 0
    for exp_name, params in QUEUE:
        for vals in itertools.product(*list(params.values())):
            submit_job(template, exp_name, dict(zip(params.keys(), vals)), dry=a.dry)
            n += 1
    mode = "printed" if a.dry else ("ran locally (DRYRUN)" if DRYRUN else "submitted")
    print(f"{n} job(s) {mode}")
    if not a.dry and not DRYRUN:
        print("when the array drains:  python paths.py --reindex")


if __name__ == "__main__":
    main()
