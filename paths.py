"""Where results go. One root, one deterministic folder per run, one index of every run.

The problem this solves: results scattered under ad-hoc directory names, where finding the run
behind a figure means remembering what it was called. Here the folder name is *derived from the
configuration*, so the same settings always land in the same place and different settings never
collide:

    results/<target>/<target>_w<W>_<methods>_r<rounds>x<batch>_s<slopes>[_<tag>]/
        config.json          every argument, plus versions and timestamps
        per_round_split0.csv  one pair per split, appended as splits finish
        queries_split0.csv
        analysis/            derived tables (anything a diagnostic script computes)
        figures/             plots made from this run

Three consequences worth knowing:

* Re-running identical settings writes to the same folder, which makes a run resumable -- a
  finished split is detectable by its files. Use ``--tag`` when you deliberately want a
  variant kept side by side, and ``reserve=True`` when you want the old one preserved untouched.
* ``results/index.csv`` gets a row per run with its settings and a pointer to the folder, so
  "which runs exist, and what were they?" is one ``pd.read_csv`` rather than a directory crawl.
* Derived output belongs in that run's ``analysis/`` and ``figures/``, never beside the code,
  so deleting a run deletes everything about it.

``AL_RESULTS_DIR`` moves the whole tree (e.g. onto a scratch disk) without touching any script.
"""
import json
import os
import platform
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.environ.get("AL_RESULTS_DIR") or os.path.join(HERE, "results")
INDEX = os.path.join(RESULTS_ROOT, "index.csv")


# ----------------------------------------------------------------------------- naming
def run_id(target, block, methods, rounds, batch, n_slopes, tag=None):
    """The folder name for one configuration -- deterministic, readable, collision-free.

    Only the settings that change what the numbers mean go in the name. The split count does
    not: splits accumulate inside a run, so running 5 splits and later 15 more is the same run.
    """
    parts = [str(target), f"w{block}", "-".join(sorted(methods)),
             f"r{rounds}x{batch}", f"s{n_slopes}"]
    if tag:
        parts.append(str(tag))
    return "_".join(parts)


# One level between the target and the run folder, so `results/computer_work/` does not become
# a flat list of thirty near-identical names. The three scopes are the three designs:
#
#   personal     one participant: their own pool, their own model (personal_al.py)
#   global       one model PER TARGET, personalised, buying from the whole cohort (main.py)
#   full_global  ONE model for everybody, one acquisition stream for the cohort (cohort_al.py)
#   global_lr    one PLAIN LR per target, buying from everyone (pooled_lr_al.py) -- main.py's
#                pool and seed with no personalisation, so the two isolate the model class
#
# Discovery accepts a run at either depth, so folders written before this existed keep working.
SCOPES = ("global", "personal", "full_global", "global_lr")


def run_dir(target, block, methods, rounds, batch, n_slopes, tag=None, root=RESULTS_ROOT,
            create=True, reserve=False, scope=None, name=None):
    """-> the directory for this configuration, created if needed.

    reserve=True never reuses a directory that already holds results: it appends _run2, _run3,
    ... instead. Use it when a previous run must not be touched; leave it off to resume.

    scope is "global" or "personal" and becomes a folder between the target and the run; None
    writes directly under the target, the layout used before scopes existed.

    name replaces the derived folder name outright. The derived name encodes every setting that
    changes the numbers, which is what makes two runs of one configuration land in the same
    folder and resume each other -- so a caller that passes `name` takes that job on itself:
    two different configurations under one name will merge silently.
    """
    if scope is not None and scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES} or None, got {scope!r}")
    rid = name or run_id(target, block, methods, rounds, batch, n_slopes, tag)
    path = os.path.join(root, str(target), *( (scope,) if scope else () ), rid)
    if reserve and _has_results(path):
        i = 2
        while _has_results(f"{path}_run{i}"):
            i += 1
        path = f"{path}_run{i}"
    if create:
        for sub in ("", "analysis", "figures"):
            os.makedirs(os.path.join(path, sub), exist_ok=True)
    return path


def _has_results(path):
    return os.path.isdir(path) and any(f.startswith("per_round_split")
                                       for f in os.listdir(path))


# ----------------------------------------------------------------------------- inside a run
def split_files(path, k):
    """The two CSVs for split k."""
    return (os.path.join(path, f"per_round_split{k}.csv"),
            os.path.join(path, f"queries_split{k}.csv"))


def done_splits(path):
    """Split indices already written -- what makes a run resumable."""
    out = []
    for f in sorted(os.listdir(path)) if os.path.isdir(path) else []:
        if f.startswith("per_round_split") and f.endswith(".csv"):
            out.append(int(f[len("per_round_split"):-len(".csv")]))
    return sorted(out)


def load_run(path, what="per_round"):
    """Concatenate every split's CSV into one frame. `what` is 'per_round' or 'queries'."""
    import glob
    files = sorted(glob.glob(os.path.join(path, f"{what}_split*.csv")))
    if not files:
        raise FileNotFoundError(f"no {what}_split*.csv in {path}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def analysis_path(path, name):
    """A file under this run's analysis/ -- for derived tables."""
    os.makedirs(os.path.join(path, "analysis"), exist_ok=True)
    return os.path.join(path, "analysis", name)


def figure_path(path, name):
    """A file under this run's figures/ -- for plots."""
    os.makedirs(os.path.join(path, "figures"), exist_ok=True)
    return os.path.join(path, "figures", name)


# ----------------------------------------------------------------------------- provenance
def write_config(path, args, extra=None):
    """Save every setting beside the numbers, so a run can be read a year later.

    `splits` and `first_split` describe the JOB, not the run: one split per job means twenty
    array tasks overwrite this file, so the last writer's slice is what survives. `_splits_done`
    is therefore derived from the CSVs on disk unioned with this job's own range -- never read
    back from the file, so a lost update cannot shrink it. Read that, or `done_splits()`, and
    never `splits`.

    Versions are included because results depend on them: scikit-activeml's TypiClust default k
    and statsmodels' optimiser have both changed across releases.
    """
    cfg = dict(vars(args)) if hasattr(args, "__dict__") else dict(args)
    cfg.update(extra or {})
    first = int(cfg.get("first_split") or 0)
    mine = range(first, first + int(cfg.get("splits") or 1))
    cfg["_splits_done"] = sorted(set(done_splits(path)) | set(mine))
    cfg["_written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    cfg["_python"] = platform.python_version()
    for mod in ("numpy", "pandas", "sklearn", "statsmodels", "skactiveml"):
        try:
            cfg[f"_{mod}"] = __import__(mod).__version__
        except Exception:
            pass
    # Atomic: concurrent array tasks sharing this path produced truncated JSON ("}}") before.
    dst = os.path.join(path, "config.json")
    tmp = f"{dst}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(cfg, fh, indent=2, default=str)
    os.replace(tmp, dst)
    return cfg


def read_config(path):
    """-> the config dict. Salvages the trailing garbage a pre-atomic concurrent write left."""
    with open(os.path.join(path, "config.json")) as fh:
        raw = fh.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cfg, end = json.JSONDecoder().raw_decode(raw.lstrip())
        print(f"  {path}/config.json: ignored {len(raw.lstrip()) - end} trailing byte(s)")
        return cfg


# ----------------------------------------------------------------------------- the index
def update_index(path, cfg=None, root=RESULTS_ROOT):
    """Add or refresh this run's row in results/index.csv. -> the index frame."""
    cfg = cfg or (read_config(path) if os.path.exists(os.path.join(path, "config.json")) else {})
    row = dict(run=os.path.basename(path), path=os.path.relpath(path, root),
               target=cfg.get("target"), block=cfg.get("block"),
               methods=" ".join(cfg.get("methods", []) or []),
               rounds=cfg.get("rounds"), batch=cfg.get("batch"),
               n_slopes=cfg.get("n_slopes"), tag=cfg.get("tag"),
               splits=len(done_splits(path)), updated=time.strftime("%Y-%m-%d %H:%M"))
    os.makedirs(root, exist_ok=True)
    idx = pd.read_csv(INDEX) if os.path.exists(INDEX) else pd.DataFrame(columns=list(row))
    idx = idx[idx["run"] != row["run"]] if "run" in idx else idx
    idx = pd.concat([idx, pd.DataFrame([row])], ignore_index=True)
    idx.sort_values(["target", "run"]).to_csv(INDEX, index=False)
    return idx


def run_glob(target, *parts, root=RESULTS_ROOT):
    """Glob inside every run folder of one target, whatever depth it sits at.

    ``run_glob("computer_work", "*cohort*", "queries_split*.csv")`` matches both
    ``<root>/computer_work/<run>/queries_split0.csv`` and
    ``<root>/computer_work/global/<run>/queries_split0.csv``. Callers that used to build the
    one-level pattern themselves silently stopped finding anything once runs moved into a
    scope folder, which is the sort of failure that reads as "the run is missing".
    """
    import glob as _glob
    base = os.path.join(root, str(target))
    hits = set(_glob.glob(os.path.join(base, *parts)))
    hits |= set(_glob.glob(os.path.join(base, "*", *parts)))
    return sorted(hits)


def run_folders(target, pattern="*", root=RESULTS_ROOT):
    """-> the run DIRECTORIES of one target, at either depth, newest-sorted by name."""
    return [d for d in run_glob(target, pattern, root=root) if os.path.isdir(d)]


def reindex(root=RESULTS_ROOT):
    """Rebuild results/index.csv from the run folders actually on disk. -> the index frame.

    update_index() only adds and refreshes, so a deleted run keeps its row forever. This drops
    the whole index and walks ``<root>/<target>/<run>/config.json`` instead, which both prunes
    stale rows and repairs an index that parallel jobs raced on. Runs are written with
    ``--no_index`` on the cluster precisely so this is called once, after the array drains.
    """
    if os.path.exists(INDEX):
        os.remove(INDEX)
    found, skipped = [], []
    for target in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        tdir = os.path.join(root, target)
        if not os.path.isdir(tdir):
            continue
        # A directory holding config.json is a run; one that does not is a scope folder, so
        # look one level inside it. Nothing deeper is searched -- two levels is the layout.
        candidates = []
        for name in sorted(os.listdir(tdir)):
            path = os.path.join(tdir, name)
            if not os.path.isdir(path):
                continue
            if os.path.exists(os.path.join(path, "config.json")):
                candidates.append((name, path))
            else:
                candidates += [(f"{name}/{sub}", os.path.join(path, sub))
                               for sub in sorted(os.listdir(path))
                               if os.path.isdir(os.path.join(path, sub))]
        for name, path in candidates:
            if os.path.exists(os.path.join(path, "config.json")):
                try:
                    update_index(path, root=root)
                except json.JSONDecodeError as e:
                    # One malformed config.json must not cost the whole index. Reindex is the
                    # recovery step after a cluster array drains, so it has to be the one call
                    # that always completes; the bad folder is named and skipped.
                    print(f"  skipped {name}: config.json is not valid JSON ({e})")
                    skipped.append(name)
                    continue
                found.append(name)
    if skipped:
        print(f"{len(skipped)} run(s) left out of the index: {', '.join(skipped)}")
    return runs(root=root), found


def runs(target=None, root=RESULTS_ROOT):
    """Every indexed run, optionally filtered -- the "what do I have?" call."""
    if not os.path.exists(INDEX):
        return pd.DataFrame()
    idx = pd.read_csv(INDEX)
    return idx[idx.target == target] if target else idx


def find_run(target=None, contains=None, root=RESULTS_ROOT):
    """Absolute path of the most recently updated run matching the filters.

    Everything here reads results/index.csv, which the cluster jobs deliberately do NOT write
    (they run with --no_index to avoid racing on one file). An empty or missing index therefore
    means "nobody has reindexed yet", not "no results" -- and it used to surface as a bare
    KeyError: 'run' from the filter below, which says nothing about the fix.
    """
    idx = runs(target, root)
    if not len(idx) or "run" not in idx:
        raise FileNotFoundError(
            f"results index {INDEX} is empty or missing -- nothing to search.\n"
            f"  Cluster jobs run with --no_index, so rebuild it first:  make reindex\n"
            f"  (or pass the folder directly:  --run results/<target>/global/<run folder>)")
    if contains:
        idx = idx[idx["run"].astype(str).str.contains(contains, regex=False)]
    if not len(idx):
        raise FileNotFoundError(f"no run for target={target!r} contains={contains!r} under {root}")
    return os.path.join(root, idx.sort_values("updated").iloc[-1]["path"])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default=None)
    ap.add_argument("--reindex", action="store_true",
                    help="rebuild index.csv from the folders on disk (prunes deleted runs)")
    a = ap.parse_args()
    if a.reindex:
        _, found = reindex()
        print(f"reindexed {len(found)} run(s)")
    idx = runs(a.target)
    print(f"results root: {RESULTS_ROOT}")
    print(idx.to_string(index=False) if len(idx) else "(no runs indexed yet)")
