"""Load ExtraSensory data for a named target.

One entry point: ``load_target("walking")`` or ``load_target("computer_work")`` returns the
same structure either way --

    data : {uuid: (X, y, day)}   rows in time order, X float, y int 0/1,
                                 day = 1-based calendar-day index within that user's usable rows
    cols : list[str]             the feature column names, identical across users

Two kinds of target exist in this dataset and they are NOT built the same way, which is the
whole reason this file has a registry rather than one function:

  * ``column``   -- the label is a column that is 1 when the activity is happening and 0 when it
                    is not, and rows where it is missing are dropped. COMPUTER_WORK and
                    IN_A_MEETING work this way. The negative class is "every other labelled
                    minute", so it is dominated by whatever else the person reported.
  * ``contrast`` -- the positive label is contrasted against a fixed list of mutually exclusive
                    activities, and anything outside that list is excluded rather than treated as
                    a negative. WALKING works this way (sitting / lying / standing / running /
                    bicycling are the negatives). A row with no known main activity is dropped.

Changing a target therefore changes the meaning of y, not just its name.

The feature side differs too: the activity targets use the phone+watch sensor groups minus the
ones that leak context, while walking is a watch-accelerometer-only problem by construction.

Data location: set ``ES_DATA_DIR`` to the directory holding the 61 ``*.features_labels.csv``
files. Nothing here writes to it.
"""
import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The CSVs are large (748 MB) and shared with the original monorepo, so they are referenced in
# place rather than copied. ES_DATA_DIR wins; the fallback assumes the old repo sits alongside.
DATA_DIR = os.environ.get("ES_DATA_DIR") or os.path.join(
    HERE, "..", "Banaware_AL-2", "DATA", "ExtraSensory")

# ----------------------------------------------------------------------------- sensor groups
# Every column is "<group>:<feature>"; these are the groups ExtraSensory ships.
ALL_GROUPS = ("raw_acc", "proc_gyro", "raw_magnet", "watch_acceleration", "watch_heading",
              "location", "location_quick_features", "audio_naive", "audio_properties",
              "discrete", "lf_measurements")

# Groups excluded from the activity targets: device orientation, compass heading, recording-gain
# metadata and the low-frequency sensors that are present for only a handful of participants.
DROP_GROUPS = ("raw_magnet", "watch_heading", "audio_properties", "lf_measurements")

# Sub-group columns that predict the label without sensing the activity. Dropping these is what
# takes the activity feature set from 175 columns to 124. Two distinct reasons, kept separate:
DROP_CONTEXT = (
    # (a) context, not evidence -- where/when the person is, not what they are doing
    "location", "location_quick_features",
    "raw_magnet",
    "discrete:time_of_day",
    "discrete:wifi_status",
    "discrete:battery_plugged", "discrete:battery_state",
    "discrete:ringer_mode", "discrete:on_the_phone",
    # (b) artifacts -- not about the person's situation at all
    "discrete:app_state",            # the ExtraSensory app's state = when they LABELLED
    "lf_measurements:relative_humidity",
    "lf_measurements:temperature_ambient",
    "lf_measurements:battery_level",  # monotone discharge within a day = a clock
    "audio_properties",               # recording gain: hardware metadata, not sound content
)

WALKING_NEGATIVES = ("label:LYING_DOWN", "label:SITTING", "label:OR_standing",
                     "label:FIX_running", "label:BICYCLING")


# ----------------------------------------------------------------------------- target registry
@dataclass(frozen=True)
class Target:
    """How one target's rows, labels and features are built."""
    name: str
    label_col: str
    mode: str                                   # "column" | "contrast"
    groups: tuple = ()                          # sensor groups to keep ("" = all but DROP_GROUPS)
    drop: tuple = ()                            # sub-group columns to remove
    negatives: tuple = ()                       # contrast mode only
    min_per_class: int = 20                     # eligibility threshold
    note: str = ""

    def columns_kept(self) -> tuple:
        return self.groups or tuple(g for g in ALL_GROUPS if g not in DROP_GROUPS)


ACTIVITY_GROUPS = tuple(g for g in ALL_GROUPS if g not in DROP_GROUPS)

TARGETS = {
    "computer_work": Target(
        name="computer_work", label_col="label:COMPUTER_WORK", mode="column",
        groups=ACTIVITY_GROUPS, drop=DROP_CONTEXT, min_per_class=20,
        note="124 features after DROP_CONTEXT; negatives are all other labelled minutes"),
    "in_a_meeting": Target(
        name="in_a_meeting", label_col="label:IN_A_MEETING", mode="column",
        groups=ACTIVITY_GROUPS, drop=DROP_CONTEXT, min_per_class=20,
        note="the original meeting target"),
    "walking": Target(
        name="walking", label_col="label:FIX_walking", mode="contrast",
        groups=("watch_acceleration",), drop=(), negatives=WALKING_NEGATIVES, min_per_class=10,
        note="46 watch-accelerometer features; negatives are five named activities only"),
}


def register_target(target: Target) -> None:
    """Add a target at runtime (e.g. a new label column) without editing this file."""
    TARGETS[target.name] = target


def get_target(target) -> Target:
    """Accept a Target, a registry name, or a raw ``label:XXX`` column for a column-mode target."""
    if isinstance(target, Target):
        return target
    key = str(target).lower()
    if key in TARGETS:
        return TARGETS[key]
    if str(target).startswith("label:"):
        return Target(name=str(target).split(":", 1)[1].lower(), label_col=str(target),
                      mode="column", groups=ACTIVITY_GROUPS, drop=DROP_CONTEXT)
    raise KeyError(f"unknown target {target!r}; known: {sorted(TARGETS)} "
                   f"(or pass a 'label:NAME' column directly)")


# ----------------------------------------------------------------------------- column selection
def feature_columns(df, groups, drop=()):
    """Feature names of `df` in the kept groups, minus `drop` (a group or a full column name)."""
    groups, keep = set(groups), []
    for c in df.columns:
        if c.startswith("label") or c == "timestamp":
            continue
        if c.split(":")[0] in groups and not any(c == d or c.startswith(d + ":") for d in drop):
            keep.append(c)
    return keep


# ----------------------------------------------------------------------------- label construction
def _y_column(df, t):
    """Column mode: y is the label column itself; rows where it is missing are dropped."""
    s = df[t.label_col]
    use = s.notna().to_numpy()
    return s.to_numpy()[use].astype(int), use


def _y_contrast(df, t):
    """Contrast mode: 1 = the positive activity, 0 = one of `negatives`, everything else dropped.

    The main activities are mutually exclusive in ExtraSensory, so a row can be positive or
    negative but never both; rows with no known main activity carry no usable label at all and
    are excluded rather than silently counted as negatives.
    """
    main = [t.label_col] + list(t.negatives)
    missing = [c for c in main if c not in df.columns]
    if missing:
        raise KeyError(f"{t.name}: missing label columns {missing}")
    assert (df[main] == 1).sum(axis=1).max() <= 1, "main activities are not mutually exclusive"
    y = np.full(len(df), -1, dtype=int)
    y[(df[t.label_col] == 1).to_numpy()] = 1
    y[(df[list(t.negatives)] == 1).any(axis=1).to_numpy()] = 0
    use = y != -1
    return y[use], use


_BUILDERS = {"column": _y_column, "contrast": _y_contrast}


# ----------------------------------------------------------------------------- loading
def participant_files(data_dir=DATA_DIR):
    files = sorted(glob.glob(os.path.join(data_dir, "*.features_labels.csv")))
    if not files:
        raise FileNotFoundError(
            f"no *.features_labels.csv under {data_dir!r}; set ES_DATA_DIR to the ExtraSensory "
            f"directory")
    return files


def load_target(target, data_dir=DATA_DIR, require_features=True, verbose=False):
    """-> ({uuid: (X, y, day)}, cols) for one target, over every participant that carries it.

    No eligibility filter is applied here -- that is `eligibility` / `filter_eligible`, kept
    separate so the counts before and after filtering are both visible.

    require_features drops rows whose kept features are entirely missing, which matters for
    walking (a minute with no watch data cannot be classified from watch features) and is a
    no-op for the phone+watch targets, where some group is essentially always present.
    """
    t = get_target(target)
    build = _BUILDERS[t.mode]
    data, cols = {}, None
    for f in participant_files(data_dir):
        uuid = os.path.basename(f).split(".")[0]
        df = pd.read_csv(f)
        if t.label_col not in df.columns:
            continue
        if cols is None:
            cols = feature_columns(df, t.columns_kept(), t.drop)
            if not cols:
                raise ValueError(f"{t.name}: no feature columns survive groups={t.columns_kept()}")
        y, use = build(df, t)
        if use.sum() == 0:
            continue
        X = df.loc[use, cols].to_numpy(float)
        if require_features:
            ok = np.isfinite(X).any(axis=1)
            X, y = X[ok], y[ok]
            ts = df.loc[use, "timestamp"].to_numpy()[ok]
        else:
            ts = df.loc[use, "timestamp"].to_numpy()
        if len(y) == 0:
            continue
        o = np.argsort(ts, kind="stable")                       # rows in time order
        dates = pd.to_datetime(ts[o], unit="s").date
        dmap = {d: i + 1 for i, d in enumerate(sorted(set(dates)))}
        data[uuid] = (X[o], y[o], np.array([dmap[d] for d in dates]))
    if verbose:
        n = sum(len(v[1]) for v in data.values())
        print(f"{t.name}: {len(data)} participants, {n} rows, {len(cols)} features "
              f"({t.mode} target on {t.label_col})")
    return data, cols


# ----------------------------------------------------------------------------- eligibility
def eligibility(data, min_per_class):
    """One row per participant: counts, positive rate, days, and the eligibility flag.

    Eligible = at least `min_per_class` rows of BOTH classes. A participant with 500 negatives
    and 3 positives cannot support a personal model or a meaningful per-person AUROC.
    """
    rows = []
    for u, (X, y, d) in data.items():
        pos, neg = int((y == 1).sum()), int((y == 0).sum())
        rows.append(dict(uuid=u, n=len(y), pos=pos, neg=neg,
                         pos_rate=float(y.mean()) if len(y) else np.nan,
                         days=len(np.unique(d)),
                         days_with_pos=len(np.unique(d[y == 1])) if pos else 0,
                         eligible=min(pos, neg) >= min_per_class))
    return pd.DataFrame(rows).sort_values("pos", ascending=False).reset_index(drop=True)


def filter_eligible(data, target, verbose=False):
    """Drop participants who fail the target's own min_per_class threshold. -> (data, table)."""
    t = get_target(target)
    el = eligibility(data, t.min_per_class)
    keep = set(el.loc[el.eligible, "uuid"])
    if verbose:
        print(f"{t.name}: {len(keep)} of {len(data)} participants have >= {t.min_per_class} "
              f"of each class")
    return {u: v for u, v in data.items() if u in keep}, el


# ----------------------------------------------------------------------------- convenience
_CACHE = {}


def load(target, data_dir=DATA_DIR, eligible_only=True, cache=True, verbose=False):
    """Load + filter in one call, memoised per process.

    Reading 61 CSVs costs more than a whole fold of model fits, and a 20-split run would
    otherwise repeat it 20 times, so the parsed result is cached on (target, data_dir,
    eligible_only).
    """
    t = get_target(target)
    key = (t.name, os.path.abspath(data_dir), bool(eligible_only))
    if cache and key in _CACHE:
        return _CACHE[key]
    data, cols = load_target(t, data_dir, verbose=verbose)
    if eligible_only:
        data, _ = filter_eligible(data, t, verbose=verbose)
    out = (data, cols)
    if cache:
        _CACHE[key] = out
    return out


def describe(data, cols, target=None):
    """Small summary dict, handy for a run header or a sanity assert."""
    y = np.concatenate([v[1] for v in data.values()]) if data else np.array([])
    return dict(target=get_target(target).name if target else None,
                participants=len(data), rows=int(len(y)),
                positives=int((y == 1).sum()),
                pos_rate=float(y.mean()) if len(y) else np.nan,
                features=len(cols))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="computer_work", help=f"one of {sorted(TARGETS)}")
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--all", action="store_true", help="skip the eligibility filter")
    a = ap.parse_args()
    d, c = load(a.target, a.data_dir, eligible_only=not a.all, cache=False, verbose=True)
    print(describe(d, c, a.target))
    print(eligibility(d, get_target(a.target).min_per_class).head(10).to_string(index=False))
