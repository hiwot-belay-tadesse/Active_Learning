"""The per-participant grid: one panel per target, the acquisition against random.

    python plots/per_person_grid.py                                  # newest computer_work run
    python plots/per_person_grid.py --target walking
    python plots/per_person_grid.py --run results/computer_work/<run folder> --rows 4

Each panel is one participant: test AUROC against labels acquired, averaged over the run's
splits with a +/-1 SE band, the acquisition in purple and random in orange. Panels are ordered
by the acquisition-random gap at the last round, best first, and the final panel is the mean
over all targets.

Read the ordering with care. Sorting panels by a noisy quantity always produces a left-to-right
gradient, so the leading panels look good even when every participant has the same true gap.
The run summary printed to stdout reports the split-half rank correlation of the per-person
gap, which is the quick check on whether that ordering means anything: near zero says the
ranking is luck. The figure itself carries no title -- the series legend sits there instead --
so the configuration it came from is recorded in the run's ``config.json``, not on the plot.

Two dashed reference lines are drawn per panel when ``analysis/reference_auroc.csv`` exists
(written by reference_models.py), both fitted on 100% of the split's labels:

  crimson   mixed_full  -- the same mixed-effects model with every label: the personalised
                           ceiling, what the curve is climbing towards
  slate     global_prior -- one pooled model for everybody, no random effects, carrying the
                            same prior as the mixed model's fixed effects: the floor
                            personalisation is supposed to beat. global_full (the unpenalised
                            pooled fit) is drawn only with --show_unpenalised, since comparing
                            against it confounds personalisation with regularisation

A curve below the slate line is a target that personalisation is hurting at this budget; a
curve reaching the crimson line has nothing left to gain from more labels.

Figures are written inside the run's own ``figures/`` directory, so deleting a run removes
everything derived from it.
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MaxNLocator  # noqa: E402
import numpy as np                                                       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import figure_path, find_run, load_run, read_config           # noqa: E402

COL = {"typiclust": "purple", "coreset": "seagreen", "random": "darkorange",
       "probcover": "dodgerblue", "uncertainty": "steelblue", "eer": "magenta",
       "laplace": "black", "laplace_var": "gold",
       # The two cohort-evaluation arms differ from `laplace` ONLY in whose variance is being
       # reduced, so they are the comparison a reader is here to make; they fell through to the
       # shared "grey" default and were indistinguishable from each other.
       "laplace_pool": "teal", "laplace_people": "saddlebrown"}

# Display names. The method STRING stays as it is written in the per-round CSVs and the run
# folder name -- renaming it there would orphan every finished run -- so the relabelling is
# cosmetic and lives only here. `--order_by` and `--methods` still take the real name.
LABEL = {"laplace": "EU", "laplace_var": "EU (own variance)",
         "laplace_pool": "EU (cohort, per row)", "laplace_people": "EU (cohort, per person)"}
# Short names for the reference models, used in the panel titles where the full name would
# not fit beside its value.
SHORT = {"mixed_full": "mixed", "global_prior": "prior", "global_sklearn": "global",
         "personal_full": "personal",
         "global_full": "unpen"}


def load_refs(runs):
    """-> (frame, where) for the first run that has reference models, else (None, None).

    Searches the primary run and then every `--also` run, because `main.py --refs` is enabled
    on ONE queue entry (the references depend on the split and the priors, not on the
    acquisition), so they usually live in a sibling folder rather than the one being plotted.

    A run whose splits were computed by the cluster array holds one
    ``reference_auroc_split<k>.csv`` per job and no combined file. Those are concatenated here
    and the result cached as ``reference_auroc.csv``, which is all
    ``reference_models.py --merge`` does -- no model is fitted, so doing it inline costs
    nothing and removes a manual step between the array draining and a figure.
    """
    import pandas as pd
    for run in runs:
        whole = os.path.join(run, "analysis", "reference_auroc.csv")
        if os.path.exists(whole):
            return pd.read_csv(whole), whole
        parts = sorted(glob.glob(os.path.join(run, "analysis",
                                              "reference_auroc_split*.csv")))
        if parts:
            R = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True) \
                  .drop_duplicates(["split", "target", "model"], keep="last")
            try:
                R.to_csv(whole, index=False)
            except OSError:                       # read-only tree: use it, just do not cache
                pass
            print(f"merged {len(parts)} per-split reference file(s) from "
                  f"{os.path.basename(run)}")
            return R, whole
    return None, None


def split_half_rho(G, n=200, seed=0):
    """Mean Spearman between per-person gaps scored on two disjoint halves of the splits.

    G is (target x split). With few splits this is noisy itself, so it is reported with the
    number of splits beside it rather than as a verdict.
    """
    from scipy.stats import spearmanr
    cols = list(G.columns)
    if len(cols) < 4:
        return np.nan
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = rng.permutation(cols)
        a, b = G[list(p[:len(p) // 2])].mean(axis=1), G[list(p[len(p) // 2:])].mean(axis=1)
        m = a.notna() & b.notna()
        if m.sum() > 2:
            out.append(spearmanr(a[m], b[m]).statistic)
    return float(np.mean(out)) if out else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="path to a run folder; default = newest indexed")
    ap.add_argument("--target", default="computer_work", help="used when --run is omitted")
    ap.add_argument("--contains", default=None, help="narrow the run lookup by folder substring")
    ap.add_argument("--also", nargs="*", default=None,
                    help="other run folders whose methods to overlay. Omit it and every "
                         "SIBLING of the primary run -- the other folders beside it that "
                         "agree on target, block, slopes, priors, seeds, rounds and batch -- "
                         "is picked up automatically, which is what the cluster fan-out "
                         "produces (one folder per method). Naming folders explicitly turns "
                         "that off and a mismatch is then an error rather than a skip. "
                         "Overlaying is only legitimate for runs that agree on those keys: "
                         "the seed sets are keyed on (seed, split, participant), so such "
                         "curves are paired, not independent. Methods already in the primary "
                         "run are skipped, so `random` is taken from there if present.")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="draw only these methods, by their real name in the per-round CSVs "
                         "(random is always kept -- every gap in the figure is measured "
                         "against it). Use it to drop an arm that is redundant with another, "
                         "e.g. --methods laplace laplace_people to show one cohort arm.")
    ap.add_argument("--no_siblings", action="store_true",
                    help="plot the primary run alone; do not look for sibling folders")
    ap.add_argument("--rows", type=int, default=4, help="panel rows; columns follow")
    ap.add_argument("--order_by", default=None,
                    help="method whose gap over random orders the panels (default: the first "
                         "non-random method present)")
    ap.add_argument("--refs", default=None,
                    help="reference CSV; default = <run>/analysis/reference_auroc.csv if present")
    ap.add_argument("--no_refs", action="store_true", help="draw the curves only")
    ap.add_argument("--show_unpenalised", action="store_true",
                    help="also draw global_full, the unpenalised pooled fit")
    ap.add_argument("--ref_models", nargs="*", default=None,
                    help="draw only these reference models, e.g. `--ref_models "
                         "global_sklearn`. Default: all of them that the CSV holds.")
    ap.add_argument("--keep_tails", action="store_true",
                    help="draw a target whose own pool runs out to the last round ANY of its "
                         "splits reached, instead of cutting every split back to the first "
                         "round one of them ran dry. The tail then averages fewer splits than "
                         "the head -- the count is printed per round -- so the band narrows "
                         "for a reason that has nothing to do with the acquisition.")
    ap.add_argument("--no_chance", action="store_true",
                    help="drop the dotted prevalence line that AUPRC otherwise carries")
    ap.add_argument("--order", default="gap", choices=["gap", "personalisation", "id"],
                    help="what orders the panels. gap = the acquisition's lead over random at "
                         "the last round (default). personalisation = mixed_full minus the "
                         "pooled global model, both on 100%% of the labels -- i.e. how much "
                         "this participant gains from having a model of their own at all, "
                         "which is a property of the PERSON and not of the acquisition. id = "
                         "numeric participant order.")
    ap.add_argument("--metric", default="auroc", choices=["auroc", "auprc"],
                    help="score column. Both are written by main.py for every round, so this "
                         "only changes what is read -- no refitting. AUPRC is the honest one "
                         "for a rare label: it moves with the positive rate, which is why the "
                         "chance line below is per target rather than a flat 0.5.")
    ap.add_argument("--share_y", default="auto", choices=["auto", "yes", "no"],
                    help="auto = share the y axis for AUROC, free it for AUPRC. A shared axis "
                         "lets you read panels against each other, but AUPRC's floor is the "
                         "target's own prevalence and that runs 0.00-0.64 here, so one shared "
                         "axis squashes most panels into a band.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    MET = {"auroc": "AUROC", "auprc": "AUPRC"}[a.metric]

    run = a.run or find_run(a.target, a.contains)
    D = load_run(run, "per_round")
    cfg = read_config(run) if os.path.exists(os.path.join(run, "config.json")) else {}

    # `first_split` is deliberately NOT here. Under the cluster fan-out each split is its own
    # job writing into the shared run folder, and every one of them rewrites config.json with
    # its own --first_split, so the stored value is whichever job happened to finish last. It
    # describes no property of the finished run. What actually has to match is the set of
    # splits present, which is checked below against the CSVs on disk rather than the config.
    SAME = ["target", "block", "n_slopes", "fe_p", "vcp_p", "n_seed", "n_seed_pos",
            "rounds", "batch", "seed", "_seeding"]
    # `_seeding` is derived, not stored: WHERE the initial labels come from has to be part of
    # the comparability gate, because `n_seed`/`n_seed_pos` keep their self-path values in a
    # cohort-seeded config and so cannot distinguish the two. The difference is not small -- on
    # computer_work the cohort seed starts at AUROC 0.68 against 0.64 for the self seed -- so
    # merging across it would compare arms that never saw the same initial model. Runs written
    # before --seed_source existed have no key at all and are self-seeded by construction.
    def seeding(c):
        src = c.get("seed_source", "self")
        return (src, c.get("n_seed_per_person") if src != "self" else None)

    cfg["_seeding"] = seeding(cfg)
    explicit = a.also is not None
    if explicit:
        others = list(a.also)
    elif a.no_siblings:
        others = []
    else:
        # Everything beside the primary run. The cluster fan-out writes one folder per method
        # (run_dir derives the name from the method list), so the siblings ARE the other
        # acquisitions -- there is nothing for the caller to type that this cannot work out.
        others = [d for d in sorted(glob.glob(os.path.join(os.path.dirname(run), "*")))
                  if os.path.isdir(d) and os.path.abspath(d) != os.path.abspath(run)
                  and os.path.exists(os.path.join(d, "config.json"))]
    merged, used = [], []
    # Folders that pass the comparability gate but are NOT drawn (usually because their split
    # set differs). Their reference fits are still valid for the splits they do cover -- the
    # references depend on the split and the priors, never on the acquisition -- so they are
    # kept as a place to look for reference_auroc*.csv, filtered by split below.
    ref_only = []
    for other in others:
        try:
            ocfg = read_config(other)
        except Exception as e:
            # One folder on disk has a malformed config.json (two concatenated objects), which
            # used to take the whole figure down and force --no_siblings. A sibling we found
            # ourselves is skipped; one the caller named is still an error worth stopping for.
            if explicit:
                raise SystemExit(f"{os.path.basename(other)}: config.json unreadable ({e})")
            print(f"  skipped {os.path.basename(other)}: config unreadable "
                  f"({type(e).__name__})")
            continue
        ocfg["_seeding"] = seeding(ocfg)
        bad = [k for k in SAME if cfg.get(k) != ocfg.get(k)]
        if bad:
            # A folder the caller named is a mistake worth stopping for; one we found
            # ourselves is just a different experiment sitting in the same directory.
            if explicit:
                raise SystemExit(f"{os.path.basename(other)} differs on {bad}; curves would "
                                 f"not be comparable")
            continue
        try:
            E = load_run(other, "per_round")
        except FileNotFoundError as e:
            # A config with no per-round CSVs: a run that was launched and cancelled, or whose
            # folder was created by --reserve. Nothing to merge, so skip it the same way.
            if explicit:
                raise SystemExit(f"{os.path.basename(other)}: {e}")
            print(f"  skipped {os.path.basename(other)}: no per-round output")
            continue
        shared = sorted(set(D.split) & set(E.split))
        if set(D.split) != set(E.split):
            # Every arm has to be averaged over the SAME splits or the gap is not paired, so a
            # short sibling can only be included by shortening everything else to match. That
            # is a real cost -- one method whose jobs half-failed would quietly decide how many
            # splits the whole figure rests on -- so it is done only when the caller named the
            # folder. A sibling we found ourselves is dropped instead, loudly.
            if not explicit:
                print(f"  skipped {os.path.basename(other)}: has splits "
                      f"{sorted(set(E.split))}, the run has {sorted(set(D.split))}; "
                      f"name it with --also to include it and shorten every arm to the "
                      f"{len(shared)} shared")
                ref_only.append(other)
                continue
            print(f"  note: {os.path.basename(other)} has splits {sorted(set(E.split))} vs "
                  f"{sorted(set(D.split))} here; restricting both to the {len(shared)} shared")
            if not shared:
                raise SystemExit(f"{os.path.basename(other)} shares no split with the run")
            D = D[D.split.isin(shared)]
            E = E[E.split.isin(shared)]
        E = E[~E.method.isin(set(D.method))]
        if len(E):
            import pandas as pd
            D = pd.concat([D, E], ignore_index=True)
            merged += sorted(set(E.method))
            used.append(other)
    if merged:
        print(f"overlaid {', '.join(merged)} from {len(used)} other run(s): "
              + ", ".join(os.path.basename(u) for u in used))
    methods = [m for m in D.method.unique()]
    if a.methods:
        keep_m = set(a.methods) | {"random"}       # the baseline every gap is measured against
        gone = sorted(set(methods) - keep_m)
        missing = sorted(keep_m - set(methods))
        if missing:
            raise SystemExit(f"--methods named {missing}, not in the run; have {sorted(methods)}")
        D = D[D.method.isin(keep_m)]
        methods = [m for m in methods if m in keep_m]
        if gone:
            print(f"  --methods: dropped {gone}")
    alt = a.order_by or next((m for m in methods if m != "random"), None)
    if alt is None or "random" not in methods:
        raise SystemExit(f"need an acquisition and random in the run; found {methods}")
    # A participant can exhaust their OWN pool before the budget is spent -- personal runs buy
    # min(batch, remaining) and then stop -- and they hit that wall at a different round in
    # each split. Leaving the tail in place would quietly change how many splits stand behind
    # a point partway along that target's curve (20 up to round 8, then 18), so every split of
    # such a target is cut back to the earliest round any split of theirs stopped at. Nothing
    # happens to a target whose splits all ran the full budget, which is every target of a
    # pooled run.
    cut = D.groupby(["target", "split"])["round"].max().groupby("target").min()
    R = D["round"].max()
    short = cut[cut < R]
    if len(short) and not a.keep_tails:
        print("  pool exhausted -- every split cut to the first round it ran out: "
              + ", ".join(f"{t} at round {int(r)}" for t, r in short.items()))
        D = D[D["round"] <= D.target.map(cut)]
    elif len(short):
        # --keep_tails: the curve runs as far as any split got, and the tail rests on fewer
        # splits than the head. Printed per target, because an SE band that narrows or widens
        # for that reason looks like a property of the acquisition and is not one.
        for t, r in short.items():
            n = (D[(D.target == t) & (D.method == "random")]
                 .groupby("round").split.nunique())
            tail = n[n < n.iloc[0]]
            print(f"  {t}: pool runs out from round {int(r) + 1}; splits behind each round "
                  f"{dict(zip(tail.index.tolist(), tail.values.tolist()))} of {int(n.iloc[0])}")
    nsplit = D.split.nunique()
    if len(short):
        n_per_round = D[D.method == "random"].groupby("round").target.nunique()
        thin = n_per_round[n_per_round < D.target.nunique()]
        if len(thin):
            print(f"  MEAN panel: rounds {list(thin.index)} average "
                  f"{sorted(set(thin.values))} targets, not {D.target.nunique()}")

    # gap per (target, split) at that target's last round -> ordering, and the stability check
    last = D.groupby("target")["round"].transform("max")
    W = (D[D["round"] == last].pivot_table(index=["target", "split"], columns="method",
                                           values=a.metric))
    gapts = (W[alt] - W["random"]).rename("gap").reset_index()
    G = gapts.pivot(index="target", columns="split", values="gap")
    gap = G.mean(axis=1).sort_values(ascending=False)
    rho = split_half_rho(G)

    REF = None
    # saddlebrown is deliberate: it collides with no acquisition colour in COL above, nor
    # with crimson/slategrey here or the mediumblue chance line.
    REFCOL = {"mixed_full": "crimson", "global_prior": "slategrey",
              "global_sklearn": "darkgreen",
              # the target's own rows, 100% of their labels -- the ceiling for a personal run,
              # where mixed_full is a bound a target-only model cannot reach
              "personal_full": "saddlebrown"}
    # Distinct dash patterns as well as colours: on a personal grid personal_full and
    # mixed_full sit close together on several participants, and two dashed lines of similar
    # length are hard to tell apart at this panel size even in different colours.
    REFLS = {"personal_full": "--", "mixed_full": (0, (6, 2, 1, 2)),
             "global_prior": (0, (4, 3)), "global_sklearn": (0, (1, 2)),
             "global_full": (0, (3, 1, 1, 1))}
    if a.show_unpenalised:
        REFCOL["global_full"] = "olive"
    if a.ref_models is not None:
        keep = set(a.ref_models)
        unknown = keep - set(REFCOL)
        if unknown:
            raise SystemExit(f"unknown reference model(s) {sorted(unknown)}; "
                             f"choose from {sorted(REFCOL)}")
        REFCOL = {k: v for k, v in REFCOL.items() if k in keep}
    if not a.no_refs:
        if a.refs:                                # an explicit file still wins
            import pandas as pd
            RAW, rpath = pd.read_csv(a.refs), a.refs
        else:
            # Never `others`: a sibling that failed the COMPARABILITY gate must not supply
            # ceilings, and load_refs returns the first run with any reference file at all, so
            # searching blindly let a neighbour fitted under different priors in silently.
            # `ref_only` is the narrower case -- comparable priors, different SPLIT set -- and
            # those fits are usable for the splits they cover, so they are searched after the
            # drawn runs and then filtered to the splits actually plotted, just below.
            RAW, rpath = load_refs([run] + used + ref_only)
        if RAW is not None and "split" in RAW.columns:
            # The curves are averaged over set(D.split); a reference averaged over any other
            # set of splits is a different number. Keep only the plotted splits, and refuse the
            # file outright if it does not cover all of them rather than drawing a ceiling that
            # rests on fewer splits than the curve beneath it.
            want = set(D.split)
            RAW = RAW[RAW.split.isin(want)]
            missing = sorted(want - set(RAW.split))
            if missing:
                print(f"references: {os.path.basename(rpath)} covers splits "
                      f"{sorted(set(RAW.split))} but the figure needs {sorted(want)}; "
                      f"not drawing them (missing {missing})")
                RAW = None
        if RAW is None:
            print("references: none found -- run the AL jobs with --refs, or pass --refs "
                  "<reference_auroc.csv>; drawing curves only")
        else:
            REF = RAW.pivot_table(index="target", columns="model", values=a.metric)
            # The references are split averages exactly like the curves, so they carry the same
            # kind of uncertainty and get the same +-1 SE band. Drawing them as bare lines made
            # a 100%-label ceiling look exact while the curve under it had a visible band.
            _n = RAW.pivot_table(index="target", columns="model", values=a.metric,
                                 aggfunc="count")
            _sd = RAW.pivot_table(index="target", columns="model", values=a.metric,
                                  aggfunc="std")
            REFSE = _sd / np.sqrt(_n)
            # MEAN panel: average over targets WITHIN a split first, then take the SE across
            # splits -- the same construction the curves use there, so the two are comparable.
            _ps = RAW.pivot_table(index="split", columns="model", values=a.metric)
            REFM, REFMSE = _ps.mean(), _ps.std(ddof=1) / np.sqrt(_ps.notna().sum())
            if "global_prior" not in REF and "global_full" in REF:     # older runs
                REFCOL["global_full"] = REFCOL.pop("global_prior", "slategrey")
            print(f"references: {', '.join(REF.columns)} over "
                  f"{RAW.split.nunique()} split(s) from {rpath}")

    # Chance for AUPRC is the positive rate of the test set, not 0.5, and it differs per
    # target and per split. Only the reference CSV carries n_test -- per_round_split*.csv
    # records n_pos_test alone -- so the line is available exactly when references are.
    PREV = PLO = PHI = PSD = None
    if a.metric == "auprc" and not a.no_chance:
        if REF is not None and {"n_pos_test", "n_test"} <= set(RAW.columns):
            pv = RAW.drop_duplicates(["split", "target"])
            gp = pv.groupby("target")
            # POOLED, not the mean of per-split ratios. The splits are day-based, so n_test
            # swings by up to 10x within one target, and weighting each split equally lets a
            # small day dominate: M33 reads 0.108 averaged but 0.055 pooled. The curve drawn
            # above is effectively a size-weighted average too, so pooled is the honest match.
            PREV = gp.n_pos_test.sum() / gp.n_test.sum()
            # The floor itself moves between splits, so a single line claims a precision the
            # data does not have. The band is +/- 1 SE across splits around the pooled value --
            # the same convention as every other "+/-" on the figure. (It was the min-max range
            # before, which is wider by roughly sqrt(n_splits) and is set by two extreme
            # splits rather than by the spread.)
            r = (pv.n_pos_test / pv.n_test).groupby(pv.target)
            _pse = r.std(ddof=1) / np.sqrt(r.count())
            PLO, PHI = PREV - _pse, PREV + _pse
            # SE across splits, printed beside pi in the title -- the same convention as the
            # reference values above each panel and the bands around the curves, so every "+/-"
            # on the figure means one thing. How far the floor actually MOVES between splits is
            # the shaded min-max band, which is wider by about sqrt(n_splits).
            PSD = _pse
        else:
            print("chance line: needs n_test from the reference CSV; drawing none")

    # Default: best acquisition gap first. `personalisation` instead ranks by how much the
    # 100%-label personalised model beats the 100%-label pooled one -- a fixed property of the
    # participant, computed without reference to any acquisition, so the two orderings answer
    # different questions and disagreeing is informative rather than a bug.
    if a.order == "personalisation":
        base = next((c for c in ("global_sklearn", "global_prior") if REF is not None
                     and c in REF.columns), None)
        if REF is None or "mixed_full" not in getattr(REF, "columns", []) or base is None:
            raise SystemExit("--order personalisation needs mixed_full and a pooled global "
                            "model in the reference CSV; found "
                            f"{list(getattr(REF, 'columns', []))}")
        pers = (REF["mixed_full"] - REF[base]).dropna().sort_values(ascending=False)
        seen = set(D.target)
        order = [t for t in pers.index if t in seen]
        order += [t for t in gap.index if t not in set(order)]      # any target without refs
        print(f"panel order: mixed_full - {base}, "
              f"{pers.iloc[0]:+.3f} ({pers.index[0]}) .. {pers.iloc[-1]:+.3f} "
              f"({pers.index[-1]})")
        order = order + ["MEAN"]
    elif a.order == "id":
        order = sorted(gap.index, key=lambda z: int(z[1:])) + ["MEAN"]
    else:
        order = list(gap.index) + ["MEAN"]
    nrow = a.rows
    ncol = int(np.ceil(len(order) / nrow))
    shy = {"yes": True, "no": False}.get(a.share_y, a.metric == "auroc")
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.15 * ncol, 2.4 * nrow),
                             sharex=True, sharey=shy, squeeze=False)
    for ax, tgt in zip(axes.ravel(), order):
        d = D if tgt == "MEAN" else D[D.target == tgt]
        for m in methods:
            g = d[d.method == m]
            # Average within a split first, so the SE is across splits in the MEAN panel too.
            #
            # Grouped by ROUND, not by `acquired`. They agree whenever every round buys a full
            # batch, but a participant whose own pool runs out buys a partial one -- the
            # personal runs do this -- and then that target alone has, say, acquired=23. In a
            # pivot over `acquired` that column holds ONE target, so the MEAN panel plotted
            # that single participant's value as if it were the mean of 43, which showed up as
            # spikes reaching M22's level. Rounds are shared by every target by construction.
            per = g.pivot_table(index="split", columns="round", values=a.metric)
            mu = per.mean()
            se = per.std(ddof=1) / np.sqrt(per.notna().sum())
            # x stays "labels acquired": the mean over the rows behind each round, which is
            # exactly round*batch except where a partial batch happened.
            xs = g.groupby("round").acquired.mean().reindex(mu.index)
            ax.plot(xs.values, mu.values, color=COL.get(m, "grey"), lw=1.4,
                    label=LABEL.get(m, m))
            ax.fill_between(xs.values, mu - se, mu + se, color=COL.get(m, "grey"),
                            alpha=0.18, lw=0)
        reftxt = []
        if REF is not None:
            for model, colour in REFCOL.items():
                if model not in REF:
                    continue
                if tgt == "MEAN":
                    v, se = REFM.get(model, np.nan), REFMSE.get(model, np.nan)
                else:
                    v, se = REF[model].get(tgt, np.nan), REFSE[model].get(tgt, np.nan)
                if np.isfinite(v):
                    ax.axhline(v, color=colour, lw=1.1, ls=REFLS.get(model, "--"),
                               label=(f"{model} (100% labels)" if tgt == order[0] else None))
                    # The SE goes in the title rather than a band: at this panel size the
                    # bands overlapped the curves they were meant to be compared against.
                    reftxt.append((colour, f"{v:.2f}"
                                   + (f"$\\pm${se:.2f}" if np.isfinite(se) and se > 0 else "")))
        pi = np.nan
        if PREV is not None:
            pi = PREV.mean() if tgt == "MEAN" else PREV.get(tgt, np.nan)
            if np.isfinite(pi):
                # Distinct from every other series: EU black, random orange, the 100%-label
                # references crimson/grey/gold. This one is not a model -- it is the AP a
                # constant scorer gets, i.e. the floor the panel should be read against.
                ax.axhline(pi, color="mediumblue", lw=1.1, ls=":",
                           label=(r"chance ($\pi$, pooled)" if tgt == order[0] else None))
            if PLO is not None:
                lo_, hi_ = ((PLO.mean(), PHI.mean()) if tgt == "MEAN"
                            else (PLO.get(tgt, np.nan), PHI.get(tgt, np.nan)))
                if np.isfinite(lo_) and np.isfinite(hi_) and hi_ > lo_:
                    ax.axhspan(lo_, hi_, color="mediumblue", alpha=0.12, lw=0,
                               label=(r"$\pi$ $\pm$1 SE over splits" if tgt == order[0]
                                      else None))
        # sd of pi across splits, for the title
        sd = np.nan
        if PSD is not None and np.isfinite(pi):
            sd = PSD.mean() if tgt == "MEAN" else PSD.get(tgt, np.nan)
        sdt = f"$\\pm${sd:.3f}" if np.isfinite(sd) else ""
        if tgt == "MEAN":
            pit = "" if not np.isfinite(pi) else f"  $\\pi$={pi:.3f}{sdt}"
            ax.set_title(f"MEAN (n={D.target.nunique()}){pit}", fontsize=8,
                         fontweight="bold", pad=14 if reftxt else None)
        else:
            pit = "" if not np.isfinite(pi) else f"  ($\\pi$={pi:.3f}{sdt})"
            # The signed gap is not printed: the panels are already ORDERED by it, best
            # first, so the position in the grid carries the same information.
            ax.set_title(f"{tgt}{pit}", fontsize=8, pad=14 if reftxt else None)
        # Each 100%-label reference's value is written in ITS OWN line colour, just under
        # the title, so the number identifies the line without a name beside it. Evenly
        # spaced across the panel width; matplotlib titles cannot colour part of a string,
        # so these are separate text artists in axes coordinates.
        for j, (colour, txt) in enumerate(reftxt):
            ax.text((j + 0.5) / len(reftxt), 1.015, txt, transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=7, color=colour)
        if a.metric != "auroc":
            # Same reason as in per_person_al_and_query: a free AUPRC axis produces
            # three-decimal labels that are wider than the space between panels.
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(order):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("labels acquired", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"test {MET}", fontsize=8)
    # The legend sits across the top of the figure, where a suptitle would otherwise go, so the
    # series are named once instead of inside one panel. Handles come from the first panel
    # because that is the only one given reference-line labels (see the axhline call above).
    rho_txt = "n/a (<4 splits)" if not np.isfinite(rho) else f"{rho:+.2f}"
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(labels), fontsize=9, frameon=False)   # one row, left to right
    # h_pad opens the gap between grid ROWS. The reference values are ax.text artists in
    # axes coordinates, and tight_layout cannot see those, so without extra room a row's
    # values print on top of the panel above it.
    fig.tight_layout(rect=[0, 0, 1, 0.955], h_pad=2.4)
    sfx = "" if a.metric == "auroc" else f"_{a.metric}"
    if a.order != "gap":
        sfx += f"_by{a.order[:4]}"
    out = figure_path(run, a.out or (f"per_person_grid_merged{sfx}.png" if merged
                                     else f"per_person_grid{sfx}.png"))
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"{nsplit} split(s), {D.target.nunique()} targets, rounds 0-{R}")
    print(f"mean {MET} gap {gap.mean():+.4f} "
          f"(SE {gap.std(ddof=1) / np.sqrt(len(gap)):.4f}), "
          f"{alt} better on {int((gap > 0).sum())}/{len(gap)}; split-half rho {rho_txt}")
    print("saved", out)


if __name__ == "__main__":
    main()
