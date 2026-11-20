"""The per-participant grid: one panel per target, the acquisition against random.

    python plots/per_person_grid.py                                  # newest computer_work run
    python plots/per_person_grid.py --run results/computer_work/<run folder> --rows 4
    python plots/per_person_grid.py --metric auprc --band sd

Panels are ordered by the acquisition-random gap at the last round; the split-half rank
correlation printed to stdout says whether that ordering means anything. Dashed reference lines
come from analysis/reference_auroc.csv (reference_models.py), fitted on 100% of the labels.
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from palette import METHOD_COLORS                                        # noqa: E402

COL = dict(METHOD_COLORS)

ORDER = ["laplace", "laplace_pool", "laplace_people", "laplace_var", "coreset",
         "typiclust", "probcover", "uncertainty", "eer", "random"]

# Display names
LABEL = {"laplace": "EU", "laplace_var": "EU (own variance)",
         "laplace_pool": "EU (cohort, per row)", "laplace_people": "EU (cohort, per person)"}
# short names for the reference models, for the panel titles
SHORT = {"mixed_full": "mixed", "global_prior": "prior", "global_sklearn": "global",
         "personal_full": "personal",
         "global_full": "unpen"}


def load_refs(runs):
    """-> (frame, where) for the first run that has reference models, else (None, None).

    Searches the primary run then every --also run, and merges per-split CSVs inline.
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
    """Mean Spearman between per-person gaps on two disjoint halves of the splits."""
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
    ap.add_argument("--band", default="both", choices=["se", "sd", "both"],
                    help="shaded band over splits: se, sd, or both (default, two figures)")
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

    # `first_split` is deliberately NOT here
    SAME = ["target", "block", "n_slopes", "fe_p", "vcp_p", "n_seed", "n_seed_pos",
            "rounds", "batch", "seed", "_seeding"]
    # `_seeding` is derived, not stored: where the initial labels come from must match
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
        # Everything beside the primary run
        others = [d for d in sorted(glob.glob(os.path.join(os.path.dirname(run), "*")))
                  if os.path.isdir(d) and os.path.abspath(d) != os.path.abspath(run)
                  and os.path.exists(os.path.join(d, "config.json"))]
    merged, used = [], []
    # Folders that pass the comparability gate but are NOT drawn (usually
    ref_only = []
    for other in others:
        try:
            ocfg = read_config(other)
        except Exception as e:
            # One folder on disk has a malformed config.json (two concatenated objects)
            if explicit:
                raise SystemExit(f"{os.path.basename(other)}: config.json unreadable ({e})")
            print(f"  skipped {os.path.basename(other)}: config unreadable "
                  f"({type(e).__name__})")
            continue
        ocfg["_seeding"] = seeding(ocfg)
        bad = [k for k in SAME if cfg.get(k) != ocfg.get(k)]
        if bad:
            # A folder the caller named is a mistake worth stopping for; one we found
            if explicit:
                raise SystemExit(f"{os.path.basename(other)} differs on {bad}; curves would "
                                 f"not be comparable")
            continue
        try:
            E = load_run(other, "per_round")
        except FileNotFoundError as e:
            # a config with no per-round CSVs: launched and cancelled, or still running
            if explicit:
                raise SystemExit(f"{os.path.basename(other)}: {e}")
            print(f"  skipped {os.path.basename(other)}: no per-round output")
            continue
        shared = sorted(set(D.split) & set(E.split))
        if set(D.split) != set(E.split):
            # Every arm has to be averaged over the SAME splits or the gap is not paired
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
    # fixed order, so the legend reads the same whatever order the sibling runs merged in
    methods = sorted(D.method.unique(), key=lambda m: ORDER.index(m) if m in ORDER
                     else len(ORDER))
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
    # A participant can exhaust their OWN pool before the budget is spent
    cut = D.groupby(["target", "split"])["round"].max().groupby("target").min()
    R = D["round"].max()
    short = cut[cut < R]
    if len(short) and not a.keep_tails:
        print("  pool exhausted -- every split cut to the first round it ran out: "
              + ", ".join(f"{t} at round {int(r)}" for t, r in short.items()))
        D = D[D["round"] <= D.target.map(cut)]
    elif len(short):
        # --keep_tails: the curve runs as far as any split got, on fewer splits at the tail
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
    # No reference shares a colour with an acquisition: saddlebrown is EU's.
    REFCOL = {"mixed_full": "crimson", "global_prior": "navy",
              "global_sklearn": "black",
              # the target's own rows, 100% of their labels: the personal-run ceiling
              "personal_full": "slategrey"}
    # Drawn only when named in --ref_models; the pooled prior crowds the panels.
    OFF_BY_DEFAULT = {"global_prior"}
    # Distinct dash patterns as well as colours
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
    else:
        REFCOL = {k: v for k, v in REFCOL.items() if k not in OFF_BY_DEFAULT}
    if not a.no_refs:
        if a.refs:                                # an explicit file still wins
            import pandas as pd
            RAW, rpath = pd.read_csv(a.refs), a.refs
        else:
            # never `others`: a sibling that failed the comparability gate cannot supply refs
            RAW, rpath = load_refs([run] + used + ref_only)
        if RAW is not None and "split" in RAW.columns:
            # The curves are averaged over set(D.split); a reference averaged over any other
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
            # The references are split averages exactly like the curves
            _n = RAW.pivot_table(index="target", columns="model", values=a.metric,
                                 aggfunc="count")
            _sd = RAW.pivot_table(index="target", columns="model", values=a.metric,
                                  aggfunc="std")
            REFSE = _sd / np.sqrt(_n)
            # MEAN panel: average over targets within a split, then SE across splits
            _ps = RAW.pivot_table(index="split", columns="model", values=a.metric)
            REFM, REFMSE = _ps.mean(), _ps.std(ddof=1) / np.sqrt(_ps.notna().sum())
            if "global_prior" not in REF and "global_full" in REF:     # older runs
                REFCOL["global_full"] = REFCOL.pop("global_prior", "navy")
            drawn = [m for m in REF.columns if m in REFCOL]
            print(f"references drawn: {', '.join(drawn)}"
                  + (f"  (in the CSV but off: "
                     f"{', '.join(m for m in REF.columns if m not in REFCOL)})"
                     if set(REF.columns) - set(REFCOL) else "")
                  + f" over {RAW.split.nunique()} split(s) from {rpath}")

    # Chance for AUPRC is the positive rate of the test set, not 0.5
    PREV = PLO = PHI = PSD = None
    if a.metric == "auprc" and not a.no_chance:
        if REF is not None and {"n_pos_test", "n_test"} <= set(RAW.columns):
            pv = RAW.drop_duplicates(["split", "target"])
            gp = pv.groupby("target")
            # POOLED, not the mean of per-split ratios
            PREV = gp.n_pos_test.sum() / gp.n_test.sum()
            # The floor itself moves between splits
            r = (pv.n_pos_test / pv.n_test).groupby(pv.target)
            _pse = r.std(ddof=1) / np.sqrt(r.count())
            PLO, PHI = PREV - _pse, PREV + _pse
            # SE across splits, printed beside pi in the title
            PSD = _pse
        else:
            print("chance line: needs n_test from the reference CSV; drawing none")

    # default: best acquisition gap first; `personalisation` ranks by headroom instead
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
    bands, saved = (["se", "sd"] if a.band == "both" else [a.band]), []
    nrow = a.rows
    ncol = int(np.ceil(len(order) / nrow))
    shy = {"yes": True, "no": False}.get(a.share_y, a.metric == "auroc")
    for band in bands:
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.15 * ncol, 2.4 * nrow),
                                 sharex=True, sharey=shy, squeeze=False)
        for ax, tgt in zip(axes.ravel(), order):
            d = D if tgt == "MEAN" else D[D.target == tgt]
            for m in methods:
                g = d[d.method == m]
                # Average within a split first, so the SE is across splits in the MEAN panel too
                per = g.pivot_table(index="split", columns="round", values=a.metric)
                mu = per.mean()
                sd = per.std(ddof=1)
                band_v = sd if band == "sd" else sd / np.sqrt(per.notna().sum())
                # x stays "labels acquired"
                xs = g.groupby("round").acquired.mean().reindex(mu.index)
                ax.plot(xs.values, mu.values, color=COL.get(m, "grey"), lw=1.4,
                        label=LABEL.get(m, m))
                ax.fill_between(xs.values, mu - band_v, mu + band_v, color=COL.get(m, "grey"),
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
                        # SE in the title, not a band: a band is unreadable at this panel size
                        reftxt.append((colour, f"{v:.2f}"
                                       + (f"$\\pm${se:.2f}" if np.isfinite(se) and se > 0 else "")))
            pi = np.nan
            if PREV is not None:
                pi = PREV.mean() if tgt == "MEAN" else PREV.get(tgt, np.nan)
                if np.isfinite(pi):
                    # distinct from every other series: EU black, random orange, refs dashed
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
                # the signed gap is not printed: the panels are already ordered by it
                ax.set_title(f"{tgt}{pit}", fontsize=8, pad=14 if reftxt else None)
            # Each 100%-label reference's value is written in ITS OWN line colour
            for j, (colour, txt) in enumerate(reftxt):
                ax.text((j + 0.5) / len(reftxt), 1.015, txt, transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=7, color=colour)
            if a.metric != "auroc":
                # a free AUPRC axis produces three-decimal ticks, so it is left shared
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.tick_params(labelsize=7)
        for ax in axes.ravel()[len(order):]:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("labels acquired", fontsize=8)
        for ax in axes[:, 0]:
            ax.set_ylabel(f"test {MET}", fontsize=8)
        # The legend sits across the top of the figure, where a suptitle would otherwise go
        rho_txt = "n/a (<4 splits)" if not np.isfinite(rho) else f"{rho:+.2f}"
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                   ncol=len(labels), fontsize=9, frameon=False)   # one row, left to right
        # h_pad opens the gap between grid ROWS. The reference values are ax.text artists in axes
        fig.tight_layout(rect=[0, 0, 1, 0.955], h_pad=2.4)
        sfx = "" if a.metric == "auroc" else f"_{a.metric}"
        sfx += "_sd" if band == "sd" else ""
        if a.order != "gap":
            sfx += f"_by{a.order[:4]}"
        out = figure_path(run, a.out or (f"per_person_grid_merged{sfx}.png" if merged
                                         else f"per_person_grid{sfx}.png"))
        fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(out)
    print(f"{nsplit} split(s), {D.target.nunique()} targets, rounds 0-{R}")
    print(f"mean {MET} gap {gap.mean():+.4f} "
          f"(SE {gap.std(ddof=1) / np.sqrt(len(gap)):.4f}), "
          f"{alt} better on {int((gap > 0).sum())}/{len(gap)}; split-half rho {rho_txt}")
    for f in saved:
        print("saved", f)


if __name__ == "__main__":
    main()
