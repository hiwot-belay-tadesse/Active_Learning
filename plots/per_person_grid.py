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
import numpy as np                                                       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import figure_path, find_run, load_run, read_config           # noqa: E402

COL = {"typiclust": "purple", "coreset": "seagreen", "random": "darkorange",
       "probcover": "dodgerblue", "uncertainty": "steelblue", "eer": "magenta",
       "laplace": "black", "laplace_var": "gold"}

# Display names. The method STRING stays as it is written in the per-round CSVs and the run
# folder name -- renaming it there would orphan every finished run -- so the relabelling is
# cosmetic and lives only here. `--order_by` and `--methods` still take the real name.
LABEL = {"laplace": "EU", "laplace_var": "EU (own variance)"}


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
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    run = a.run or find_run(a.target, a.contains)
    D = load_run(run, "per_round")
    cfg = read_config(run) if os.path.exists(os.path.join(run, "config.json")) else {}

    # `first_split` is deliberately NOT here. Under the cluster fan-out each split is its own
    # job writing into the shared run folder, and every one of them rewrites config.json with
    # its own --first_split, so the stored value is whichever job happened to finish last. It
    # describes no property of the finished run. What actually has to match is the set of
    # splits present, which is checked below against the CSVs on disk rather than the config.
    SAME = ["target", "block", "n_slopes", "fe_p", "vcp_p", "n_seed", "n_seed_pos",
            "rounds", "batch"]
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
    for other in others:
        ocfg = read_config(other)
        bad = [k for k in SAME if cfg.get(k) != ocfg.get(k)]
        if bad:
            # A folder the caller named is a mistake worth stopping for; one we found
            # ourselves is just a different experiment sitting in the same directory.
            if explicit:
                raise SystemExit(f"{os.path.basename(other)} differs on {bad}; curves would "
                                 f"not be comparable")
            continue
        E = load_run(other, "per_round")
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
    alt = a.order_by or next((m for m in methods if m != "random"), None)
    if alt is None or "random" not in methods:
        raise SystemExit(f"need an acquisition and random in the run; found {methods}")
    R = D["round"].max()
    nsplit = D.split.nunique()

    # gap per (target, split) at the last round -> ordering, and the stability check
    W = (D[D["round"] == R].pivot_table(index=["target", "split"], columns="method",
                                        values="auroc"))
    gapts = (W[alt] - W["random"]).rename("gap").reset_index()
    G = gapts.pivot(index="target", columns="split", values="gap")
    gap = G.mean(axis=1).sort_values(ascending=False)
    rho = split_half_rho(G)

    REF = None
    REFCOL = {"mixed_full": "crimson", "global_prior": "slategrey",
              "global_sklearn": "gold"}
    if a.show_unpenalised:
        REFCOL["global_full"] = "olive"
    if not a.no_refs:
        if a.refs:                                # an explicit file still wins
            import pandas as pd
            RAW, rpath = pd.read_csv(a.refs), a.refs
        else:
            RAW, rpath = load_refs([run] + others)
        if RAW is None:
            print("references: none found -- run the AL jobs with --refs, or pass --refs "
                  "<reference_auroc.csv>; drawing curves only")
        else:
            REF = RAW.pivot_table(index="target", columns="model", values="auroc")
            if "global_prior" not in REF and "global_full" in REF:     # older runs
                REFCOL["global_full"] = REFCOL.pop("global_prior", "slategrey")
            print(f"references: {', '.join(REF.columns)} over "
                  f"{RAW.split.nunique()} split(s) from {rpath}")

    order = list(gap.index) + ["MEAN"]
    nrow = a.rows
    ncol = int(np.ceil(len(order) / nrow))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.15 * ncol, 2.4 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    for ax, tgt in zip(axes.ravel(), order):
        d = D if tgt == "MEAN" else D[D.target == tgt]
        for m in methods:
            g = d[d.method == m]
            # average within a split first, so the SE is across splits in the MEAN panel too
            per = g.pivot_table(index="split", columns="acquired", values="auroc")
            mu = per.mean()
            se = per.std(ddof=1) / np.sqrt(per.notna().sum())
            ax.plot(mu.index, mu.values, color=COL.get(m, "grey"), lw=1.4, label=LABEL.get(m, m))
            ax.fill_between(mu.index, mu - se, mu + se, color=COL.get(m, "grey"),
                            alpha=0.18, lw=0)
        if REF is not None:
            for model, colour in REFCOL.items():
                if model not in REF:
                    continue
                v = REF[model].mean() if tgt == "MEAN" else REF[model].get(tgt, np.nan)
                if np.isfinite(v):
                    ax.axhline(v, color=colour, lw=1.1, ls="--",
                               label=(f"{model} (100% labels)" if tgt == order[0] else None))
        if tgt == "MEAN":
            ax.set_title(f"MEAN (n={D.target.nunique()})  {gap.mean():+.3f}",
                         fontsize=8, fontweight="bold")
        else:
            npos = D[(D.target == tgt) & (D["round"] == R)].n_pos_test.mean()
            ax.set_title(f"{tgt}  {gap[tgt]:+.3f}  (pos~{int(npos)})", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(order):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("labels acquired", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("test AUROC", fontsize=8)
    # The legend sits across the top of the figure, where a suptitle would otherwise go, so the
    # series are named once instead of inside one panel. Handles come from the first panel
    # because that is the only one given reference-line labels (see the axhline call above).
    rho_txt = "n/a (<4 splits)" if not np.isfinite(rho) else f"{rho:+.2f}"
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=len(labels), fontsize=9, frameon=False)   # one row, left to right
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out = figure_path(run, a.out or ("per_person_grid_merged.png" if merged
                                     else "per_person_grid.png"))
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"{nsplit} split(s), {D.target.nunique()} targets, rounds 0-{R}")
    print(f"mean gap {gap.mean():+.4f} (SE {gap.std(ddof=1) / np.sqrt(len(gap)):.4f}), "
          f"{alt} better on {int((gap > 0).sum())}/{len(gap)}; split-half rho {rho_txt}")
    print("saved", out)


if __name__ == "__main__":
    main()
