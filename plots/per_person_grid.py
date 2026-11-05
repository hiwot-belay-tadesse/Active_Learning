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
The title line reports the split-half rank correlation of the per-person gap, which is the
quick check on whether that ordering means anything: near zero says the ranking is luck.

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
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import figure_path, find_run, load_run, read_config           # noqa: E402

COL = {"typiclust": "purple", "coreset": "seagreen", "random": "darkorange",
       "probcover": "teal", "uncertainty": "steelblue"}


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
    ap.add_argument("--rows", type=int, default=4, help="panel rows; columns follow")
    ap.add_argument("--refs", default=None,
                    help="reference CSV; default = <run>/analysis/reference_auroc.csv if present")
    ap.add_argument("--no_refs", action="store_true", help="draw the curves only")
    ap.add_argument("--show_unpenalised", action="store_true",
                    help="also draw global_full, the unpenalised pooled fit")
    ap.add_argument("--out", default="per_person_grid.png")
    a = ap.parse_args()

    run = a.run or find_run(a.target, a.contains)
    D = load_run(run, "per_round")
    cfg = read_config(run) if os.path.exists(os.path.join(run, "config.json")) else {}
    methods = [m for m in D.method.unique()]
    alt = next((m for m in methods if m != "random"), None)
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
    REFCOL = {"mixed_full": "crimson", "global_prior": "slategrey"}
    if a.show_unpenalised:
        REFCOL["global_full"] = "darkseagreen"
    rpath = a.refs or os.path.join(run, "analysis", "reference_auroc.csv")
    if not a.no_refs and os.path.exists(rpath):
        import pandas as pd
        REF = (pd.read_csv(rpath).pivot_table(index="target", columns="model", values="auroc"))
        if "global_prior" not in REF and "global_full" in REF:     # older runs
            REFCOL["global_full"] = REFCOL.pop("global_prior", "slategrey")
        print(f"references: {', '.join(REF.columns)} from {os.path.relpath(rpath, run)}")

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
            ax.plot(mu.index, mu.values, color=COL.get(m, "grey"), lw=1.4, label=m)
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
    axes[0, 0].legend(fontsize=7, loc="lower right")

    rho_txt = "n/a (<4 splits)" if not np.isfinite(rho) else f"{rho:+.2f}"
    fig.suptitle(f"{cfg.get('target', a.target)} — {alt} vs random, W={cfg.get('block', '?')}, "
                 f"mean over {nsplit} split(s) ±1 SE; panel title = gap at {R} rounds; "
                 f"split-half rank ρ of the gap = {rho_txt}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = figure_path(run, a.out)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"{nsplit} split(s), {D.target.nunique()} targets, rounds 0-{R}")
    print(f"mean gap {gap.mean():+.4f} (SE {gap.std(ddof=1) / np.sqrt(len(gap)):.4f}), "
          f"{alt} better on {int((gap > 0).sum())}/{len(gap)}; split-half rho {rho_txt}")
    print("saved", out)


if __name__ == "__main__":
    main()
