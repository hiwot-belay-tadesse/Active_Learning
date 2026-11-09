"""Per-participant bar chart: the mixed-effects model against a pooled logistic regression.

One bar per participant for the AUROC difference at 100% of the labels, sorted by that
difference. Both models are read from the run's ``analysis/reference_auroc.csv`` (written by
reference_models.py), so nothing is refitted here and the numbers match the dashed reference
lines on the per-person grid.

    python plots/personalisation_gain_bar.py --contains r10x3
    python plots/personalisation_gain_bar.py --contains r10x3 --global_model global_prior
    python plots/personalisation_gain_bar.py --contains r10x3 --direction global_minus_mixed

Sign convention: the default is MIXED - GLOBAL, i.e. what personalisation buys, so a positive
bar means the random effects helped that participant. ``--direction`` flips it.

``--global_model`` picks which pooled fit to compare against, and the choice matters:

  global_sklearn  plain scikit-learn LogisticRegression(penalty=None) -- the default here,
                  because it is the "global logistic regression" in the plainest sense.
  global_full     the same unpenalised model fitted by statsmodels. Agrees with the above to
                  ~0.001 on COMPUTER_WORK, but its coefficient norm is larger: lbfgs stops on
                  tolerance where IRLS does not, which acts as weak shrinkage.
  global_prior    ridge-penalised with the SAME Gaussian prior the mixed model puts on its
                  fixed effects. This is the fairer contrast if the question is "do the random
                  effects help", because it is the only one that does not also differ in
                  regularisation -- an unpenalised comparison confounds the two.

The bars are a paired difference (same split, same test rows, same features), so the spread
across participants is real heterogeneity in who personalisation helps, not sampling noise
between two independent fits.

Whiskers show how much each participant's gain moved ACROSS SPLITS (``--whisker``). Read them
as a scale, not a test: with a handful of splits the SD they are built from is itself poorly
estimated, the splits share most of their training data so the replicates are not independent,
and no correction is made for comparing 43 participants at once. A bar whose whisker crosses
zero is not established, but a bar whose whisker clears zero is not thereby significant either.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import figure_path, find_run, read_config                     # noqa: E402

POS, NEG = "#2a7f62", "#b5503f"     # personalisation helps / hurts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None)
    ap.add_argument("--target", default="computer_work")
    ap.add_argument("--contains", default=None)
    ap.add_argument("--refs", default=None, help="reference_auroc.csv from another run")
    ap.add_argument("--global_model", default="global_sklearn",
                    choices=["global_sklearn", "global_full", "global_prior"])
    ap.add_argument("--mixed_model", default="mixed_full")
    ap.add_argument("--direction", default="mixed_minus_global",
                    choices=["mixed_minus_global", "global_minus_mixed"])
    ap.add_argument("--metric", default="auroc", choices=["auroc", "auprc"])
    ap.add_argument("--whisker", default="se", choices=["se", "sd", "range"],
                    help="se: +/-1 standard error over splits (default); sd: +/-1 SD; "
                         "range: min to max. With only a few splits all three are themselves "
                         "noisy -- they indicate scale, they do not test anything.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    run = a.run or find_run(a.target, a.contains)
    cfg = read_config(run)
    rpath = a.refs or os.path.join(run, "analysis", "reference_auroc.csv")
    if not os.path.exists(rpath):
        raise SystemExit(f"no reference table at {rpath}; run reference_models.py first")

    R = pd.read_csv(rpath)
    have = set(R.model)
    for m in (a.mixed_model, a.global_model):
        if m not in have:
            raise SystemExit(f"{m!r} not in {rpath} (has {sorted(have)}); "
                             f"reference_models.py writes global_sklearn only on a recent run")

    # Difference per (target, split) FIRST, then summarise across splits. Differencing inside a
    # split keeps the comparison paired -- both models saw the same training rows and are scored
    # on the same test rows -- so the spread across splits is split-to-split variation in that
    # participant's gain, not variation in the two models' absolute AUROC.
    P = R.pivot_table(index=["target", "split"], columns="model", values=a.metric)
    per = (P[a.mixed_model] - P[a.global_model]).dropna()
    if a.direction == "global_minus_mixed":
        per = -per
    g = per.groupby("target")
    d, sd, k = g.mean(), g.std(ddof=1), g.count()
    if a.whisker == "se":
        err, err_txt = sd / np.sqrt(k), "+/-1 SE over splits"
    elif a.whisker == "sd":
        err, err_txt = sd, "+/-1 SD over splits"
    else:
        err = pd.concat([d - g.min(), g.max() - d], axis=1).T.to_numpy()
        err_txt = "min-max over splits"
    d = d.sort_values(ascending=False)
    err = (err.reindex(d.index) if a.whisker != "range"
           else err[:, [list(g.mean().index).index(t) for t in d.index]])
    nsplit = int(k.max())

    lo = a.mixed_model if a.direction == "mixed_minus_global" else a.global_model
    hi = a.global_model if a.direction == "mixed_minus_global" else a.mixed_model
    label = f"{lo} - {hi}  ({a.metric})"

    fig, ax = plt.subplots(figsize=(max(7, 0.3 * len(d) + 2), 4.2))
    ax.bar(range(len(d)), d.values, color=[POS if v > 0 else NEG for v in d.values],
           width=0.78,
           yerr=err if a.whisker == "range" else err.values,
           error_kw=dict(ecolor="0.25", lw=0.9, capsize=2.0, capthick=0.9))
    ax.axhline(0, color="black", lw=0.9)
    ax.axhline(d.mean(), color="black", ls="--", lw=1.0,
               label=f"mean {d.mean():+.4f}  ({int((d > 0).sum())}/{len(d)} positive)")
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d.index, rotation=90, fontsize=7)
    ax.set_xlim(-0.8, len(d) - 0.2)
    ax.set_ylabel(label, fontsize=9)
    ax.set_title(f"whiskers: {err_txt} ({nsplit} splits)", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    name = a.out or f"gain_{a.mixed_model}_vs_{a.global_model}_{a.metric}.png"
    out = figure_path(run, name)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")

    print(f"{os.path.relpath(run)}\n  {len(d)} participants, {nsplit} split(s), "
          f"W={cfg.get('block', '?')}, 100% of labels")
    print(f"  {label}: mean {d.mean():+.4f}, median {d.median():+.4f}, "
          f"positive for {int((d > 0).sum())}/{len(d)}")
    print(f"  best  {d.index[0]} {d.iloc[0]:+.4f}   worst {d.index[-1]} {d.iloc[-1]:+.4f}")
    se = (sd / np.sqrt(k)).reindex(d.index)
    print(f"  whiskers: {err_txt}; median per-person SE {se.median():.4f}, "
          f"{int((d.abs() > se).sum())}/{len(d)} participants with |gain| > their own SE")
    print("saved", out)


if __name__ == "__main__":
    main()
