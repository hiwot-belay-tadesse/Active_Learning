"""Grid of per-target active-learning curves, one panel per participant.

Pools the 20-split runs: `random` comes from the laplace-random folder, the geometric
baselines from the coreset-probcover-typiclust folder. Both have the same 20 splits, the
same 43 targets and byte-identical seed sets (cluster/run.py relies on that), so curves
from the two folders are paired and can share an axis.

  --methods  which arms to draw (default: everything but laplace/EU)
  --metric   auroc | auprc
"""
import argparse, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "results", "computer_work")
FOLDERS = ["computer_work_w15_laplace-random_r10x3_s5_20split",
           "computer_work_w15_coreset-probcover-typiclust_r10x3_s5_20split",
           "computer_work_w15_laplace_var_r10x3_s5_20split"]
COLORS = {"random": "#ff7f0e", "coreset": "#2ca02c", "probcover": "#8c564b",
          "typiclust": "#b23fb2", "laplace": "#1f77b4", "laplace_var": "#17becf"}
# 100%-label reference models, from analysis/reference_auroc.csv (cluster/run.py `refs=1`).
# mixed_full is the SAME mixed-effects model fitted on every label -- the ceiling this budget
# is chasing. global_sklearn is a plain LogisticRegression(penalty=None) on the same pooled
# rows: the no-personalisation, no-prior reference. global_prior (the pooled statsmodels fit
# under the same fe_p prior) tracks it to within 0.003 and is not drawn.
REFS = {"mixed_full": ("#d62728", "-."), "global_sklearn": ("#b8860b", "--")}


def load_refs():
    p = os.path.join(ROOT, FOLDERS[0], "analysis", "reference_auroc.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def load():
    fr = []
    for f in FOLDERS:
        for p in glob.glob(os.path.join(ROOT, f, "per_round_split*.csv")):
            fr.append(pd.read_csv(p))
    d = pd.concat(fr, ignore_index=True)
    return d[~d.fit_failed]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+",
                    default=["random", "typiclust", "coreset", "probcover"])
    ap.add_argument("--metric", default="auroc", choices=["auroc", "auprc"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        stem = "auroc_grid_with_EU" if a.metric == "auroc" else "pr-auc_grid_with_EU"
        a.out = os.path.join(ROOT, FOLDERS[0], "figures", f"{stem}.png")

    d = load()
    d = d[d.method.isin(a.methods)]
    m = a.metric
    CHANCE_C = "#0b63d6" if a.metric == "auprc" else "k"
    ref = load_refs()
    if ref is not None:
        pi_src = ref.copy()
        ref = ref[ref.model.isin(REFS)].dropna(subset=[m])
        # per target: mean and s.e. over the 20 splits
        rt_g = ref.groupby(["target", "model"])[m]
        ref_t, ref_t_se = rt_g.mean(), rt_g.std() / np.sqrt(rt_g.count())
        # pooled: macro-average over targets within a split, then mean/s.e. over splits
        rp_g = ref.groupby(["model", "split"])[m].mean().groupby("model")
        ref_p, ref_p_se = rp_g.mean(), rp_g.std() / np.sqrt(rp_g.count())

    # per (target, method, round): mean over the 20 splits, and its standard error
    g = (d.groupby(["target", "method", "n_labeled"])[m]
           .agg(mean="mean", sd="std", n="count").reset_index())
    g["se"] = g["sd"] / np.sqrt(g["n"])

    # pooled over targets as well, for the lead panel
    pool = (d.groupby(["method", "n_labeled", "split"])[m].mean().reset_index()
              .groupby(["method", "n_labeled"])[m]
              .agg(mean="mean", sd="std", n="count").reset_index())
    pool["se"] = pool["sd"] / np.sqrt(pool["n"])

    # order panels by how much personalisation is worth for that participant at 100% labels:
    # mixed_full - global_sklearn, descending. Where the references are missing, fall back to
    # mean final score so every target still gets a place.
    if ref is not None:
        rw = ref.pivot_table(index="target", columns="model", values=m)
        gain = (rw["mixed_full"] - rw["global_sklearn"]).dropna()
    else:
        gain = pd.Series(dtype=float)
    fin = g[g.n_labeled == g.n_labeled.max()].groupby("target")["mean"].mean()
    key = gain.reindex(fin.index)
    key = key.fillna(key.min() - 1) if len(gain) else fin
    targets = list(key.sort_values(ascending=False).index)

    ncol, n = 6, len(targets) + 1
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.1 * nrow),
                             sharex=True, sharey=True)
    axes = axes.ravel()

    def draw(ax, df, title, bold=False, refs=None, chance=None, chance_se=np.nan,
             pi_txt=None):
        if refs is not None:
            for mod, (v, se) in refs.items():
                c, ls = REFS[mod]
                ax.axhline(v, color=c, ls=ls, lw=1.1, zorder=0,
                           label=f"{mod} (100% labels)" if bold else None)
                if np.isfinite(se):
                    ax.axhspan(v - se, v + se, color=c, alpha=0.15, lw=0, zorder=0)
        for meth in a.methods:
            s = df[df.method == meth].sort_values("n_labeled")
            if not len(s):
                continue
            c = COLORS.get(meth, "k")
            ax.plot(s.n_labeled, s["mean"], color=c, lw=1.8 if bold else 1.2,
                    marker="o" if bold else None, ms=3, label=meth)
            ax.fill_between(s.n_labeled, s["mean"] - s.se, s["mean"] + s.se,
                            color=c, alpha=0.18, lw=0)
        if chance is not None and np.isfinite(chance):
            ax.axhline(chance, color=CHANCE_C, lw=1.0, ls=":",
                       label="π (test positive rate)" if bold and m == "auprc" else None)
        ax.set_title(title, fontsize=9, fontweight="bold" if bold else "normal")
        if pi_txt:
            ax.text(1.0, 1.015, pi_txt, transform=ax.transAxes, ha="right", va="bottom",
                    color=CHANCE_C, fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.4)

    rp = ({k: (ref_p[k], ref_p_se[k]) for k in ref_p.index} if ref is not None else None)
    # AUROC's chance level is 0.5 everywhere; AUPRC's is the positive rate of that
    # participant's test set, which differs per panel, so it is computed per target.
    if ref is not None and {"n_test", "n_pos_test"} <= set(pi_src.columns):
        # one pi per (target, split); the band is its s.e. across the 20 splits
        ps = (pi_src.drop_duplicates(["target", "split"])
                    .assign(pi=lambda t: t.n_pos_test / t.n_test))
        bg = ps.groupby("target").pi
        base, base_se = bg.mean(), bg.std() / np.sqrt(bg.count())
        pooled = ps.groupby("split").pi.mean()
        base_m, base_m_se = pooled.mean(), pooled.std() / np.sqrt(len(pooled))
    else:
        base = base_se = pd.Series(np.nan, index=sorted(d.target.unique()))
        base_m = base_m_se = np.nan
    # short enough that the blue pi annotation at the right does not run into it
    mean_ttl = f"MEAN of {d.target.nunique()}"
    mean_pi = (f"π={base_m:.2f}±{base_m_se:.2f}" if m == "auprc" else None)
    draw(axes[0], pool, mean_ttl, bold=True, refs=rp, pi_txt=mean_pi,
         chance=0.5 if m == "auroc" else float(base_m),
         chance_se=np.nan if m == "auroc" else float(base_m_se))
    handles, labels = axes[0].get_legend_handles_labels()
    for ax, t in zip(axes[1:], targets):
        rt = ({mod: (ref_t[(t, mod)], ref_t_se[(t, mod)])
               for mod in REFS if (t, mod) in ref_t.index} if ref is not None else None)
        # pi, the test set's positive rate: AUPRC's chance level, and the thing that makes
        # AUPRC values incomparable between panels. Annotated in the chance line's own colour
        # so the number and the line it marks read as one object.
        pi_txt = None
        if m == "auprc" and t in base.index and np.isfinite(base[t]):
            se_t = base_se.get(t, np.nan)
            pi_txt = (f"π={base[t]:.2f}±{se_t:.2f}" if np.isfinite(se_t)
                      else f"π={base[t]:.2f}")
        c = 0.5 if m == "auroc" else base.get(t, np.nan)
        draw(ax, g[g.target == t], t, refs=rt,
             chance=None if pd.isna(c) else float(c),
             chance_se=np.nan if m == "auroc" else float(base_se.get(t, np.nan)),
             pi_txt=pi_txt)
    for ax in axes[n:]:
        ax.axis("off")


    for ax in axes[:n]:
        if ax.get_subplotspec().is_last_row() or True:
            ax.set_xticks([3, 13, 23, 33])
    fig.legend(handles, labels, fontsize=10, loc="upper center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.5, 0.968))
    fig.supxlabel("labels acquired (3 seed + 3/round)", fontsize=10)
    fig.supylabel(m.upper(), fontsize=10)
    fig.suptitle(f"Per-target AL curves, {m.upper()}, mean over 20 splits "
                 f"— panels sorted by personalisation gain at 100% labels "
                 f"(mixed_full − global_sklearn), best first",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0.015, 0.02, 1, 0.955])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print("->", a.out)

    # the numbers behind the lead panel
    print(pool[pool.n_labeled.isin([3, 33])]
          .pivot(index="method", columns="n_labeled", values="mean").round(4).to_string())


main()
