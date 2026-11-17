"""The benchmark read-out: both evaluation axes, the paired tests, and the figures.

    python two_stage/analyse.py                      # the newest run in two_stage/results
    python two_stage/analyse.py --run <folder> --out <dir>

Axis 1 -- predictive performance on the target participant
    Learning curves of Macro PR-AUC against labels bought, and a final-budget table with the
    paired difference against equal-budget random sampling. "Paired" is the whole point: every
    arm saw the same split, the same target, the same seed set and the same pool, so the unit of
    comparison is a (split, target) cell and the difference within a cell removes the enormous
    between-participant variance that otherwise swamps a 33-label experiment. Two tests are
    reported because they answer different questions:

      * **Wilcoxon signed-rank over all (split, target) cells** -- is the method better for the
        typical personalised model? Many cells, no normality assumption, but the cells of one
        split share a pool and are not independent.
      * **t-test over per-split means** -- would this replicate on a new day split? Ten
        independent numbers, which is the conservative reading and the one to quote.

Axis 2 -- participant burden equity
    Cohort Gini coefficient G_q and max-to-mean query ratio of the per-participant label counts,
    pooled over every personalised model, against the pool-proportional baseline (``diagnose.py``
    explains why that, and not zero, is the reference). Reported at the full budget and as a
    curve, plus the equity-performance plane, which is where a two-axis claim either holds up or
    does not.

Also written: the phase read-out for the two-stage arms (how much of the gain is in place by the
end of Phase 1), and label efficiency (how many labels each arm needs to reach random's
full-budget score).
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                              # noqa: E402
import numpy as np                                                           # noqa: E402
import pandas as pd                                                          # noqa: E402
from scipy import stats                                                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from diagnose import COLOR, _methods, burden_curve, pool_baseline            # noqa: E402
from pool_attributes import load_attributes                                  # noqa: E402

RESULTS = os.environ.get("TWO_STAGE_RESULTS_DIR") or os.path.join(HERE, "results")
PRIMARY = "macro_prauc"
METRICS = ["macro_prauc", "macro_rocauc", "auroc", "auprc"]


# --------------------------------------------------------------------------- input
def newest_run(target="computer_work", contains=""):
    cands = [d for d in glob.glob(os.path.join(RESULTS, target, f"*{contains}*"))
             if glob.glob(os.path.join(d, "per_round_split*.csv"))]
    if not cands:
        raise SystemExit(f"no run with results under {os.path.join(RESULTS, target)}")
    return max(cands, key=os.path.getmtime)


def load_also(folders, R, Q):
    """Merge other run folders in, relabelled by their Phase 1 budget. -> (R, Q).

    Used for the ``n_calib`` sweep: those runs hold the ``twostage`` arm only, so they need the
    main benchmark's ``random`` arm to be paired against. A merged arm is renamed
    ``twostage@nc<k>`` so one table can hold several schedules, and only the splits present in
    BOTH runs survive -- a paired comparison against a split the other arm never ran is not a
    comparison.
    """
    import json
    for f in folders:
        cfg = json.load(open(os.path.join(f, "config.json")))
        nc = cfg.get("n_calib", "?")
        Rx, Qx = load_run(f)
        # relabel only where it would collide: a merged run holding an arm this run does not have
        # (e.g. `laplace`) keeps its own name, while a second `twostage` at a different Phase 1
        # budget becomes `twostage@nc9`.
        clash = set(Rx.method.unique()) & set(R.method.unique())
        if clash:
            tag = f"@nc{nc}" if cfg.get("tag") is None else f"@{cfg['tag']}"
            Rx["method"] = Rx.method.where(~Rx.method.isin(clash), Rx.method + tag)
            Qx["method"] = Qx.method.where(~Qx.method.isin(clash), Qx.method + tag)
        keep = sorted(set(R.split.unique()) & set(Rx.split.unique()))
        R = pd.concat([R[R.split.isin(keep)], Rx[Rx.split.isin(keep)]], ignore_index=True)
        Q = pd.concat([Q[Q.split.isin(keep)], Qx[Qx.split.isin(keep)]], ignore_index=True)
    return R, Q


def load_run(run):
    R = pd.concat([pd.read_csv(f) for f in
                   sorted(glob.glob(os.path.join(run, "per_round_split*.csv")))],
                  ignore_index=True)
    Q = pd.concat([pd.read_csv(f) for f in
                   sorted(glob.glob(os.path.join(run, "queries_split*.csv")))],
                  ignore_index=True)
    print(f"{run}\n  {len(R)} per-round rows, {len(Q)} queries, "
          f"{R.split.nunique()} splits x {R.target.nunique()} targets x "
          f"{R.method.nunique()} methods, rounds 0..{R['round'].max()}")
    return R, Q


# --------------------------------------------------------------------------- axis 1
def curves(R, metric=PRIMARY):
    """-> mean +/- SE of one metric per (method, round), SE taken over per-split means.

    Keyed on the ROUND, with ``n_labeled`` averaged in afterwards. An arm can end a round with
    fewer labels than nominal -- ``calib`` runs out of own blocks for a participant who
    contributed few, and a greedy batch can come up short -- and grouping on ``n_labeled``
    would then split one round into two points and put a zigzag in the curve.
    """
    per_split = R.groupby(["method", "round", "split"]).agg(
        v=(metric, "mean"), n_labeled=("n_labeled", "mean")).reset_index()
    out = (per_split.groupby(["method", "round"])
           .agg(mean=("v", "mean"), se=("v", lambda v: v.std(ddof=1) / max(np.sqrt(len(v)), 1)),
                n_labeled=("n_labeled", "mean"), n_splits=("split", "nunique")).reset_index())
    return out


def paired_table(R, metric=PRIMARY, ref="random", rnd=None):
    """-> one row per method: final-budget score, paired gap against `ref`, and both tests."""
    rnd = R["round"].max() if rnd is None else rnd                 # the full-budget round
    last = R[R["round"] == rnd]
    wide = last.pivot_table(index=["split", "target"], columns="method", values=metric)
    if ref not in wide:
        raise SystemExit(f"reference method {ref!r} is not in this run")
    rows = []
    for m in wide.columns:
        d = (wide[m] - wide[ref]).dropna()
        cell = d.reset_index().rename(columns={0: "d"})
        cell.columns = ["split", "target", "d"]
        by_split = cell.groupby("split")["d"].mean()
        w = stats.wilcoxon(d) if m != ref and len(d) > 5 and d.abs().sum() > 0 else None
        t = (stats.ttest_1samp(by_split, 0) if m != ref and len(by_split) > 1 else None)
        rows.append(dict(
            method=m, score=float(wide[m].mean()),
            score_se=float(wide[m].groupby("split").mean().std(ddof=1)
                           / max(np.sqrt(wide[m].groupby("split").ngroups), 1)),
            gap=float(d.mean()), gap_se=float(by_split.std(ddof=1)
                                              / max(np.sqrt(len(by_split)), 1)),
            win_rate=float((d > 0).mean()), n_cells=int(len(d)), n_splits=int(len(by_split)),
            wilcoxon_p=float(w.pvalue) if w else np.nan,
            split_t=float(t.statistic) if t else np.nan,
            split_p=float(t.pvalue) if t else np.nan))
    return pd.DataFrame(rows).sort_values("gap", ascending=False)


def label_efficiency(R, metric=PRIMARY, ref="random"):
    """-> labels each method needs to match `ref`'s FULL-budget score (NaN if it never does)."""
    c = curves(R, metric)
    tgt = c[(c.method == ref)].sort_values("n_labeled").iloc[-1]["mean"]
    out = []
    for m, g in c.groupby("method"):
        g = g.sort_values("n_labeled")
        hit = g[g["mean"] >= tgt]
        out.append(dict(method=m, labels_to_match=int(hit.iloc[0].n_labeled) if len(hit)
                        else np.nan, final=float(g.iloc[-1]["mean"])))
    return pd.DataFrame(out).sort_values("labels_to_match"), float(tgt)


def phase_readout(R, Q, n_calib=6, metric=PRIMARY):
    """-> for the scheduled arms, the share of their final gain already present after Phase 1."""
    rows = []
    # labels per round, from one method's own trajectory rather than across methods
    one = R[R.method == R.method.iloc[0]].groupby("round").n_labeled.mean().sort_index()
    d = one.diff().dropna()
    batch = int(round(float(d[d > 0].median()))) if len(d[d > 0]) else 3
    r_switch = int(np.ceil(n_calib / max(batch, 1)))
    last = R["round"].max()
    base = R[R.method == "random"].pivot_table(index=["split", "target"], columns="round",
                                               values=metric)
    for m in sorted(R.method.unique()):
        w = R[R.method == m].pivot_table(index=["split", "target"], columns="round",
                                         values=metric)
        if 0 not in w or r_switch not in w or last not in w:
            continue
        g_phase1 = (w[r_switch] - w[0]).mean()
        g_total = (w[last] - w[0]).mean()
        rows.append(dict(method=m, round_switch=r_switch,
                         gain_phase1=float(g_phase1), gain_total=float(g_total),
                         share_in_phase1=float(g_phase1 / g_total) if g_total else np.nan,
                         gap_at_switch=float((w[r_switch] - base[r_switch]).mean()),
                         gap_at_end=float((w[last] - base[last]).mean())))
    return pd.DataFrame(rows).sort_values("gap_at_end", ascending=False)


def self_share_effect(R, tab, metric=PRIMARY):
    """Is performance at this budget mostly a function of HOW MANY labels came from the target?

    Two readings, which are different claims:

    * **Between arms.** Each arm's mean self-share against its mean gain over random. If this is
      close to a straight line, then what separates nine acquisition strategies at a 33-label
      budget is not their geometry or their information criterion but the share of the budget
      they happened to spend on the target -- which is a statement about the *estimator*, not
      about the strategies.
    * **Within an arm.** The same correlation computed across (split, target) cells with each
      arm's own mean removed, so it is driven by cells where one arm bought more self-labels than
      it usually does. This is the weaker but cleaner test; the between-arm version can be
      confounded by anything else the arms differ in.

    The mechanism being tested is the one measured in ``vc_diagnostic.py``: a GLMM fitted on
    singleton labels from many participants cannot identify its variance components, shrinks them
    to zero, and stops being a personalised model at all.
    """
    rnd = R["round"].max()
    last = R[R["round"] == rnd]
    per_arm = last.groupby("method").agg(self_share=("self_share", "mean"),
                                         score=(metric, "mean")).reset_index()
    per_arm = per_arm.merge(tab[["method", "gap", "gap_se"]], on="method", how="left")
    between = (stats.pearsonr(per_arm.self_share, per_arm.gap)
               if per_arm.self_share.std() > 0 and len(per_arm) > 3 else None)
    # within arm, de-meaned by (method, split)
    cell = last[["method", "split", "target", "self_share", metric]].dropna()
    g = cell.groupby(["method", "split"])
    z = cell.copy()
    for c in ("self_share", metric):
        z[c] = cell[c] - g[c].transform("mean")
    z = z.dropna()
    within = (stats.pearsonr(z.self_share, z[metric])
              if len(z) > 30 and z.self_share.std() > 0 else None)
    return per_arm.sort_values("self_share"), between, within


def fig_selfshare(per_arm, out, between=None, metric=PRIMARY):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for _, r in per_arm.iterrows():
        ax.errorbar(r.self_share, r.gap, yerr=r.gap_se, marker="o", ms=8, capsize=2, lw=1,
                    color=COLOR.get(r.method, "#555"))
        ax.annotate(r.method, (r.self_share, r.gap), textcoords="offset points", xytext=(7, 4),
                    fontsize=8, color=COLOR.get(r.method, "#555"))
    if len(per_arm) > 2:
        b = np.polyfit(per_arm.self_share, per_arm.gap, 1)
        xs = np.linspace(0, max(per_arm.self_share.max(), 0.05), 50)
        ax.plot(xs, np.polyval(b, xs), color="#999", ls="--", lw=1.2,
                label=(f"OLS, r = {between.statistic:+.2f}" if between else "OLS"))
        ax.legend(fontsize=8, frameon=False)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("share of the budget bought from the target participant")
    ax.set_ylabel(f"paired gain in {metric} over random")
    ax.set_title("At a 33-label budget, what separates the arms?", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def ncalib_curve(tab, n_calib_default=6, total=30):
    """-> the schedule's dose-response curve: gain over random against the Phase 1 budget.

    Reads the Phase 1 budget out of the arm names, so the sweep runs (`twostage@nc3`, ...) and the
    two endpoints that are already arms of the main benchmark line up on one axis:
    ``twfeg`` is n_calib = 0 (no calibration phase at all) and ``calib`` is n_calib = the whole
    budget (never switch). This is the figure that says whether "5-10 queries" is the right size
    for Phase 1 or whether the schedule wants more.
    """
    rows = []
    for _, r in tab.iterrows():
        m = str(r.method)
        if m == "twfeg":
            nc = 0
        elif m == "calib":
            nc = total
        elif m == "twostage":
            nc = n_calib_default
        elif m.startswith("twostage@nc"):
            try:
                nc = int(m.split("@nc")[1])
            except ValueError:
                continue
        else:
            continue
        rows.append(dict(n_calib=nc, method=m, gap=r.gap, gap_se=r.gap_se, score=r.score,
                         win_rate=r.win_rate, split_p=r.split_p))
    return pd.DataFrame(rows).sort_values("n_calib")


def fig_ncalib(nc, out, metric=PRIMARY, ref_gap=None):
    if len(nc) < 3:
        return False
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    if ref_gap is not None:
        # the control from 5.3: uniform sampling restricted to the target's own blocks. The
        # schedule only overtakes it once most of the budget is in Phase 1, which is the single
        # most useful thing this figure says.
        ax.axhline(ref_gap, color="#b5503f", ls="--", lw=1.3,
                   label="self_random control")
    ax.errorbar(nc.n_calib, nc.gap, yerr=nc.gap_se, color="#1b4f72", marker="o", ms=5, lw=1.8,
                capsize=3)
    ax.axhline(0, color="k", lw=0.9, ls="--")
    for _, r in nc.iterrows():
        ax.annotate(r.method.replace("twostage@nc", ""), (r.n_calib, r.gap),
                    textcoords="offset points", xytext=(0, 9), fontsize=7, ha="center",
                    color="#555")
    ax.set_xlabel("labels spent on Phase 1 (Target Calibration) out of 30")
    ax.set_ylabel(f"paired gain in {metric} over random")
    ax.set_title("Where does the phase boundary belong?\n"
                 "0 = TW-FEG only, 30 = calibration only", fontsize=10)
    if ref_gap is not None:
        ax.annotate("self_random: uniform over the target's own blocks", (0.2, ref_gap),
                    textcoords="offset points", xytext=(0, 5), fontsize=8, color="#b5503f")
        ax.set_ylim(min(0, float(nc.gap.min()) - 0.012), None)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- figures
def fig_curves(R, out, metric=PRIMARY, methods=None):
    c = curves(R, metric)
    ms = methods or _methods(None, c.method.unique())
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for m in ms:
        g = c[c.method == m].sort_values("n_labeled")
        ax.plot(g.n_labeled, g["mean"], color=COLOR.get(m, "#555"), lw=1.8, marker="o", ms=3,
                label=m)
        ax.fill_between(g.n_labeled, g["mean"] - g.se, g["mean"] + g.se,
                        color=COLOR.get(m, "#555"), alpha=0.13, lw=0)
    ax.set_xlabel("labels revealed (3 seed + 3 per round)")
    ax.set_ylabel({"macro_prauc": "Macro PR-AUC on the target's held-out days"}
                  .get(metric, metric))
    ax.set_title(f"Personalised performance under a low label budget\n"
                 f"mean over {R.target.nunique()} targets x {R.split.nunique()} splits, "
                 f"band = SE over splits", fontsize=10)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_per_person(R, out, metric=PRIMARY, rows=5, order_by=None, free_y=False):
    """One panel per target participant: each arm's curve, averaged over splits, +/-1 SE.

    The pooled curve in ``fig_curves.png`` hides the thing a personalised method lives or dies
    on, which is how many *individual* participants it helps. 43 panels say that directly: a
    method with a good mean and a bad median is visible here and nowhere else.

    Panels are ordered by the acquisition-minus-random gap at the final round, best first, with
    the cohort mean as the last panel — the parent repo's convention in
    ``plots/per_person_grid.py``, so the two grids read the same way. Read that ordering with
    care: sorting panels by a noisy quantity always produces a left-to-right gradient, so the
    leading panels look good even when every participant has the same true gap. The number
    printed in each panel title is that participant's gap, and the split-level spread behind it
    is in ``per_person_gap.csv``.

    The y-axis is shared by default. Per-participant levels differ enormously (a target whose
    test days are nearly all negative scores 0.9 on anything), so free axes would make every
    panel look like a dramatic improvement; ``free_y=True`` is there when the shapes rather than
    the levels are the question.
    """
    ms = _methods(None, R.method.unique())
    ref = "random" if "random" in ms else ms[-1]
    order_by = order_by or next((m for m in ms if m != ref), ref)
    last = R["round"].max()
    # per (method, target, round): mean over splits and the SE of that mean
    g = (R.groupby(["method", "target", "round"])
          .agg(v=(metric, "mean"), sd=(metric, "std"), n=(metric, "size"),
               n_labeled=("n_labeled", "mean")).reset_index())
    g["se"] = g.sd / np.sqrt(g.n.clip(lower=1))
    fin = g[g["round"] == last].pivot_table(index="target", columns="method", values="v")
    if order_by in fin and ref in fin:
        gap = (fin[order_by] - fin[ref]).dropna().sort_values(ascending=False)
    else:
        gap = pd.Series(dtype=float, index=fin.index)
    targets = list(gap.index) + [t for t in fin.index if t not in gap.index]

    n_panels = len(targets) + 1                              # + the cohort mean
    cols = int(np.ceil(n_panels / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(1.75 * cols, 1.55 * rows),
                             sharex=True, sharey=not free_y)
    axes = np.atleast_1d(axes).ravel()
    # full range, not percentiles: a shared axis that clips the best participant's curve is
    # worse than one that is slightly loose
    lo = float(np.nanmin(g.v - g.se.fillna(0))) - 0.01
    hi = float(np.nanmax(g.v + g.se.fillna(0))) + 0.01

    for ax, t in zip(axes, targets):
        for m in ms:
            d = g[(g.method == m) & (g.target == t)].sort_values("n_labeled")
            if not len(d):
                continue
            c = COLOR.get(m, "#555")
            ax.plot(d.n_labeled, d.v, color=c, lw=1.2)
            ax.fill_between(d.n_labeled, d.v - d.se, d.v + d.se, color=c, alpha=0.15, lw=0)
        ttl = f"{t}" + (f"  {gap[t]:+.02f}" if t in gap.index and np.isfinite(gap[t]) else "")
        ax.set_title(ttl, fontsize=7, pad=2)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2)
        if not free_y:
            ax.set_ylim(lo, hi)

    # the cohort mean, computed exactly as fig_curves does: per-split target means, then the
    # mean over splits with its SE -- not an average of the panels, which would weight splits
    # by how many targets survived in them
    axm = axes[len(targets)]
    c_all = curves(R, metric)
    for m in ms:
        d = c_all[c_all.method == m].sort_values("n_labeled")
        c = COLOR.get(m, "#555")
        axm.plot(d.n_labeled, d["mean"], color=c, lw=1.8)
        axm.fill_between(d.n_labeled, d["mean"] - d.se, d["mean"] + d.se, color=c, alpha=0.2,
                         lw=0)
    axm.set_title("cohort mean", fontsize=7, fontweight="bold", pad=2)
    axm.tick_params(labelsize=6)
    axm.grid(alpha=0.2)
    if not free_y:
        axm.set_ylim(lo, hi)
    for ax in axes[n_panels:]:
        ax.axis("off")

    handles = [plt.Line2D([], [], color=COLOR.get(m, "#555"), lw=2, label=m) for m in ms]
    fig.legend(handles=handles, loc="lower right", fontsize=8, frameon=False,
               bbox_to_anchor=(0.995, 0.01))
    fig.supxlabel("labels revealed", fontsize=9)
    fig.supylabel({"macro_prauc": "Macro PR-AUC on the target's held-out days"}.get(metric,
                                                                                    metric),
                  fontsize=9)
    fig.suptitle(f"Per-participant learning curves — {R.split.nunique()} splits, "
                 f"band = +/-1 SE over splits; panel titles carry the final gap over {ref}",
                 fontsize=9)
    fig.tight_layout(rect=(0.01, 0.02, 1, 0.97))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return gap


def fig_gap(tab, out, metric=PRIMARY):
    t = tab[tab.method != "random"].sort_values("gap")
    fig, ax = plt.subplots(figsize=(6.0, 0.42 * len(t) + 1.8))
    ax.barh(range(len(t)), t.gap, xerr=t.gap_se, height=0.7,
            color=[COLOR.get(m, "#555") for m in t.method],
            error_kw=dict(lw=0.9, ecolor="#222"))
    ax.axvline(0, color="k", lw=0.9)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([f"{m}  ({p:.3f})" for m, p in zip(t.method, t.split_p)], fontsize=8)
    ax.set_xlabel(f"paired gain in {metric} over equal-budget random (SE over splits)")
    ax.set_title("Final-budget gain against random, per-cell paired\n"
                 "(split-level p in brackets)", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_plane(tab, bur, out, base=None, metric=PRIMARY):
    """The two axes at once: performance gain against burden inequality."""
    b = bur.sort_values("budget").groupby("method").tail(1).set_index("method")
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for _, r in tab.iterrows():
        m = r.method
        if m not in b.index:
            continue
        ax.errorbar(b.loc[m, "gini"], r.gap, xerr=b.loc[m, "gini_se"], yerr=r.gap_se,
                    color=COLOR.get(m, "#555"), marker="o", ms=8, capsize=2, lw=1)
        ax.annotate(m, (b.loc[m, "gini"], r.gap), textcoords="offset points", xytext=(7, 4),
                    fontsize=8, color=COLOR.get(m, "#555"))
    ax.axhline(0, color="k", lw=0.8, ls="--")
    if base:
        ax.axvline(base["gini"], color="k", lw=1.0, ls=":")
        ax.annotate("pool-proportional", (base["gini"], ax.get_ylim()[0]),
                    textcoords="offset points", xytext=(4, 8), fontsize=7, rotation=90)
    ax.set_xlabel("cohort burden inequality, Gini $G_q$  (lower is fairer)")
    ax.set_ylabel(f"paired gain in {metric} over random  (higher is better)")
    ax.set_title("The equity-performance plane at the full budget\n"
                 "up and to the left is better", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="run folder (default: newest in two_stage/results)")
    ap.add_argument("--target", default="computer_work")
    ap.add_argument("--contains", default="")
    ap.add_argument("--metric", default=PRIMARY, choices=METRICS)
    ap.add_argument("--n_calib", type=int, default=6)
    ap.add_argument("--block", type=int, default=15)
    ap.add_argument("--methods", nargs="+", default=None,
                    help="restrict the read-out to these arms (the reference `random` is kept "
                         "automatically). The headline comparison of the framework is "
                         "`--methods twostage`; the other arms stay on disk either way")
    ap.add_argument("--also", nargs="+", default=None,
                    help="merge these run folders in, relabelled by their --n_calib "
                         "(for the Phase 1 budget sweep)")
    ap.add_argument("--rows", type=int, default=5, help="panel rows in the per-participant grid")
    ap.add_argument("--free_y", action="store_true",
                    help="per-participant panels get their own y-range (shapes, not levels)")
    ap.add_argument("--out", default=None, help="default: <run>/analysis")
    a = ap.parse_args()

    run = a.run or newest_run(a.target, a.contains)
    out = a.out or os.path.join(run, "analysis")
    os.makedirs(out, exist_ok=True)
    R, Q = load_run(run)
    if a.also:
        R, Q = load_also(a.also, R, Q)
    if a.methods:
        keep = set(a.methods) | {"random"}
        # a prefix match so a sweep arm can be named by its base (`twostage` keeps
        # `twostage@nc9`), which is what makes the schedule sweep readable in one call
        sel = R.method.isin(keep) | R.method.str.split("@").str[0].isin(keep)
        selq = Q.method.isin(keep) | Q.method.str.split("@").str[0].isin(keep)
        R, Q = R[sel], Q[selq]
        print(f"  restricted to {sorted(R.method.unique())}")
    splits = sorted(R.split.unique())

    # --- axis 1 -------------------------------------------------------------
    tabs = {m: paired_table(R, m) for m in METRICS}
    tab = tabs[a.metric]
    for m, t in tabs.items():
        t.to_csv(os.path.join(out, f"final_{m}.csv"), index=False)
    curves(R, a.metric).to_csv(os.path.join(out, "curves.csv"), index=False)
    eff, ref_score = label_efficiency(R, a.metric)
    ph = phase_readout(R, Q, a.n_calib, a.metric)
    eff.to_csv(os.path.join(out, "label_efficiency.csv"), index=False)
    ph.to_csv(os.path.join(out, "phase_readout.csv"), index=False)

    # --- axis 2 -------------------------------------------------------------
    _, P = load_attributes(a.target, splits, a.block, verbose=False)
    bur, bur_raw = burden_curve(Q, P)
    base = pool_baseline(P)
    bur.to_csv(os.path.join(out, "burden_curve.csv"), index=False)

    per_arm, between, within = self_share_effect(R, tab, a.metric)
    per_arm.to_csv(os.path.join(out, "self_share_effect.csv"), index=False)
    fig_selfshare(per_arm, os.path.join(out, "fig_selfshare.png"), between, a.metric)
    print("\n=== what separates the arms: self-share vs gain ===")
    print(per_arm.set_index("method").round(4).to_string())
    if between:
        print(f"between arms: Pearson r = {between.statistic:+.3f} (p = {between.pvalue:.4f}, "
              f"n = {len(per_arm)} arms)")
    if within:
        print(f"within arm x split, de-meaned: r = {within.statistic:+.3f} "
              f"(p = {within.pvalue:.2g})")

    nc = ncalib_curve(tab, a.n_calib, int(R.acquired.max()))
    if len(nc):
        nc.to_csv(os.path.join(out, "ncalib_curve.csv"), index=False)
        ref_row = tab[tab.method == "self_random"]
        ref_gap = float(ref_row.gap.iloc[0]) if len(ref_row) else None
        if fig_ncalib(nc, os.path.join(out, "fig_ncalib.png"), a.metric, ref_gap):
            print("\n=== the schedule's dose-response: Phase 1 budget vs gain over random ===")
            print(nc.set_index("n_calib")[["method", "gap", "gap_se", "win_rate", "split_p"]]
                    .round(4).to_string())
    fig_curves(R, os.path.join(out, "fig_curves.png"), a.metric)
    gap_pp = fig_per_person(R, os.path.join(out, "fig_curves_per_person.png"), a.metric,
                            rows=a.rows, free_y=a.free_y)
    if len(gap_pp):
        gap_pp.rename("gap").to_csv(os.path.join(out, "per_person_gap.csv"))
        print(f"\n=== per participant ({a.metric} gap over random at the full budget) ===")
        print(f"  better on {int((gap_pp > 0).sum())}/{len(gap_pp)} participants; "
              f"best {gap_pp.index[0]} {gap_pp.iloc[0]:+.4f}, "
              f"worst {gap_pp.index[-1]} {gap_pp.iloc[-1]:+.4f}, "
              f"median {gap_pp.median():+.4f}")
    fig_gap(tab, os.path.join(out, "fig_gap.png"), a.metric)
    fig_plane(tab, bur, os.path.join(out, "fig_plane.png"), base, a.metric)

    # --- printed read-out ---------------------------------------------------
    pd.set_option("display.width", 200)
    print(f"\n=== axis 1: {a.metric} at the full budget ({R.n_labeled.max()} labels), "
          f"paired vs random ===")
    print(tab.set_index("method")[["score", "score_se", "gap", "gap_se", "win_rate",
                                   "wilcoxon_p", "split_p", "n_cells", "n_splits"]]
             .round(4).to_string())
    print("\n--- the same table on the other metrics (gap vs random) ---")
    comp = pd.DataFrame({m: t.set_index("method")["gap"] for m, t in tabs.items()})
    print(comp.reindex(_methods(None, comp.index)).round(4).to_string())

    print(f"\n=== label efficiency: labels needed to reach random's full-budget "
          f"{a.metric} = {ref_score:.4f} ===")
    print(eff.set_index("method").round(4).to_string())

    print("\n=== phase read-out (share of the arm's own gain in place after Phase 1) ===")
    print(ph.set_index("method").round(4).to_string())

    print("\n=== axis 2: cohort burden at the full budget ===")
    last = bur.sort_values("budget").groupby("method").tail(1)
    print(last.set_index("method").reindex(_methods(None, last.method))
              [["gini", "gini_se", "max_to_mean", "max_to_mean_se", "top_share", "n_touched",
                "self_share"]].round(3).to_string())
    print(f"pool-proportional reference: Gini {base['gini']:.3f}, "
          f"max-to-mean {base['max_to_mean']:.2f}")
    print(f"\n-> tables and figures in {out}")


if __name__ == "__main__":
    main()
