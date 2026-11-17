"""Stage 0: what the geometric acquisitions actually buy, and what it costs the cohort.

    python two_stage/diagnose.py                       # every r10x3 run on disk
    python two_stage/diagnose.py --splits 0 1 2 --out two_stage/diagnosis

This is the empirical half of the proposal's diagnosis. It takes the ``queries_split*.csv``
records of finished runs -- which row, bought by which method, for which target, owned by whom --
joins the method-independent row attributes from ``pool_attributes.py``, and answers three
questions with numbers rather than assertion:

1. **What kind of block does each method buy?** For every attribute, the mean over the rows a
   method bought against the pool mean, as a standardised effect size

       d = (mean over queried rows - pool mean) / pool SD

   computed per split and averaged, with the paired difference against random sampling on the
   same split. d = 0 is "indistinguishable from drawing at random"; the interesting numbers are
   on ``knn_dist`` (isolation), ``own_z`` (subject-specific extremity), ``impurity`` and
   ``is_transition`` (phase-shift blocks), and ``local_purity`` (label noise, sign reversed).

2. **Whose labels?** Per participant, the share of all purchased labels against their share of
   the candidate pool, and the rank correlation between that share and the participant's
   isolation from the rest of the cohort. A geometric method that over-buys from outlier
   individuals shows up here as a positive correlation; the burden axis then says how much.

3. **What does the burden distribution look like as the budget grows?** Gini coefficient and
   max-to-mean ratio of the per-participant query counts, pooled over every personalised model in
   the cohort (every participant takes a turn as the target), as a function of labels per target.
   Also the Lorenz curves at the final budget, which is the same information without the summary
   statistic.

Every number is computed per split and then averaged over splits, because the pool, the day
split and the eligible cohort all change with the split; the per-split spread is the error bar
that matters. Methods differ in how many splits exist on disk, so every table carries
``n_splits`` and the figures say which splits they used.
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
from scipy.stats import spearmanr                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from metrics import burden, counts_over, gini, max_to_mean                   # noqa: E402
from pool_attributes import load_attributes                                  # noqa: E402

ATTRS = ["knn_dist", "own_z", "impurity", "is_transition", "local_purity", "pos"]
NICE = {"knn_dist": "isolation\n(mean kNN distance)", "own_z": "extremity for its owner\n(own_z)",
        "impurity": "phase-shift block\n(label impurity)", "is_transition": "transition block",
        "local_purity": "neighbourhood label purity", "pos": "positive label"}
ORDER = ["twostage", "twfeg", "calib", "bald", "coreset", "probcover", "typiclust", "eer",
         "laplace", "random", "self_random", "twostage_kmed", "twfeg_own"]
# `random` is orange to match the parent repo's plotting convention (plots/per_person_grid.py
# draws random in darkorange), so a curve means the same thing in both sets of figures. That
# frees nothing else to be orange: probcover moves to mustard and typiclust to brown.
COLOR = {"twostage": "#1b4f72", "twfeg": "#2e86c1", "calib": "#85c1e9", "bald": "#8e44ad",
         "coreset": "#b5503f", "probcover": "#7d6608", "typiclust": "#935116",
         "eer": "#117a65", "laplace": "#2a7f62", "random": "#ff8c00",
         "self_random": "#f5b041", "twostage_kmed": "#5499c7", "twfeg_own": "#a9cce3"}


# --------------------------------------------------------------------------- input
def _query_files(roots, contains, target, runs=None):
    """The query CSVs to read: either the explicitly named run folders, or a glob over the roots.

    Naming runs explicitly matters once several runs hold an arm with the SAME name but different
    settings -- the Phase 1 budget sweep writes a `twostage` arm per value of ``n_calib`` -- since
    the de-duplication below keys on (method, split, target, round, row) and would silently pool
    them into one method. It also keeps the parent repo's pre-seed-fix `random` arm out: those
    runs replayed one draw of 30 rows for all 43 targets (see the note in the parent `main.py`),
    which leaves its *attribute* means unbiased but its burden statistics meaningless.
    """
    if runs:
        files = []
        for r in runs:
            f = sorted(glob.glob(os.path.join(r, "queries_split*.csv")))
            if not f:
                raise SystemExit(f"no queries_split*.csv in {r}")
            files += f
        return files
    files = []
    for root in roots:
        files += sorted(glob.glob(os.path.join(root, target, f"*{contains}*",
                                               "queries_split*.csv")))
    return files


def load_queries(roots, contains="r10x3", target="computer_work", runs=None):
    """-> every query record on disk under the given results roots, de-duplicated.

    A method can appear in several run folders (``random`` was paired with every acquisition in
    the parent repo), and the seed set and strategy RNG do not depend on the method list, so
    those copies are identical draws rather than independent replicates. Keeping one row per
    (method, split, target, round, row) stops a method being counted once per folder.
    """
    files = _query_files(roots, contains, target, runs)
    if not files:
        raise SystemExit(f"no queries_split*.csv matching *{contains}* under {roots}")
    Q = pd.concat([pd.read_csv(f).assign(run=os.path.basename(os.path.dirname(f)))
                   for f in files], ignore_index=True)
    before = len(Q)
    Q = Q.drop_duplicates(["method", "split", "target", "round", "row"])
    print(f"{len(files)} query file(s): {len(Q)} records ({before - len(Q)} duplicate rows "
          f"dropped), methods {sorted(Q.method.unique())}, splits {sorted(Q.split.unique())}")
    return Q


# --------------------------------------------------------------------------- 1. what is bought
def row_profile(Q, R):
    """-> per (method, attribute) effect size against the pool, and against random."""
    rows = []
    for s, Rs in R.groupby("split"):
        pool_mu = Rs[ATTRS].mean()
        pool_sd = Rs[ATTRS].std(ddof=1).replace(0, np.nan)
        qs = Q[Q.split == s].merge(Rs[["row"] + ATTRS], on="row", how="left",
                                   suffixes=("", "_attr"))
        if qs[ATTRS].isna().all(axis=None):
            continue
        for m, g in qs.groupby("method"):
            mu = g[ATTRS].mean()
            d = (mu - pool_mu) / pool_sd
            for a in ATTRS:
                rows.append(dict(split=s, method=m, attr=a, queried=mu[a], pool=pool_mu[a],
                                 d=d[a], n=len(g)))
    D = pd.DataFrame(rows)
    rnd = D[D.method == "random"].set_index(["split", "attr"])["d"]
    D["d_vs_random"] = D.apply(lambda r: r["d"] - rnd.get((r["split"], r["attr"]), np.nan), axis=1)
    out = (D.groupby(["method", "attr"])
             .agg(d=("d", "mean"), d_se=("d", lambda v: v.std(ddof=1) / max(np.sqrt(len(v)), 1)),
                  d_vs_random=("d_vs_random", "mean"),
                  d_vs_random_se=("d_vs_random",
                                  lambda v: v.std(ddof=1) / max(np.sqrt(len(v)), 1)),
                  queried=("queried", "mean"), pool=("pool", "mean"),
                  n_splits=("split", "nunique"), n_queries=("n", "sum"))
             .reset_index())
    out["ratio"] = out.queried / out["pool"]
    return out, D


def load_scores(roots, contains="r10x3", target="computer_work", runs=None):
    """-> every per-round score record, from the same run folders as the queries."""
    files = [f.replace("queries_split", "per_round_split")
             for f in _query_files(roots, contains, target, runs)]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        return pd.DataFrame()
    S = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return S.drop_duplicates(["method", "split", "target", "round"])


def mechanism(Q, R, S, metric="auroc", attrs=ATTRS):
    """Does buying noisier blocks actually cost performance? -> (correlation table, cells).

    The diagnosis so far establishes a *difference in what is bought* and a *difference in
    outcome*; it does not establish that the first explains the second. This closes that gap as
    far as observational data can: for every (split, target, method) cell, the mean attribute of
    the blocks that cell bought against its final score, with the cell's score and attributes
    **de-meaned within (method, split)**. De-meaning within method removes every between-method
    difference -- so the correlation is not "methods that buy anomalies score worse", which could
    be driven by anything else those methods do, but "*within one method on one split*, the
    targets for which it happened to buy more anomalous blocks scored worse". De-meaning within
    split additionally removes the split's own difficulty.

    Reported on AUROC rather than Macro PR-AUC by default, because the parent repo's runs -- which
    supply the `typiclust`, `eer` and `laplace` arms -- predate the macro columns.
    """
    if not len(S):
        return pd.DataFrame(), pd.DataFrame()
    last = S[S["round"] == S.groupby(["method", "split", "target"])["round"].transform("max")]
    cells = []
    for sp, Rs in R.groupby("split"):
        q = Q[Q.split == sp].merge(Rs[["row"] + attrs], on="row", how="left")
        g = q.groupby(["method", "target"])[attrs].mean().reset_index()
        g["split"] = sp
        cells.append(g)
    if not cells:
        return pd.DataFrame(), pd.DataFrame()
    C = pd.concat(cells, ignore_index=True).merge(
        last[["method", "split", "target", metric]], on=["method", "split", "target"],
        how="inner").dropna(subset=[metric])
    if not len(C):
        return pd.DataFrame(), C
    grp = C.groupby(["method", "split"])
    Z = C.copy()
    for col in attrs + [metric]:
        Z[col] = C[col] - grp[col].transform("mean")
    rows = []
    for a in attrs:
        d = Z[[a, metric]].dropna()
        if len(d) < 30 or d[a].std() == 0:
            continue
        r = stats.pearsonr(d[a], d[metric])
        rho = spearmanr(d[a], d[metric])
        rows.append(dict(attr=a, pearson_r=r.statistic, pearson_p=r.pvalue,
                         spearman_rho=rho.statistic, spearman_p=rho.pvalue, n_cells=len(d)))
    return pd.DataFrame(rows), C


# --------------------------------------------------------------------------- 2. whose labels
def person_profile(Q, P):
    """-> per (method, participant) query share vs pool share, plus the isolation correlation."""
    rows = []
    for s, Ps in P.groupby("split"):
        cohort = list(Ps.owner)
        share = dict(zip(Ps.owner, Ps.pool_share))
        iso = dict(zip(Ps.owner, Ps.isolation))
        for m, g in Q[Q.split == s].groupby("method"):
            c = counts_over(cohort, list(g.owner))
            tot = c.sum()
            for u, n in zip(cohort, c):
                rows.append(dict(split=s, method=m, owner=u, n_queries=n,
                                 q_share=n / tot if tot else np.nan, pool_share=share[u],
                                 isolation=iso[u]))
    PP = pd.DataFrame(rows)
    PP["over"] = PP.q_share - PP.pool_share
    corr = []
    for (m, s), g in PP.groupby(["method", "split"]):
        rho = spearmanr(g.isolation, g.q_share).statistic
        top3 = g.nlargest(3, "isolation")
        corr.append(dict(method=m, split=s, rho_isolation=rho,
                         top3_iso_share=float(top3.q_share.sum()),
                         top3_pool_share=float(top3.pool_share.sum())))
    C = pd.DataFrame(corr)
    agg = (C.groupby("method")
            .agg(rho_isolation=("rho_isolation", "mean"),
                 rho_se=("rho_isolation", lambda v: v.std(ddof=1) / max(np.sqrt(len(v)), 1)),
                 top3_iso_share=("top3_iso_share", "mean"),
                 top3_pool_share=("top3_pool_share", "mean"),
                 n_splits=("split", "nunique")).reset_index())
    agg["top3_overbuy"] = agg.top3_iso_share / agg.top3_pool_share
    return agg, PP


# --------------------------------------------------------------------------- 3. burden
def burden_curve(Q, P, batch=3):
    """-> cohort burden as a function of the per-target budget, per method and split."""
    rows = []
    for s, Ps in P.groupby("split"):
        cohort = list(Ps.owner)
        Qs = Q[Q.split == s]
        for m, g in Qs.groupby("method"):
            for cut in sorted(g["round"].unique()):
                sub = g[g["round"] <= cut]
                b = burden(cohort, list(sub.owner))
                rows.append(dict(split=s, method=m, round=int(cut),
                                 budget=int(cut) * batch, **b,
                                 self_share=float(sub.is_self.mean())))
    B = pd.DataFrame(rows)
    agg = (B.groupby(["method", "round", "budget"])
            .agg(gini=("gini", "mean"), gini_se=("gini", lambda v: v.std(ddof=1) /
                                                  max(np.sqrt(len(v)), 1)),
                 max_to_mean=("max_to_mean", "mean"),
                 max_to_mean_se=("max_to_mean", lambda v: v.std(ddof=1) /
                                 max(np.sqrt(len(v)), 1)),
                 top_share=("top_share", "mean"), n_touched=("n_touched", "mean"),
                 self_share=("self_share", "mean"), n_splits=("split", "nunique"))
            .reset_index())
    return agg, B


def pool_baseline(P):
    """-> the burden statistics that sampling PROPORTIONALLY TO THE POOL would produce.

    The reference for G_q is not zero. Participants contribute very different numbers of blocks
    (43 to 400 at W=15), so even a perfectly unbiased sampler inherits the pool's own inequality;
    reading a method's Gini against 0 would call that inherited inequality a fairness failure.
    This is the line every burden figure is drawn against.
    """
    rows = []
    for s, Ps in P.groupby("split"):
        n = Ps.n_pool.to_numpy(float)
        rows.append(dict(split=s, gini=gini(n), max_to_mean=max_to_mean(n),
                         top_share=float(n.max() / n.sum())))
    B = pd.DataFrame(rows)
    return dict(gini=float(B.gini.mean()), gini_se=float(B.gini.std(ddof=1) /
                max(np.sqrt(len(B)), 1)), max_to_mean=float(B.max_to_mean.mean()),
                top_share=float(B.top_share.mean()), n_splits=int(B.split.nunique()))


def lorenz(Q, P, round_max=None):
    """-> {method: (x, y)} Lorenz curves of the per-participant burden at the final budget."""
    out = {}
    for m, g in Q.groupby("method"):
        cs = []
        for s, Ps in P.groupby("split"):
            gs = g[g.split == s]
            if round_max is not None:
                gs = gs[gs["round"] <= round_max]
            if not len(gs):
                continue
            c = np.sort(counts_over(list(Ps.owner), list(gs.owner)))
            cs.append(c / c.sum())
        if not cs:
            continue
        c = np.mean(np.vstack(cs), axis=0)
        y = np.r_[0, np.cumsum(c)]
        out[m] = (np.linspace(0, 1, len(y)), y / y[-1])
    return out


# --------------------------------------------------------------------------- figures
def _methods(df, available):
    """Methods in a fixed display order, unknown ones appended alphabetically.

    ``available`` is coerced to a set of values on purpose: ``"coreset" in some_series`` tests
    the pandas *index*, not the values, and silently returns False -- which empties the table
    rather than raising.
    """
    av = set(np.asarray(list(available)).ravel().tolist())
    return [m for m in ORDER if m in av] + sorted(av - set(ORDER))


def fig_row_profile(prof, out, show=("knn_dist", "own_z", "impurity", "is_transition",
                                     "local_purity")):
    ms = _methods(prof, prof.method.unique())
    fig, axes = plt.subplots(1, len(show), figsize=(3.0 * len(show), 0.34 * len(ms) + 2.2),
                             sharey=True)
    for ax, a in zip(np.atleast_1d(axes), show):
        g = prof[prof.attr == a].set_index("method").reindex(ms)
        ax.barh(range(len(ms)), g.d, xerr=g.d_se, color=[COLOR.get(m, "#555") for m in ms],
                height=0.72, error_kw=dict(lw=0.8, ecolor="#333"))
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(NICE.get(a, a), fontsize=9)
        ax.set_xlabel("effect size d vs pool", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", alpha=0.25)
    np.atleast_1d(axes)[0].set_yticks(range(len(ms)))
    np.atleast_1d(axes)[0].set_yticklabels(ms, fontsize=8)
    fig.suptitle("What each acquisition buys: queried blocks vs the pool they were drawn from",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_burden(agg, out, base=None):
    ms = _methods(agg, agg.method.unique())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    if base:
        axes[0].axhline(base["gini"], color="k", ls=":", lw=1.2,
                        label="pool-proportional sampling")
        axes[1].axhline(base["max_to_mean"], color="k", ls=":", lw=1.2)
    for m in ms:
        g = agg[agg.method == m].sort_values("budget")
        c = COLOR.get(m, "#555")
        axes[0].errorbar(g.budget, g.gini, yerr=g.gini_se, color=c, marker="o", ms=3, lw=1.5,
                         label=m, capsize=2)
        axes[1].errorbar(g.budget, g.max_to_mean, yerr=g.max_to_mean_se, color=c, marker="o",
                         ms=3, lw=1.5, capsize=2)
        axes[2].plot(g.budget, g.n_touched, color=c, marker="o", ms=3, lw=1.5)
    axes[0].set_ylabel("Gini coefficient $G_q$ of query counts")
    axes[1].set_ylabel("max-to-mean query ratio")
    axes[2].set_ylabel("participants ever asked")
    for ax, t in zip(axes, ["Burden inequality", "Worst-off participant", "Coverage"]):
        ax.set_xlabel("labels bought per target")
        ax.set_title(t, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2, frameon=False)
    fig.suptitle("Participant burden equity across the cohort, pooled over every "
                 "personalised model", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_lorenz(cur, out):
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect equality")
    for m in _methods(None, cur.keys()):
        x, y = cur[m]
        ax.plot(x, y, color=COLOR.get(m, "#555"), lw=1.8, label=m)
    ax.set_xlabel("cumulative share of participants (least-asked first)")
    ax.set_ylabel("cumulative share of labels supplied")
    ax.set_title("Lorenz curves of labelling burden at the full budget", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_isolation(PP, out):
    ms = [m for m in _methods(None, PP.method.unique())
          if m in ("twostage", "twfeg", "bald", "coreset", "probcover", "random")]
    fig, axes = plt.subplots(1, len(ms), figsize=(2.5 * len(ms), 3.1), sharey=True, sharex=True)
    for ax, m in zip(np.atleast_1d(axes), ms):
        g = PP[PP.method == m].groupby("owner").agg(q=("q_share", "mean"),
                                                    p=("pool_share", "mean"),
                                                    iso=("isolation", "mean"))
        ax.scatter(g.iso, g.q / g.p, s=14, color=COLOR.get(m, "#555"), alpha=0.8)
        ax.axhline(1, color="k", lw=0.8, ls="--")
        rho = spearmanr(g.iso, g.q / g.p).statistic
        ax.set_title(f"{m}\nSpearman rho = {rho:+.2f}", fontsize=9)
        ax.set_xlabel("participant isolation", fontsize=8)
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
    np.atleast_1d(axes)[0].set_ylabel("queried share / pool share", fontsize=9)
    fig.suptitle("Over-querying of outlier individuals", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="computer_work")
    ap.add_argument("--block", type=int, default=15)
    ap.add_argument("--contains", default="r10x3")
    ap.add_argument("--roots", nargs="+",
                    default=[os.path.join(HERE, "results"), os.path.join(REPO, "results")])
    ap.add_argument("--runs", nargs="+", default=None,
                    help="explicit run folders, instead of globbing the roots. Use this when "
                         "two runs hold an arm with the same name but different settings")
    ap.add_argument("--splits", type=int, nargs="+", default=None,
                    help="restrict to these splits (default: every split present)")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--k", type=int, default=20, help="neighbours for the kNN attributes")
    ap.add_argument("--out", default=os.path.join(HERE, "diagnosis"))
    ap.add_argument("--mech_metric", default="auroc",
                    help="score column for the mechanism test (auroc is present in every run; "
                         "macro_prauc only in this subdirectory's runs)")
    ap.add_argument("--refresh", action="store_true", help="recompute the attribute cache")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    Q = load_queries(a.roots, a.contains, a.target, a.runs)
    if a.methods:
        Q = Q[Q.method.isin(a.methods)]
    splits = a.splits or sorted(Q.split.unique())
    Q = Q[Q.split.isin(splits)]
    R, P = load_attributes(a.target, splits, a.block, a.k, a.refresh)

    S = load_scores(a.roots, a.contains, a.target, a.runs)
    S = S[S.split.isin(splits)] if len(S) else S
    prof, prof_raw = row_profile(Q, R)
    pers, PP = person_profile(Q, P)
    bcurve, braw = burden_curve(Q, P)
    prof.to_csv(os.path.join(a.out, "row_profile.csv"), index=False)
    pers.to_csv(os.path.join(a.out, "person_profile.csv"), index=False)
    bcurve.to_csv(os.path.join(a.out, "burden_curve.csv"), index=False)
    PP.to_csv(os.path.join(a.out, "person_shares.csv"), index=False)

    fig_row_profile(prof, os.path.join(a.out, "fig_row_profile.png"))
    base = pool_baseline(P)
    pd.DataFrame([base]).to_csv(os.path.join(a.out, "pool_baseline.csv"), index=False)
    fig_burden(bcurve, os.path.join(a.out, "fig_burden.png"), base)
    fig_lorenz(lorenz(Q, P), os.path.join(a.out, "fig_lorenz.png"))
    fig_isolation(PP, os.path.join(a.out, "fig_isolation.png"))

    # --- what gets printed is the diagnosis itself ---------------------------
    print("\n=== 1. what each method buys (effect size vs the pool; paired vs random) ===")
    show = prof[prof.attr.isin(["knn_dist", "own_z", "impurity", "is_transition",
                                "local_purity"])]
    piv = show.pivot(index="method", columns="attr", values="d_vs_random")
    piv = piv.reindex(_methods(None, piv.index)).round(3)
    print(piv.to_string())
    print("\n(positive = buys more isolated / more extreme-for-its-owner / more phase-shifted "
          "blocks than random on the same split; local_purity negative = buys where labels "
          "disagree with the neighbourhood)")

    mech, mech_cells = mechanism(Q, R, S, a.mech_metric)
    if len(mech):
        mech.to_csv(os.path.join(a.out, "mechanism.csv"), index=False)
        print(f"\n=== 1b. does buying such blocks COST performance? "
              f"({a.mech_metric}, de-meaned within method x split, "
              f"{int(mech.n_cells.max())} cells) ===")
        print(mech.set_index("attr").round(4).to_string())
        print("(negative r on isolation / own_z / impurity = within one method on one split, the "
              "targets whose purchases were more anomalous ended up worse)")

    print("\n=== 2. whose labels (outlier individuals) ===")
    print(pers.set_index("method").reindex(_methods(None, pers.method))
              [["rho_isolation", "rho_se", "top3_iso_share", "top3_pool_share", "top3_overbuy",
                "n_splits"]].round(3).to_string())

    print("\n=== 3. burden at the full budget ===")
    last = bcurve.sort_values("budget").groupby("method").tail(1)
    print(last.set_index("method").reindex(_methods(None, last.method))
              [["budget", "gini", "gini_se", "max_to_mean", "max_to_mean_se", "top_share",
                "n_touched", "self_share", "n_splits"]].round(3).to_string())
    print(f"\nreference: sampling proportionally to the pool gives Gini "
          f"{base['gini']:.3f} (SE {base['gini_se']:.3f}), max-to-mean "
          f"{base['max_to_mean']:.2f}, top share {base['top_share']:.3f} -- the pool's own "
          f"inequality, which no unbiased sampler removes")
    print(f"\n-> tables and figures in {a.out}")


if __name__ == "__main__":
    main()
