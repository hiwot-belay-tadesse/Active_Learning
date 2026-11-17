"""Acquisition curves under cohort vs self seeding, mean +- SE over targets x splits.

    python plots/cohort_curves.py                  # one figure per metric (auroc, auprc)
    python plots/cohort_curves.py --no_siblings    # only the folders named below

Every comparable run on disk is merged, so a newly added acquisition appears here without the
script being edited: folders under results/<target>/ are grouped by HOW THEY WERE SEEDED and
gated on the same keys plots/per_person_grid.py uses, then all their methods are drawn in one
panel. Arms are restricted to the splits they SHARE -- a gap averaged over different splits per
arm is not paired, and the pairing is the whole point -- and the intersection is printed.

PR-AUC is read against the positive rate, not against 0.5 -- a constant predictor scores the
target's own prevalence. The per-round CSV carries n_pos_test but not the test row count, so
that baseline cannot be drawn from saved output and is left off; the pooled training rate is
about 16%.
"""
import argparse, glob, json, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys                                                               # noqa: E402
sys.path.insert(0, REPO)
from paths import run_folders                                            # noqa: E402

# Fixed colours for the arms seen so far, so a figure keeps its colour coding between runs; any
# method not listed is assigned from the tail of the palette in sorted order.
COLOR = {"coreset": "#1f77b4", "probcover": "#2ca02c", "typiclust": "#9467bd",
         "random": "#d62728", "laplace": "#000000", "laplace_pool": "#17becf",
         "laplace_people": "#8c564b", "laplace_var": "#7f7f7f", "eer": "#e377c2"}
SPARE = ["#bcbd22", "#393b79", "#637939", "#8c6d31", "#843c39"]
LABEL = {"auroc": "AUROC", "auprc": "PR-AUC"}
# Must match plots/per_person_grid.py: everything that has to agree before two folders may be
# drawn in the same panel. `seed_source` is absent in runs written before the option existed,
# which were self-seeded by construction.
SAME = ["target", "block", "n_slopes", "fe_p", "vcp_p", "n_seed", "n_seed_pos",
        "rounds", "batch", "seed"]


def seeding(c):
    src = c.get("seed_source", "self")
    return (src, c.get("n_seed_per_person") if src != "self" else None)


def describe(key):
    src, n = key
    return f"cohort seed ({n}/participant)" if src != "self" else "self seed (own labels only)"


def load(folder):
    parts = sorted(glob.glob(os.path.join(folder, "per_round_split*.csv")))
    return pd.concat([pd.read_csv(f) for f in parts]) if parts else None


def resolve(target, name):
    """-> the absolute path of the run folder called `name`, whichever depth it sits at.

    Joining <results>/<target>/<name> stopped being right once runs moved into a scope folder,
    and an unresolvable name would silently become a path that does not exist -- so fall back
    to that join only when nothing matches, where the caller's own error says so.
    """
    hits = run_folders(target, name)
    return hits[0] if hits else os.path.join(REPO, "results", target, name)


def collect(target, no_siblings, pinned):
    """-> {seeding key: (frame, [folder names])}, arms restricted to their shared splits.

    Runs live either directly under the target or inside a scope folder (global / personal);
    run_folders finds both, so a folder written before scopes existed is still discovered.
    """
    folders = pinned if no_siblings else run_folders(target)
    groups = {}
    for d in folders:
        if not os.path.isdir(d):
            continue
        try:
            cfg = json.load(open(os.path.join(d, "config.json")))
        except Exception as e:                      # no config, or the malformed one on disk
            print(f"  skipped {os.path.basename(d)}: config unreadable ({type(e).__name__})")
            continue
        D = load(d)
        if D is None or not len(D):
            continue
        groups.setdefault(seeding(cfg), []).append((os.path.basename(d), cfg, D))

    out = {}
    for key, runs in groups.items():
        ref = runs[0][1]
        keep = [(n, D) for n, c, D in runs
                if not [k for k in SAME if c.get(k) != ref.get(k)]]
        for n, c, _ in runs:
            bad = [k for k in SAME if c.get(k) != ref.get(k)]
            if bad:
                print(f"  skipped {n}: differs from {runs[0][0]} on {bad}")
        shared = set.intersection(*[set(D.split) for _, D in keep])
        if not shared:
            print(f"  {describe(key)}: folders share no split; using the largest alone")
            keep = [max(keep, key=lambda t: t[1].split.nunique())]
            shared = set(keep[0][1].split)
        frames, names = [], []
        for n, D in keep:
            dropped = sorted(set(D.split) - shared)
            D = D[D.split.isin(shared)]
            if not len(D):
                continue
            # first folder wins a method present in two, so an arm is never double-counted
            D = D[~D.method.isin({m for f in frames for m in set(f.method)})]
            if len(D):
                frames.append(D)
                names.append(n + (f" (-{len(dropped)} split(s))" if dropped else ""))
        out[key] = (pd.concat(frames, ignore_index=True), names)
        print(f"  {describe(key)}: {len(shared)} shared split(s) {sorted(shared)}, "
              f"methods {sorted(set(out[key][0].method))}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="computer_work")
    ap.add_argument("--out_run", default="computer_work_w15_coreset-probcover-random-typiclust"
                                        "_r10x3_s5_seed_cohort",
                    help="folder whose figures/ dir the PNGs are written to")
    ap.add_argument("--no_siblings", action="store_true",
                    help="plot only --out_run and the self-seeded run it was compared with")
    a = ap.parse_args()

    base = os.path.join(REPO, "results", a.target)
    pinned = [resolve(a.target, a.out_run),
              resolve(a.target, "computer_work_w15_coreset-probcover-random_r10x3_s5")]
    groups = collect(a.target, a.no_siblings, pinned)
    if not groups:
        raise SystemExit(f"no runs with per-round output under {base}")
    # cohort panel first, self second, so the figure reads left-to-right as "more seed labels"
    order = sorted(groups, key=lambda k: (k[0] == "self", str(k)))
    methods = sorted({m for frame, _ in groups.values() for m in set(frame.method)})
    colors = dict(COLOR)
    for i, m in enumerate(m for m in methods if m not in colors):
        colors[m] = SPARE[i % len(SPARE)]

    for metric in ("auroc", "auprc"):
        fig, axes = plt.subplots(1, len(order), figsize=(5.3 * len(order), 4.3), sharey=True,
                                 squeeze=False)
        for ax, key in zip(axes[0], order):
            d, names = groups[key]
            for m in methods:
                g = d[d.method == m].groupby("acquired")[metric].agg(["mean", "sem"])
                if not len(g):
                    continue
                ax.plot(g.index, g["mean"], color=colors[m], lw=2, label=m)
                ax.fill_between(g.index, g["mean"] - g["sem"], g["mean"] + g["sem"],
                                color=colors[m], alpha=0.15, lw=0)
            ax.set_title(f"{describe(key)}\n{d.split.nunique()} split(s), "
                         f"{d.target.nunique()} targets", fontsize=10)
            ax.set_xlabel("labels acquired")
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel(f"{LABEL[metric]} on the target's held-out days")
        axes[0][0].legend(frameon=False, fontsize=9)
        fig.suptitle(f"Personalised AL on {a.target.upper()}: acquisitions x seeding",
                     fontsize=11)
        fig.tight_layout()
        out = os.path.join(resolve(a.target, a.out_run), "figures",
                           f"repr_methods_curves_{metric}.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("->", out)


if __name__ == "__main__":
    main()
