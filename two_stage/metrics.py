"""The two evaluation axes: predictive quality on the target, and burden equity across the cohort.

Everything here is deliberately small and dependency-light, because both axes are reported in
several places (the AL loop writes them per round, the diagnosis recomputes them on finished
runs, the analysis aggregates them) and they must mean exactly the same thing in all three.

Axis 1 -- threshold-free predictive performance on the target participant
-------------------------------------------------------------------------
``macro_prauc`` is the average of the area under the precision-recall curve computed **once per
class**: the usual average precision for the positive class, and the average precision of the
complementary problem (predict "not COMPUTER_WORK" from 1 - p) for the negative class. At a
~16% positive rate the positive-class PR-AUC alone can be moved a long way by a handful of test
blocks, while the negative-class curve is nearly saturated for any sane model; averaging them is
the standard macro reading and is what keeps a method that buys nothing but easy negatives from
looking good. The single-class case returns NaN rather than 0.5, so a degenerate round is
dropped from an average instead of dragging it down.

``macro_rocauc`` is reported for completeness and is *numerically identical* to plain AUROC in
the binary case -- ROC-AUC(y, p) == ROC-AUC(1 - y, 1 - p) exactly -- so the macro label is about
reporting symmetry, not about a different number. It is computed the long way anyway, because a
future multi-class target would need it and a silently-identical column is better than a column
that quietly changes meaning.

Axis 2 -- participant burden equity
-----------------------------------
A personalised run buys labels from whoever the acquisition likes, so the cohort-level question
is how the total labelling work lands on real people. ``gini`` is the Gini coefficient of the
per-participant query counts **over the fixed cohort, zeros included**: 0 when every participant
is asked equally often, approaching 1 when one person answers everything. Dropping the zeros
would be the single easiest way to make an inequitable method look equitable, so ``counts_over``
takes the cohort explicitly and fills in the people a method never touched.

``max_to_mean`` is the complementary extreme-value reading: the worst-off participant's load
divided by the mean load. Gini can stay moderate while one individual carries 15x the average,
which is precisely the failure mode a wearable study cares about (that person drops out).
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

__all__ = ["macro_prauc", "macro_rocauc", "target_scores", "gini", "max_to_mean",
           "counts_over", "burden"]


# --------------------------------------------------------------------------- axis 1
def _usable(y):
    y = np.asarray(y)
    return 0 < int(np.sum(y == 1)) < len(y)


def macro_prauc(y, p):
    """Mean of the per-class average precision. NaN if the test rows are single-class.

    ``p`` is the predicted probability of the positive class; the negative-class curve is
    computed on (1 - y, 1 - p), which is the same ranking read from the other end.
    """
    if not _usable(y):
        return np.nan
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean([average_precision_score(y, p),
                          average_precision_score(1 - y, 1 - p)]))


def macro_rocauc(y, s):
    """Macro-averaged ROC-AUC. Equals plain AUROC in the binary case -- see the module docstring."""
    if not _usable(y):
        return np.nan
    y, s = np.asarray(y, float), np.asarray(s, float)
    return float(np.mean([roc_auc_score(y, s), roc_auc_score(1 - y, -s)]))


def target_scores(y, logit):
    """-> dict of every axis-1 number, from the linear predictor on the target's held-out rows.

    AUROC and AUPRC keep the names the parent repo's CSVs already use, so a run from this
    subdirectory can be read by the existing plotting scripts; the macro columns are additions.
    """
    from scipy.special import expit
    if not _usable(y):
        return dict(auroc=np.nan, auprc=np.nan, macro_prauc=np.nan, macro_rocauc=np.nan)
    p = np.clip(expit(np.asarray(logit, float)), 1e-9, 1 - 1e-9)
    return dict(auroc=float(roc_auc_score(y, logit)),
                auprc=float(average_precision_score(y, p)),
                macro_prauc=macro_prauc(y, p),
                macro_rocauc=macro_rocauc(y, logit))


# --------------------------------------------------------------------------- axis 2
def counts_over(cohort, owners):
    """-> query counts per participant, in ``cohort`` order, zeros included.

    ``owners`` is the flat list of whose block each purchased label came from. The cohort has to
    be passed in: a method that only ever asks three people would otherwise be scored on a
    three-person cohort and come out perfectly equitable.
    """
    c = {u: 0 for u in cohort}
    for o in owners:
        if o in c:
            c[o] += 1
        else:                       # an owner outside the stated cohort is a bug, not a zero
            raise KeyError(f"query owner {o!r} is not in the cohort")
    return np.array([c[u] for u in cohort], float)


def gini(x):
    """Gini coefficient of a non-negative vector. 0 = perfectly equal, -> 1 = one person pays all.

    Computed as the mean absolute difference over twice the mean (the relative mean absolute
    difference), which needs no sorting assumption and handles the all-zero vector by returning
    NaN rather than dividing by zero.
    """
    x = np.asarray(x, float)
    if np.any(x < 0):
        raise ValueError("gini is defined for non-negative quantities")
    n, s = len(x), x.sum()
    if n == 0 or s <= 0:
        return np.nan
    xs = np.sort(x)
    # sum_i (2i - n - 1) x_i / (n^2 mean) -- the sorted form, O(n log n)
    i = np.arange(1, n + 1)
    return float((np.sum((2 * i - n - 1) * xs)) / (n * s))


def max_to_mean(x):
    """Heaviest participant's load / mean load. 1.0 when the load is flat, n when one pays all."""
    x = np.asarray(x, float)
    m = x.mean()
    return float(x.max() / m) if m > 0 else np.nan


def burden(cohort, owners):
    """-> dict(gini, max_to_mean, n_queries, n_touched, top_share) for one set of queries."""
    c = counts_over(cohort, owners)
    tot = c.sum()
    return dict(gini=gini(c), max_to_mean=max_to_mean(c), n_queries=int(tot),
                n_touched=int((c > 0).sum()),
                top_share=float(c.max() / tot) if tot > 0 else np.nan)


if __name__ == "__main__":
    # the three properties worth asserting out loud
    assert abs(gini(np.ones(10))) < 1e-12, "flat load must have Gini 0"
    one = np.zeros(10); one[0] = 10
    assert gini(one) > 0.89, "one-person load must have Gini ~ (n-1)/n"
    assert abs(max_to_mean(one) - 10) < 1e-12
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.16, 300).astype(float)
    s = rng.normal(size=300) + 2 * y
    print("gini(flat)=0  gini(one)=%.3f  max/mean(one)=%.1f" % (gini(one), max_to_mean(one)))
    print("macro_rocauc == auroc:", abs(macro_rocauc(y, s) - roc_auc_score(y, s)) < 1e-12)
    print("scores:", target_scores(y, s))
