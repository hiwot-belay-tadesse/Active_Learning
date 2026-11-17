"""The target's operational manifold: an estimate of p_T(x) / p_pool(x) over the pooled blocks.

Why a ratio at all
------------------
Phase 2 of the framework buys labels from *anyone*, but only wants fixed-effect information that
is useful **where the target participant actually lives**. "Where the target lives" could be read
off the target's own blocks directly, and that is one of the two options here (``"own"``), but a
single participant contributes ~200 blocks out of ~11000, so an expectation taken over them is
both noisy and supported only on points the target happens to have recorded. Re-weighting the
*whole* pool by

    r(x) = p_T(x) / p_pool(x)

keeps the expectation anchored on the target's density while evaluating it on the feature-space
region the pool can actually supply candidates from -- which is the quantity the acquisition has
to integrate against.

How it is estimated
-------------------
Density-ratio estimation by probabilistic classification (Sugiyama et al., *Density Ratio
Estimation in Machine Learning*, 2012): fit a discriminator q(x) = P(row is the target's | x) on
the target's blocks against the pool's, and read

    r(x) ∝ q(x) / (1 - q(x)).

Three decisions that matter more than the choice of classifier:

* **Class weights, not a prior correction.** With ``class_weight="balanced"`` the discriminator
  is fitted as if both samples were equally likely, so the odds *are* the density ratio up to a
  constant. The constant is irrelevant: the weights are normalised before use.
* **Out-of-fold probabilities.** 124 features against ~200 target rows separates almost
  perfectly in-sample, which would make r(x) a 0/1 indicator of "is this literally one of the
  target's rows" -- the opposite of a smooth manifold weight. ``cross_val_predict`` fits on
  folds and predicts out of fold, so every pool row gets a probability from a model that did not
  see it. Strong L2 (``C=0.01``) does the rest.
* **Clipping before normalising.** Even out of fold, the top of the ratio distribution is heavy;
  one row with r = 400 would turn a weighted expectation into a single-point evaluation. Weights
  are clipped at a high quantile (default 99%) and renormalised, which costs a little fidelity
  in the extreme tail and buys a well-conditioned weighted second-moment matrix.

Label-free, and that is the point: the discriminator uses participant identity and features
only, never ``y``. Nothing here peeks at a label the active learner has not bought.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

__all__ = ["target_ratio", "eval_weights", "second_moment"]


def target_ratio(Z, owner, target, C=0.01, cv=3, seed=0, max_pool=None):
    """-> r, an array of length len(Z) with the estimated p_T(x)/p_pool(x) up to a constant.

    Parameters
    ----------
    Z : (n, p) standardised pooled features, WITHOUT the person column.
    owner : (n,) whose block each row is.
    target : the participant whose density is the numerator.
    C, cv, seed : discriminator regularisation, folds for the out-of-fold probabilities, RNG.
    max_pool : subsample this many non-target rows before fitting (None = all). Only a speed
        knob; the ratio is a function of x, so a subsample changes the estimate's variance and
        not what it estimates.
    """
    Z = np.asarray(Z, float)
    lab = (np.asarray(owner) == target).astype(int)
    if lab.sum() == 0:
        raise ValueError(f"target {target!r} owns none of the rows")
    idx = np.arange(len(Z))
    if max_pool is not None and (len(Z) - lab.sum()) > max_pool:
        rng = np.random.default_rng(seed)
        others = idx[lab == 0]
        keep = np.r_[idx[lab == 1], rng.choice(others, max_pool, replace=False)]
        keep.sort()
    else:
        keep = idx
    clf = LogisticRegression(C=C, max_iter=300, class_weight="balanced")
    # cross_val_predict with stratified folds: the target class is ~2% of the rows, so an
    # unstratified fold can miss it entirely and return a constant probability.
    q = np.full(len(Z), np.nan)
    q_keep = cross_val_predict(clf, Z[keep], lab[keep], cv=cv, method="predict_proba")[:, 1]
    q[keep] = q_keep
    if np.isnan(q).any():                      # rows held out of the fit: score with a full fit
        clf.fit(Z[keep], lab[keep])
        miss = np.isnan(q)
        q[miss] = clf.predict_proba(Z[miss])[:, 1]
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return q / (1.0 - q)


def eval_weights(Z, owner, target, measure="ratio", clip_q=0.99, **kw):
    """-> weights pi (length len(Z), summing to 1) defining the evaluation measure of Phase 2.

    measure
        ``"ratio"``  pi ∝ clipped density ratio over the whole pool -- the target's operational
        manifold as seen through every participant's blocks. The framework's default.
        ``"own"``    pi uniform on the target's own rows, zero elsewhere. The ablation that
        answers "is the density re-weighting doing anything, or would the target's own blocks
        have been enough?"
        ``"pool"``   pi uniform over the whole pool: the un-personalised A-optimal criterion,
        included so that "target-weighted" can be shown to be the part that matters.
    """
    n = len(Z)
    if measure == "own":
        w = (np.asarray(owner) == target).astype(float)
    elif measure == "pool":
        w = np.ones(n)
    elif measure == "ratio":
        r = target_ratio(Z, owner, target, **kw)
        hi = np.quantile(r, clip_q)
        w = np.clip(r, 0.0, max(hi, 1e-12))
    else:
        raise ValueError(f"measure must be 'ratio', 'own' or 'pool', got {measure!r}")
    s = w.sum()
    if s <= 0:
        raise ValueError("evaluation weights are all zero")
    return w / s


def second_moment(Z, pi):
    """-> A_T = sum_e pi_e xt_e xt_e' over the FIXED-EFFECT design [1, x], shape (1+p, 1+p).

    This is the only thing Phase 2 needs from the evaluation measure, it does not depend on the
    round, and it is what makes the criterion computable over the whole pool: once A_T is formed
    (and factored), each round costs a few matrix products rather than a pass over the evaluation
    set per candidate.
    """
    Z = np.asarray(Z, float)
    Xf = np.column_stack([np.ones(len(Z)), Z])
    return Xf.T @ (np.asarray(pi, float)[:, None] * Xf)


def _self_test():
    """Recover a known ratio: two Gaussians whose log-ratio is linear in x.

    p_T = N(mu, I), p_pool = N(0, I) gives log r(x) = mu'x - |mu|^2/2, so the estimated log-ratio
    should correlate near-perfectly with mu'x. This checks the plumbing (orientation, clipping,
    normalisation), not the quality of the estimator in 124 dimensions.
    """
    rng = np.random.default_rng(0)
    p, n_t, n_p = 8, 300, 6000
    mu = np.r_[1.0, -0.8, np.zeros(p - 2)]
    Zt = rng.normal(size=(n_t, p)) + mu
    Zp = rng.normal(size=(n_p, p))
    Z = np.vstack([Zt, Zp])
    owner = np.array(["T"] * n_t + ["P"] * n_p)
    r = target_ratio(Z, owner, "T", seed=0)
    true_log = Z @ mu - 0.5 * mu @ mu
    c = np.corrcoef(np.log(r), true_log)[0, 1]
    pi = eval_weights(Z, owner, "T")
    A = second_moment(Z, pi)
    # the weighted mean of x under pi should sit near mu, not near 0
    mean_w = A[0, 1:]
    print(f"corr(log r_hat, log r_true) = {c:.3f}   (want > 0.9)")
    print(f"pi-weighted mean of x[:2]   = {np.round(mean_w[:2], 2)}   (want ~ {mu[:2]})")
    print(f"pool mean of x[:2]          = {np.round(Z[:, :2].mean(0), 2)}")
    ok = c > 0.9 and mean_w[0] > 0.5 and mean_w[1] < -0.3
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
