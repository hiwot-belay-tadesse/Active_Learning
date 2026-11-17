"""Standard BALD for the mixed-effects model, via posterior sampling from the Laplace fit.

BALD (Houlsby et al., 2011) queries the candidate maximising the mutual information between its
label and the parameters,

    I(y_c; theta) = H[ E_theta p(y_c|theta) ] - E_theta H[ p(y_c|theta) ],

i.e. "the model's members disagree about this point even though each is individually confident".
It needs a *distribution over models*, which rules out ``predict_proba`` alone -- a single
predictive probability cannot separate the two terms. scikit-activeml implements the criterion
(``skactiveml.pool.GreedyBALD``) but needs a committee: either an ensemble estimator, or a
classifier exposing a method that *samples* member predictions. This file supplies the second.

The committee is the Laplace posterior itself. ``fit_map()`` gives the mode theta^ and
``laplace_cov`` the covariance Sigma of the (beta, b) block, so

    theta_s ~ N(theta^, Sigma),    p_s(x) = sigmoid(xt(x)' theta_s),   s = 1..S

is a sample from the (approximate) posterior predictive -- a committee of S GLMMs that differ in
both their population coefficients and every participant's random effects. That is the right
committee for this model: a bootstrap or a bagged ensemble would resample rows and destroy the
very grouping structure the random effects estimate.

Two consequences worth stating, because they are what makes BALD a meaningful *baseline* here
rather than a straw man:

* It is posterior-aware, like the proposed criterion and unlike CoreSet/ProbCover, so the
  comparison is not "model-based vs geometric". BALD and TW-FEG differ in *what* they are
  informative about: BALD maximises information about the whole parameter vector, TW-FEG about
  beta under the target's density. A candidate whose label would mostly pin down a *stranger's*
  random effect is valuable to BALD and worth nothing to the target.
* Being label-free but entropy-based, it concentrates on rows where the posterior is wide, which
  in a 124-feature wearable pool means boundary and anomalous blocks. That is exactly the
  behaviour the diagnosis section attributes to the geometric methods, arrived at from the
  opposite direction -- which is why both families are benchmarked.

Candidates keep their OWNER's participant code, as everywhere else in this subdirectory.

    python two_stage/bald.py        # self-test: sampler moments, and BALD on a toy pool
"""
import os
import sys

import numpy as np
from scipy.special import expit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from laplace_variance import laplace_cov, stack_design            # noqa: E402
from utility import MixedEffectsClassifier                        # noqa: E402

__all__ = ["SampledMixedEffectsClassifier", "build_bald"]


class SampledMixedEffectsClassifier(MixedEffectsClassifier):
    """:class:`utility.MixedEffectsClassifier` plus ``sample_proba`` -- a posterior committee.

    Parameters
    ----------
    n_post : int, default=20
        Default committee size when ``sample_proba`` is called without ``n_samples``. Twenty
        draws is enough for a ranking by mutual information at this pool size; the BALD utility
        of a given candidate has a Monte-Carlo SD of order 1/sqrt(S), and the ordering is what
        matters.
    post_jitter : float, default=1e-8
        Ridge added before the Cholesky of Sigma. The Laplace covariance of a 33-row fit with
        hundreds of parameters is positive definite in theory and borderline in floating point.

    Everything else is inherited, so the estimator used for evaluation and the estimator used by
    BALD are the same model with the same priors -- no accidental difference between the arms.
    """

    def __init__(self, persons, slope_idx=(), fe_p=1.0, vcp_p=0.5, classes=None,
                 missing_label=np.nan, cost_matrix=None, random_state=None, n_post=20,
                 post_jitter=1e-8):
        super().__init__(persons=persons, slope_idx=slope_idx, fe_p=fe_p, vcp_p=vcp_p,
                         classes=classes, missing_label=missing_label, cost_matrix=cost_matrix,
                         random_state=random_state)
        self.n_post = n_post
        self.post_jitter = post_jitter

    # -- the committee -------------------------------------------------------
    def _posterior(self):
        """-> (theta_hat, L) with Sigma = L L', cached per fit."""
        if getattr(self, "_post_cache", None) is None:
            res = self.res_
            k_fep, k_vcp = res.model.k_fep, res.model.k_vcp
            p = np.asarray(res.params, float)
            theta = np.r_[p[:k_fep], p[k_fep + k_vcp:]]
            S = laplace_cov(res)
            S = 0.5 * (S + S.T) + self.post_jitter * np.eye(len(S))
            try:
                L = np.linalg.cholesky(S)
            except np.linalg.LinAlgError:
                w, V = np.linalg.eigh(S)
                L = V * np.sqrt(np.clip(w, 1e-12, None))
            self._post_cache = (theta, L)
        return self._post_cache

    def fit(self, X, y, sample_weight=None):
        self._post_cache = None
        return super().fit(X, y, sample_weight)

    def sample_proba(self, X, n_samples=None):
        """-> (n_members, n_samples_in_X, 2) class probabilities, one slice per posterior draw.

        The signature scikit-activeml's ``sample_predictions_method_name`` expects. A failed fit
        returns the labelled base rate for every member, so the BALD utility is identically zero
        and the round degenerates to random rather than raising.
        """
        X = np.asarray(X, float)
        S = int(n_samples or self.n_post)
        if self.fit_failed_ or self.res_ is None:
            p = np.full((S, len(X)), float(self.base_rate_))
            return np.stack([1 - p, p], axis=-1)
        theta, L = self._posterior()
        rng = np.random.default_rng(
            None if self.random_state is None else int(self.random_state))
        persons = list(self.persons)
        who = [persons[int(c)] for c in X[:, -1]]
        Xt = stack_design(X[:, :-1], who, self.res_)
        draws = theta[None, :] + (rng.standard_normal((S, len(theta))) @ L.T)
        p = expit(Xt @ draws.T).T                              # (S, n)
        return np.stack([1 - p, p], axis=-1)


def build_bald(seed, n_post=20, greedy=True):
    """-> a configured ``GreedyBALD`` reading the committee off ``sample_proba``.

    ``greedy=True`` is top-k within a batch (``GreedyBALD``); ``False`` gives BatchBALD's joint
    selection, which is quadratic in the batch and unnecessary at batch 3 but is what the library
    defaults to, so the choice is made explicit.
    """
    from skactiveml.pool import BatchBALD, GreedyBALD
    cls = GreedyBALD if greedy else BatchBALD
    return cls(n_MC_samples=n_post, random_state=seed,
               sample_predictions_method_name="sample_proba",
               sample_predictions_dict={"n_samples": n_post})


# --------------------------------------------------------------------------- self-test
def _self_test():
    """Two checks: the sampler's mean/spread, and that GreedyBALD accepts this classifier."""
    sys.path.insert(0, HERE)
    from utility import attach_person
    rng = np.random.default_rng(0)
    n, G, n_feat = 300, 5, 4
    g = rng.integers(0, G, n)
    Z = rng.normal(size=(n, n_feat))
    b = rng.normal(0, 0.8, G)
    eta = -0.3 + Z @ np.array([1.2, -0.7, 0.4, 0.0]) + b[g]
    y = rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(float)
    persons = [f"P{i}" for i in range(G)]
    X = attach_person(Z, [f"P{i}" for i in g], persons)

    clf = SampledMixedEffectsClassifier(persons=persons, slope_idx=[0, 1], classes=[0.0, 1.0],
                                        missing_label=np.nan, random_state=0, n_post=50)
    y_al = np.full(n, np.nan)
    lab = rng.choice(n, 40, replace=False)
    y_al[lab] = y[lab]
    clf.fit(X, y_al)
    P = clf.sample_proba(X[:20], n_samples=50)
    mean = P[:, :, 1].mean(axis=0)
    point = clf.predict_proba(X[:20])[:, 1]
    print(f"sample_proba shape {P.shape}; |mean draw - point estimate| max = "
          f"{np.abs(mean - point).max():.3f}  (Jensen gap, should be small but non-zero)")
    spread = P[:, :, 1].std(axis=0)
    print(f"member spread per row: min {spread.min():.3f}  max {spread.max():.3f}  "
          f"(zero would mean the committee is degenerate)")

    qs = build_bald(0, n_post=50)
    idx, U = qs.query(X=X, y=y_al, ensemble=clf, fit_ensemble=False, batch_size=3,
                      return_utilities=True)
    finite = np.isfinite(U[0])
    print(f"GreedyBALD picked {idx}; utilities finite on {finite.sum()} of {n} rows, "
          f"range [{np.nanmin(U[0]):.4f}, {np.nanmax(U[0]):.4f}]")

    # a failed fit must not raise and must give a flat utility surface
    bad = SampledMixedEffectsClassifier(persons=persons, slope_idx=[0], classes=[0.0, 1.0],
                                        missing_label=np.nan, random_state=0)
    y_one = np.full(n, np.nan)
    y_one[np.where(y == 0)[0][:5]] = 0.0
    bad.fit(X, y_one)
    Pb = bad.sample_proba(X[:5], n_samples=5)
    print(f"failed fit: fit_failed_={bad.fit_failed_}, sample spread = {Pb[:, :, 1].std():.1e}")

    ok = (P.shape == (50, 20, 2) and spread.max() > 1e-3 and len(set(idx)) == 3
          and bad.fit_failed_ and Pb[:, :, 1].std() < 1e-12)
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_self_test())
