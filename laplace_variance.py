"""Variance-reduction acquisition from the mixed model's Laplace posterior.

A scikit-activeml strategy that is *aware of the estimator's posterior*, which none of the
library's own strategies can be for this model: ``UncertaintySampling`` sees only
``predict_proba``, and ``EpistemicUncertaintySampling`` -- the one that makes the right
distinction -- is hardwired to ``ParzenWindowClassifier`` or a wrapped
``sklearn.linear_model.LogisticRegression`` by ``isinstance`` checks, so it cannot accept
:class:`utility.MixedEffectsClassifier` at all.

The criterion
-------------
``statsmodels``' ``fit_map()`` is a Laplace fit: it finds the posterior mode and inverts the
Hessian of the log posterior there, giving a Gaussian approximation to p(beta, b | D) with

    Sigma = (Xt' W Xt + Lam)^-1,
    Xt = [X, Z] (fixed and random effect designs stacked),  W = diag(mu (1 - mu)),
    Lam = blockdiag(I / fe_p^2, diag(1 / s^2)),  s = exp(vcp) the random-effect SDs.

Under that Gaussian the linear predictor at a row xt has variance xt' Sigma xt -- the EPISTEMIC
(reducible) uncertainty, as against the aleatoric mu(1 - mu) which no label can shrink.

Labelling a candidate c adds a rank-one term w_c xt_c xt_c' to the Hessian, so by
Sherman-Morrison the variance summed over an evaluation set {xt_e} falls by

    Delta(c) = w_c * sum_e (xt_e' Sigma xt_c)^2 / (1 + w_c * xt_c' Sigma xt_c),   w_c = mu_c(1-mu_c)

and we query the candidate maximising it. Three consequences worth knowing:

* **No label simulation.** Delta(c) does not contain y_c; w_c uses only the current fitted mu_c.
  So unlike expected error reduction there is no expectation over the two possible labels and no
  refit per candidate -- one Sigma per round, then matrix products. It scores the whole pool
  where ``MonteCarloEER`` needed ``SubSamplingWrapper(max_candidates=50)``.
* **Provenance falls out of the formula.** xt_e carries the target's random-effect columns and
  xt_c the candidate's; the cross term is non-zero through the shared fixed-effect block, so the
  criterion says on its own that a stranger's label helps the target via the population
  parameters while the target's own label helps via both. See `Candidate coding` below -- this
  only holds if you hand it the true owner codes.
* **It is A-optimality on the linear predictor, not on AUROC.** Shrinking var(eta) is a proxy
  for better ranking, not the same thing. Being label-free it also will not preferentially hunt
  positives, which at ~16% prevalence is a real behavioural difference from
  ``UncertaintySampling(method="expected_average_precision")``. Which way that cuts is an
  empirical question.

Candidate coding
----------------
The participant code rides in the LAST column of ``X`` (``utility.attach_person``), and this
strategy reads it. That makes the coding you pass a modelling decision, not a detail:

* ``owner`` coding (each block coded to whoever actually recorded it) gives the criterion above,
  and is what makes the provenance term meaningful. **Recommended for this strategy.**
* ``target`` coding (every candidate coded as the target, which is what ``main.py`` currently
  does for ``CLF_STRATEGIES`` because expected error reduction wants "how much would this label
  help ME") makes every candidate's random-effect block identical to the target's. The
  criterion still works and is still coherent -- "which feature vector, labelled as if it were
  mine, most reduces my predictive variance" -- but the provenance term is gone.

Nothing here detects which you passed, so pass the one you mean.

Known approximations
--------------------
* ``Sigma`` is the covariance of the (fixed, random) block **conditional on the variance
  components at the mode**. statsmodels' parameter vector also holds the ``vcp`` log-SDs, and
  the marginal block differs from the conditional one; the formula ignores uncertainty in the
  random-effect SDs. See ``--self-test`` for the size of that gap.
* A candidate owned by someone with no labels yet has NO random-effect columns in the current
  fit, so its random block is all zeros and it is ranked on the fixed-effect term alone. Buying
  that label would in truth ADD columns, and a rank-one update cannot represent a growing
  parameter space. The effect is conservative (it understates a first label from a stranger) and
  it disappears as the labelled set covers more people. At round 1 only the target has labels,
  so every other candidate is ranked purely on the fixed-effect cross term.
* ``W`` is evaluated at the current mode. Within a batch the rank-one update is applied
  sequentially (greedy), but the mode itself is not refitted until the round ends.

Classifier contract
-------------------
Deliberately does not import from ``utility`` -- it needs only these public attributes of a
fitted classifier, so any estimator providing them works:
``persons``, ``slope_idx``, ``fe_p``, ``res_`` (the statsmodels ``BayesMixedGLMResults``),
``fit_failed_``. When the fit failed (a single-class labelled set, which happens at round 0 for
a target whose seed set found no positive) there is no posterior and utilities fall back to
uniform, i.e. the round degenerates to random rather than raising.

Usage
-----
    from laplace_variance import LaplaceVarianceReduction
    qs = LaplaceVarianceReduction(random_state=0)
    idx = qs.query(X=Xq, y=y_al, clf=clf, X_eval=Xq[own_unlabelled], batch_size=3)

To wire into ``main.py``: one branch in ``build_strategy()`` and the name added to
``CLF_STRATEGIES``. Note it takes no ``fit_clf``/``ignore_partial_fit`` -- it refits once.

    python laplace_variance.py --self-test
"""
import numpy as np

from skactiveml.base import SingleAnnotatorPoolQueryStrategy
from skactiveml.utils import MISSING_LABEL, is_labeled, simple_batch

__all__ = ["LaplaceVarianceReduction", "laplace_cov", "vc_spec", "stack_design"]


# --------------------------------------------------------------------------- reading the fit
def vc_spec(res):
    """-> [(component, person)] for the random-effect columns of ``res``, in column order.

    statsmodels names these after the factor expression, not the ``vc_formulas`` key:
    ``C(person)[M4]`` for an intercept, ``C(person)[M4]:f11`` for a slope on feature 11. The
    names line up with the columns of ``model.exog_vc`` and with the ``vc`` block of the
    parameter vector. Parsing the dict key instead silently yields nothing, which looks like a
    model with no random effects.
    """
    spec = []
    for nm in list(getattr(res.model, "vc_names", []) or []):
        i, j = nm.find("["), nm.find("]")
        person = nm[i + 1:j] if i >= 0 else nm
        tail = nm[j + 1:]
        comp = tail[1:] if tail.startswith(":") else "intercept"
        spec.append((comp, person))
    return spec


def _blocks(res):
    """Parameter-vector layout and the pieces Lam is built from."""
    m = res.model
    k_fep, k_vcp, k_vc = m.k_fep, m.k_vcp, m.k_vc
    p = np.asarray(res.params, float)
    vcp = p[k_fep:k_fep + k_vcp]
    s = np.exp(vcp[m.ident]) if k_vc else np.empty(0)      # random-effect SDs, per column
    return k_fep, k_vcp, k_vc, s


def _dense(a):
    return np.asarray(a.todense(), float) if hasattr(a, "todense") else np.asarray(a, float)


def laplace_cov(res, method="hessian", jitter=0.0):
    """-> Sigma, the posterior covariance of the (fixed, random) block, shape (d, d).

    ``d = k_fep + k_vc`` and the ordering is [Intercept, f0..fn, random-effect columns], i.e.
    exactly what :func:`stack_design` builds a row to match.

    method
        ``"hessian"``  invert ``res.cov_params()``, take the (fep, vc) sub-block of the implied
        Hessian and invert that -- the conditional covariance, dropping the ``vcp`` rows.
        ``"analytic"`` rebuild ``(Xt' W Xt + Lam)^-1`` from the model's own design matrices.
        They agree to ~1e-8 (``--self-test``); "hessian" is the default because it does not
        re-derive W, "analytic" is there to check it.
    """
    k_fep, k_vcp, k_vc, s = _blocks(res)
    keep = np.r_[np.arange(k_fep), np.arange(k_fep + k_vcp, k_fep + k_vcp + k_vc)]

    if method == "hessian":
        C = np.asarray(res.cov_params(), float)
        C = 0.5 * (C + C.T)                                 # numerical Hessian is not exactly symmetric
        H = _inv(C)
        A = H[np.ix_(keep, keep)]
    elif method == "analytic":
        m = res.model
        p = np.asarray(res.params, float)
        fep, vc = p[:k_fep], p[k_fep + k_vcp:]
        eta = m.exog @ fep
        if k_vc:
            eta = eta + np.asarray(m.exog_vc.dot(vc), float).ravel()
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = mu * (1.0 - mu)
        Xt = np.hstack([np.asarray(m.exog, float), _dense(m.exog_vc)]) if k_vc \
            else np.asarray(m.exog, float)
        lam = np.r_[np.ones(k_fep) / m.fe_p ** 2, 1.0 / s ** 2]
        A = Xt.T @ (w[:, None] * Xt) + np.diag(lam)
    else:
        raise ValueError(f"method must be 'hessian' or 'analytic', got {method!r}")

    A = 0.5 * (A + A.T)
    if jitter:
        A = A + jitter * np.eye(len(A))
    return _inv(A)


def _inv(A):
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)


def stack_design(Z, persons, res):
    """-> Xt, the stacked [1, features, random-effect] design for arbitrary rows.

    ``Z`` is (n, n_feat) features WITHOUT the person column; ``persons`` is the length-n list of
    participant ids those rows belong to. Columns match :func:`laplace_cov`'s ordering.

    A row whose participant has no random-effect columns in this fit (nobody has bought them a
    label yet) gets zeros across the whole random block -- see `Known approximations`.
    """
    Z = np.asarray(Z, float)
    n = len(Z)
    k_fep = res.model.k_fep
    if k_fep != Z.shape[1] + 1:
        raise ValueError(f"fit has {k_fep} fixed effects, features are {Z.shape[1]} wide; "
                         f"expected Intercept + one per feature")
    spec = vc_spec(res)
    out = np.zeros((n, k_fep + len(spec)))
    out[:, 0] = 1.0
    out[:, 1:k_fep] = Z
    if spec:
        who = np.asarray(persons, dtype=object)
        for j, (comp, person) in enumerate(spec):
            m = who == person
            if not m.any():
                continue
            out[m, k_fep + j] = 1.0 if comp == "intercept" else Z[m, int(comp[1:])]
    return out


# --------------------------------------------------------------------------- the strategy
class LaplaceVarianceReduction(SingleAnnotatorPoolQueryStrategy):
    """Query the candidate whose label most shrinks posterior predictive variance on ``X_eval``.

    Parameters
    ----------
    criterion : {"reduction", "variance"}, default="reduction"
        ``"reduction"`` is the Sherman-Morrison criterion in the module docstring: how much the
        label shrinks variance on the evaluation set. ``"variance"`` is the ablation -- the
        candidate's OWN epistemic variance ``xt' Sigma xt``, ignoring ``X_eval`` entirely. The
        ablation is included because it is the naive reading of "most uncertain under the
        posterior" and it should be worse: it buys extrapolation points in sparse regions of
        feature space that teach the target's prediction nothing. Worth running once to confirm
        that the evaluation set is doing the work.
    cov_method : {"hessian", "analytic"}, default="hessian"
        Passed to :func:`laplace_cov`.
    jitter : float, default=0.0
        Ridge added to the Hessian before inversion, for an ill-conditioned early-round fit.
    missing_label, random_state
        Standard scikit-activeml arguments. ``random_state`` breaks ties and drives the uniform
        fallback when the model could not be fitted.
    """

    def __init__(self, criterion="reduction", cov_method="hessian", jitter=0.0,
                 missing_label=MISSING_LABEL, random_state=None):
        super().__init__(missing_label=missing_label, random_state=random_state)
        self.criterion = criterion
        self.cov_method = cov_method
        self.jitter = jitter

    def query(self, X, y, clf, fit_clf=True, candidates=None, X_eval=None,
              batch_size=1, return_utilities=False):
        """Query ``batch_size`` rows.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features + 1)
            The pool, participant code in the LAST column. See `Candidate coding`.
        y : array-like, shape (n_samples,)
            Labels, ``missing_label`` where unlabelled.
        clf : fitted or unfitted estimator meeting the contract in the module docstring.
        fit_clf : bool, default=True
            Fit ``clf`` on ``(X, y)`` first. ``False`` uses the posterior already on ``clf``.
        X_eval : array-like, shape (n_eval, n_features + 1), optional
            Rows whose predictive variance is to be reduced -- for a personalised run, the
            target's own unlabelled blocks. Defaults to the unlabelled candidates, which makes
            the criterion pool-average rather than target-specific.
        batch_size : int, default=1
            Picked greedily, re-applying the rank-one update after each pick.

        Returns
        -------
        query_indices : ndarray, shape (batch_size,)
        utilities : ndarray, shape (batch_size, n_samples)
            Only if ``return_utilities``. Row t holds Delta at pick t; already-picked and
            non-candidate entries are nan.
        """
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True)
        X_cand, mapping = self._transform_candidates(candidates, X, y)
        if self.criterion not in ("reduction", "variance"):
            raise ValueError(f"criterion must be 'reduction' or 'variance', "
                             f"got {self.criterion!r}")

        if fit_clf:
            clf = clf.fit(X, y)

        n_total = len(X) if mapping is not None else len(X_cand)
        res = getattr(clf, "res_", None)
        if getattr(clf, "fit_failed_", False) or res is None:
            # no posterior to reduce the variance of: fall back to uniform, so the round is
            # random rather than an exception. Callers should count these.
            u = np.zeros(n_total) if mapping is None else self._scatter(
                np.zeros(len(X_cand)), mapping, len(X))
            return simple_batch(u, self.random_state_, batch_size=batch_size,
                                return_utilities=return_utilities)

        Sigma = laplace_cov(res, method=self.cov_method, jitter=self.jitter)
        Xt_cand = self._design(X_cand, clf, res)
        w = self._weights(X_cand, clf)

        if self.criterion == "variance":
            d = np.einsum("ij,jk,ik->i", Xt_cand, Sigma, Xt_cand)
            u = d if mapping is None else self._scatter(d, mapping, len(X))
            return simple_batch(u, self.random_state_, batch_size=batch_size,
                                return_utilities=return_utilities)

        if X_eval is None:
            lab = is_labeled(y, missing_label=self.missing_label_)
            X_eval = X[~lab] if mapping is not None else X_cand
        Xt_eval = self._design(np.asarray(X_eval, float), clf, res)

        picked, util_rows = [], []
        S = Sigma
        for _ in range(batch_size):
            SC = S @ Xt_cand.T                                    # (d, n_cand)
            quad = np.einsum("ij,ji->i", Xt_cand, SC)             # xt_c' S xt_c
            cross = ((Xt_eval @ SC) ** 2).sum(axis=0)             # sum_e (xt_e' S xt_c)^2
            delta = w * cross / (1.0 + w * quad)
            delta[picked] = -np.inf                               # no repeats within a batch
            row = np.where(np.isfinite(delta), delta, np.nan)
            util_rows.append(row if mapping is None else self._scatter(row, mapping, len(X)))
            b = int(np.nanargmax(np.where(np.isfinite(delta), delta, np.nan)))
            picked.append(b)
            sc = SC[:, b]
            S = S - np.outer(sc, sc) * (w[b] / (1.0 + w[b] * quad[b]))   # Sherman-Morrison

        idx = np.asarray(picked if mapping is None else mapping[picked], dtype=int)
        if return_utilities:
            return idx, np.vstack(util_rows)
        return idx

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _scatter(v, mapping, n):
        out = np.full(n, np.nan)
        out[mapping] = v
        return out

    @staticmethod
    def _design(Xp, clf, res):
        """Split the person column off and stack the full design."""
        Xp = np.asarray(Xp, float)
        persons = list(clf.persons)
        who = [persons[int(c)] for c in Xp[:, -1]]
        return stack_design(Xp[:, :-1], who, res)

    @staticmethod
    def _weights(Xp, clf):
        """w_c = mu_c (1 - mu_c) at the current mode."""
        p = np.asarray(clf.predict_proba(Xp), float)[:, 1]
        return np.clip(p * (1.0 - p), 1e-12, None)


# --------------------------------------------------------------------------- self-test
def _self_test():
    """Check the algebra, not the approximation.

    Three things are verifiable without an oracle, and all three are things that can break
    silently while still producing plausible-looking rankings:

      1. ``laplace_cov(method="analytic")`` == ``method="hessian"``, i.e. Sigma really is
         (Xt' W Xt + Lam)^-1 as claimed, and our Lam matches statsmodels' priors.
      2. the Sherman-Morrison step == an explicit re-inversion of (H + w xt xt').
      3. Delta(c) == the drop in summed evaluation variance that step actually produces.

    What this does NOT check is whether the Laplace Gaussian is a good approximation to the true
    posterior, or whether reducing var(eta) improves AUROC. Neither is checkable here.
    """
    import pandas as pd
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    rng = np.random.default_rng(0)
    n, G, n_feat = 500, 6, 4
    g = rng.integers(0, G, n)
    Z = rng.normal(size=(n, n_feat))
    b = rng.normal(0, 0.8, G)
    eta = -0.3 + Z @ np.array([1.2, -0.7, 0.4, 0.0]) + b[g]
    y = rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(float)
    people = [f"P{i}" for i in g]

    df = pd.DataFrame(Z, columns=[f"f{i}" for i in range(n_feat)])
    df["y"], df["person"] = y, people
    vc = {"person": "0 + C(person)", "slope_f0": "0 + C(person):f0"}
    res = BinomialBayesMixedGLM.from_formula(
        "y ~ " + " + ".join(f"f{i}" for i in range(n_feat)), vc, df,
        vcp_p=0.5, fe_p=1.0).fit_map()

    k_fep, k_vcp, k_vc, _ = _blocks(res)
    print(f"fit: {k_fep} fixed + {k_vcp} vcp + {k_vc} random = {len(res.params)} params")
    print(f"     {len(vc_spec(res))} random-effect columns, e.g. {vc_spec(res)[:2]}")

    # 1. the two routes to Sigma
    Sh = laplace_cov(res, method="hessian")
    Sa = laplace_cov(res, method="analytic")
    print(f"\n1. analytic vs hessian Sigma      max|diff| = {np.abs(Sh - Sa).max():.2e}")

    # ... and the conditional/marginal gap the docstring warns about
    C = np.asarray(res.cov_params(), float)
    keep = np.r_[np.arange(k_fep), np.arange(k_fep + k_vcp, k_fep + k_vcp + k_vc)]
    marg = C[np.ix_(keep, keep)]
    print(f"   conditional vs marginal block  max|diff| = {np.abs(Sh - marg).max():.2e}"
          f"   (the vcp uncertainty the formula drops)")

    # 2 & 3. rank-one update against explicit re-inversion
    Xt = stack_design(Z, people, res)
    H = _inv(Sh)
    fep = np.asarray(res.params, float)[:k_fep]
    vcv = np.asarray(res.params, float)[k_fep + k_vcp:]
    mu = 1 / (1 + np.exp(-(Xt[:, :k_fep] @ fep + Xt[:, k_fep:] @ vcv)))
    w = mu * (1 - mu)
    ev = rng.choice(n, 40, replace=False)
    E = Xt[ev]
    worst_S, worst_d = 0.0, 0.0
    for c in rng.choice(n, 12, replace=False):
        xc, wc = Xt[c], w[c]
        S_exact = _inv(H + wc * np.outer(xc, xc))
        sc = Sh @ xc
        S_sm = Sh - np.outer(sc, sc) * (wc / (1 + wc * xc @ sc))
        worst_S = max(worst_S, np.abs(S_exact - S_sm).max())
        drop_true = np.einsum("ij,jk,ik->i", E, Sh - S_exact, E).sum()
        drop_formula = wc * ((E @ sc) ** 2).sum() / (1 + wc * (xc @ sc))
        worst_d = max(worst_d, abs(drop_true - drop_formula))
    print(f"\n2. Sherman-Morrison vs re-inversion  max|diff| = {worst_S:.2e}  (12 candidates)")
    print(f"3. Delta(c) vs realised variance drop max|diff| = {worst_d:.2e}  (40 eval rows)")

    ok = np.abs(Sh - Sa).max() < 1e-6 and worst_S < 1e-8 and worst_d < 1e-8
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="verify the algebra and exit")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(_self_test())
    ap.print_help()
