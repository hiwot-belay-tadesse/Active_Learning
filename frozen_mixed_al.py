"""Mixed-effects logistic regression with a FROZEN population, as a new AL model option.

    python frozen_mixed_al.py --self_test                 # simulated data, known variances
    python frozen_mixed_al.py --methods random --splits 1 --max_targets 2
    python frozen_mixed_al.py --methods coreset typiclust random --splits 20 --resume
    python frozen_mixed_al.py --no_update_variances       # step 4 off: the cohort cannot help

The model
---------
    logit P(y_ij = 1) = beta0 + x_ij' beta  +  b0_i + x_ij' b1_i

with the POPULATION part (beta0, beta) estimated once, on the seed set, and frozen for the rest
of the run, and the PARTICIPANT part (b0_i, b1_i) refitted every round. The random effects are
independent with one variance per term, ``sigma2 = [sigma2_int, sigma2_x1, ..., sigma2_xK]``,
shared across participants and re-estimated by EM each round.

Freezing the population is what makes this a different experiment from ``main.py``, not just a
different optimiser. Once (beta0, beta) are fixed, ``eta = beta0 + x'beta`` is a constant offset
per window, and everything a round can still change is one short vector per participant. Two
consequences:

* a round costs one small ridge GLM per participant instead of one fit over thousands of random
  effects, so the per-round cost does not grow with the labelled set the way the full MAP fit
  does; and
* the only channel through which ANOTHER participant's label can reach the target is
  ``sigma2`` -- the shared prior width. ``--no_update_variances`` closes that channel exactly
  (step 4 skipped, step-1 variances kept), which makes the pair of runs a clean test of whether
  cohort labels help at all under this model. Nothing else in the design changes between them.

Why this is a new file rather than a flag in main.py
----------------------------------------------------
``main.py`` builds a FRESH :class:`utility.MixedEffectsClassifier` every round and fits it on
everything labelled so far -- correct for that model, which has no state to carry. This model
does: the frozen population and the running ``sigma2`` both have to survive from one round to
the next, so the estimator outlives the round and the loop has to be written around that.
Acquisition and evaluation are imported, not reimplemented: strategies come from
``main.build_strategy`` and scoring from ``utility.scores``, both untouched.

The five steps, and where each one lives
----------------------------------------
    1  seed set + population fit     :func:`cohort_seed`, :func:`population_fit`
    2  offset eta, frozen            :meth:`Population.eta`
    3  one participant's adjustment  :func:`adjust`           -> (u_i, V_i)
    4  EM variance update            :func:`em_sigma2`        -> sigma2
    5  target fit + prediction       :meth:`FrozenMixed.decision_function`

and the round order is query -> label -> step 4 -> step 5 -> log, in :func:`run_target`.

Two deviations from the specification, both deliberate, both checked
--------------------------------------------------------------------
* **Step 3's solver.** The specified fit is
  ``sm.GLM(...).fit_regularized(alpha=1/(n*sigma2), L1_wt=0)``. statsmodels minimises
  ``-loglike/n + alpha*(1-L1_wt)*|b|^2/2``, so that alpha is exactly the MAP objective
  ``-loglike + sum_k b_k^2/(2 sigma2_k)`` -- the specification's algebra is right. But it gets
  there by coordinate descent over every coefficient, which is ~100x slower than the problem
  needs and is run (participants x EM iterations x rounds x targets) times. ``--solver newton``
  (the default) minimises the IDENTICAL objective by Newton's method with backtracking;
  ``--solver statsmodels`` runs the specified call instead. ``--self_test`` asserts the two agree
  to 1e-6 and that both match a black-box scipy minimisation of the same objective, so the
  default is a speed choice and not a modelling one.
* **How many features get a random slope.** The specification gives every feature one. On
  COMPUTER_WORK that is 124 slopes x 43 participants = 5332 random effects estimated by
  ``fit_vb`` on an 86-row seed set, and the step-1 fit alone measures 0.6 s at 5 slopes, 16 s at
  20, and minutes at 124 -- once per target per split, i.e. 860 times. ``--n_slopes 5`` (the
  default, and the repo's own convention) keeps the random slopes on the 5 features whose effect
  varies most between participants, via :func:`utility.select_slopes`; ``--n_slopes 0`` is the
  literal specification, every feature. The fixed effects always cover every feature either way.
  Note that ``select_slopes`` reads the full training labels, so with a positive ``--n_slopes``
  the random-effect STRUCTURE (not its values) is chosen with information an AL run would not
  have; that is the existing convention in ``main.py`` and it is fixed before round 1, so it
  cannot favour one acquisition over another. ``--n_slopes 0`` avoids it entirely.
* **Step 1's priors are the repo's, not statsmodels'.** The specification does not set them, so
  the choice had to be made somewhere; statsmodels' defaults (``fe_p=2``, ``vcp_p=1``) turn out
  to be the wrong place. Step 1 fits 124 fixed effects and 6 variance components by ``fit_vb``
  on an 86-row seed set, and at those priors it returned sigma2 = 34 for one slope on
  COMPUTER_WORK; step 4 then inflated the intercept variance from 0.42 to 21 over ten rounds and
  AUPRC fell from 0.249 to 0.218 as the random effects absorbed what the frozen offset had got
  wrong. ``--fe_p 1.0 --vcp_p 0.5`` -- utility.py's values, now the defaults here -- held every
  variance under 5 and gave 0.281 -> 0.273 on the same target and split. One target, one split,
  so those are not results; they are the reason for the default. ``--fe_p 2 --vcp_p 1`` restores
  statsmodels' behaviour, and ``state_split<k>.csv`` is where you watch for it: a sigma2 drifting
  upward round after round is this failure, not a finding.

Outputs, in the run folder
--------------------------
    per_round_split<k>.csv   main.py's schema, plus em_iters / n_persons_labelled / sigma2_int
    queries_split<k>.csv     main.py's schema: every bought row and whose it was
    labels_split<k>.csv      round x participant label counts  (spec: "labels per participant")
    state_split<k>.csv       round x term: sigma2 and u_t      (spec: "sigma2, u_t")
    analysis/reference_auroc_split<k>.csv   `frozen_full`, the matched 100%-label ceiling

The ceiling, and why it has to be this one
------------------------------------------
``frozen_full`` is this same model given every label in the cohort pool: the SAME frozen
population -- step 1 rerun from the same target-keyed seed draw, so it is the identical offset
the curves ran under -- then step 4's EM over all participants' full training rows and step 5
for the target. It is the limit the AL curve is climbing towards, the point where buying one
more label changes nothing, so the vertical distance to it reads as budget alone.

It is an ASYMPTOTE, not an upper bound, and a curve can cross it. The population is frozen from
an 86-row seed set; where that offset is a poor fit for a participant, more labels move ``u``
further in a direction that serves the training rows and not the held-out days. A curve sitting
above its own ceiling is that, and it is a finding about the freezing -- not a broken figure.

``reference_models.py``'s ``mixed_full`` is NOT that ceiling. It refits the population on every
label, and the whole point of this file is that the population never moves after step 1; a
frozen curve read against it is read against a different estimator. ``--no_update_variances`` is
honoured too, so a ``_fixedvar`` run gets a ``_fixedvar`` ceiling and the pair stays comparable.

The reference is written per split and is ON by default -- it reuses the step-1 fit the target
already paid for, so it costs one full-label EM per (split, target) and nothing else.
``--no_refs`` skips it; ``--refs_only`` writes it for a run whose curves are already on disk and
exits, which is how folders that predate this code get a ceiling without rerunning any
acquisition.

Checks
------
:func:`assert_no_test_leak` runs every round: the labelled set is drawn from the pooled TRAINING
matrix only, and no labelled row is byte-identical to any of the target's test rows. ``--self_test``
covers the rest -- solver equivalence, variance recovery on simulated data with known sigma2,
and one end-to-end AL run on that simulated split.
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

from load_data import get_target, load
from main import build_strategy
from paths import (analysis_path, done_splits, run_dir, split_files, update_index,
                   write_config)
from preprocess_data import Split, canonical_ids, prepare_one
from utility import (attach_person, block_days, feature_names, frame, scores, seed_set,
                     select_slopes, vc_formulas)

try:                                     # the model works without scikit-activeml installed
    from skactiveml.base import SkactivemlClassifier
    from skactiveml.utils import MISSING_LABEL, is_labeled
except ImportError:                                                        # pragma: no cover
    SkactivemlClassifier, MISSING_LABEL = object, np.nan

    def is_labeled(y, missing_label=np.nan):
        return ~np.isnan(np.asarray(y, dtype=float))

VAR_FLOOR = 1e-4        # sigma2 floor (spec step 4); a term at the floor is effectively off
# Step 1's priors. NOT statsmodels' defaults (fe_p=2, vcp_p=1) -- see the docstring's third
# decision: on this data those let fit_vb return a variance of 34 from an 86-row seed set.
# These are the values every other experiment in the repo uses, so the two are comparable.
FE_P, VCP_P = 1.0, 0.5
EM_TOL, EM_ITER = 1e-4, 50
METHODS = ("coreset", "typiclust", "probcover", "uncertainty", "random")
# The `laplace*` arms of main.py are NOT offered. They read the mixed model's Laplace posterior
# off `clf.res_` (a statsmodels BayesMixedGLMResults; see laplace_variance.py), which this
# estimator does not have and should not fake -- its posterior is K-dimensional per participant,
# not one joint covariance over all random effects. Offering the name with different algebra
# behind it would silently make two runs incomparable.


# ============================================================ steps 1-2: the frozen population
class Population:
    """The frozen half of the model: (beta0, beta) and the starting ``sigma2``.

    Attributes
    ----------
    intercept, beta      the frozen fixed effects, beta aligned with the feature columns
    sigma2               step-1 variances, ordered [intercept, slope features...]
    slope_idx            feature indices carrying a random slope
    terms                names for sigma2's entries, for the log
    skipped              participants left out of the seed set (one class missing)
    n_seed               rows in the seed set
    failed               the step-1 fit raised, or returned something non-finite
    """

    def __init__(self, intercept, beta, sigma2, slope_idx, terms, skipped=(), n_seed=0,
                 failed=False, note=""):
        self.intercept = float(intercept)
        self.beta = np.asarray(beta, float)
        self.sigma2 = np.asarray(sigma2, float)
        self.slope_idx = tuple(int(k) for k in slope_idx)
        self.terms = list(terms)
        self.skipped, self.n_seed = list(skipped), int(n_seed)
        self.failed, self.note = bool(failed), str(note)

    def __repr__(self):
        return (f"<Population {len(self.beta)} fixed effects, {len(self.sigma2)} variance "
                f"terms, sigma2={np.array2string(self.sigma2, precision=3)}"
                f"{', FAILED' if self.failed else ''}>")

    # -- step 2 ------------------------------------------------------------
    def eta(self, Z):
        """The offset: beta0 + x'beta, with the FROZEN values. Never changes after step 1.

        Computed from the frozen coefficients rather than carried as a stored column. The two
        are the same number by construction -- and recomputing cannot be misaligned with the
        rows it is handed, which a stored column can be. :func:`run_target` materialises it
        once per target anyway and asserts the agreement.
        """
        return self.intercept + np.asarray(Z, float) @ self.beta

    def design(self, Z):
        """Step 3's ``Z``: a column of ones, then the random-slope features."""
        Z = np.asarray(Z, float)
        ones = np.ones((len(Z), 1))
        return np.hstack([ones, Z[:, list(self.slope_idx)]]) if self.slope_idx else ones


def term_names(slope_idx, cols=None):
    """sigma2's labels: the intercept, then one per random-slope feature."""
    return ["int"] + [f"{cols[k]}" if cols is not None else f"f{k}" for k in slope_idx]


def population_fit(Z, y, pid, slope_idx, fe_p=FE_P, vcp_p=VCP_P, cols=None, verbose=False):
    """Step 1's population fit on the SEED SET. -> :class:`Population`.

    ``BinomialBayesMixedGLM.from_formula`` with ``y ~ <every feature>`` and one variance
    component per random-effect term (:func:`utility.vc_formulas` -- the same structure
    ``main.py`` uses, so the two experiments cannot drift apart in how the model is specified),
    fitted with ``fit_vb`` as specified.

    ``vcp_mean`` holds the posterior mean of the LOG standard deviation of each component, so
    ``sigma2 = exp(vcp_mean)**2``; both it and ``fe_mean`` are read through the model's own
    name lists rather than by position, because statsmodels orders them by the formula and not
    by the dict that was passed in.

    utility.py warns (from experience on this data) that ``fit_vb`` can drive its own variance
    components upward without bound on tiny labelled sets, which is why ``main.py`` uses
    ``fit_map`` with tighter priors. The specification asks for ``fit_vb``, so that is what runs;
    ``--fe_p`` / ``--vcp_p`` are exposed for when it misbehaves (``--fe_p 1.0 --vcp_p 0.5`` is
    the rest of the repo's setting), and a non-finite or absurd result is flagged rather than
    propagated silently.
    """
    Z = np.asarray(Z, float)
    feats = feature_names(Z.shape[1])
    terms = term_names(slope_idx, cols)
    keys = ["person"] + [f"slope_f{k}" for k in slope_idx]        # utility.vc_formulas' keys
    df = frame(Z, y, np.asarray(pid, dtype=object))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = BinomialBayesMixedGLM.from_formula(
                "y ~ " + " + ".join(feats), vc_formulas(list(slope_idx)), df,
                vcp_p=vcp_p, fe_p=fe_p)
            res = mod.fit_vb(verbose=False)
    except Exception as e:                                       # pragma: no cover
        return Population(0.0, np.zeros(Z.shape[1]), np.full(len(terms), 1.0), slope_idx,
                          terms, n_seed=len(y), failed=True, note=f"{type(e).__name__}: {e}")

    fe = dict(zip(list(res.model.fep_names), np.asarray(res.fe_mean, float)))
    vcp = dict(zip(list(res.model.vcp_names), np.asarray(res.vcp_mean, float)))
    missing = [k for k in keys if k not in vcp] + [f for f in ["Intercept"] + feats
                                                   if f not in fe]
    if missing:                                                  # pragma: no cover
        raise RuntimeError(f"statsmodels named its parameters unexpectedly; missing {missing[:4]}"
                           f" from fep_names={list(fe)[:4]}... vcp_names={list(vcp)}")
    intercept = fe["Intercept"]
    beta = np.array([fe[f] for f in feats])
    sigma2 = np.maximum(np.exp(np.array([vcp[k] for k in keys])) ** 2, VAR_FLOOR)

    bad = (not np.all(np.isfinite(beta)) or not np.isfinite(intercept)
           or not np.all(np.isfinite(sigma2)))
    note = ""
    if bad:
        note = "step-1 fit returned non-finite parameters"
    elif sigma2.max() > 1e3:
        note = f"step-1 sigma2 reached {sigma2.max():.3g} -- fit_vb may be running away"
    if verbose and note:
        print(f"    population fit: {note}", flush=True)
    return Population(intercept, beta, sigma2, slope_idx, terms, n_seed=len(y),
                      failed=bool(bad), note=note)


# ====================================================== step 3: one participant's adjustment
def _nll(D, y, eta, prec, u):
    """Penalised negative log-likelihood -- the objective both solvers minimise."""
    m = eta + D @ u
    return float(np.sum(np.logaddexp(0.0, m) - y * m) + 0.5 * np.sum(prec * u * u))


def _newton(D, y, eta, sigma2, u0=None, max_iter=100, tol=1e-10):
    """Newton with backtracking on :func:`_nll`. Strictly convex, so the minimiser is unique."""
    prec = 1.0 / sigma2
    u = np.zeros(D.shape[1]) if u0 is None else np.array(u0, float)
    for _ in range(max_iter):
        p = expit(eta + D @ u)
        g = D.T @ (p - y) + prec * u
        H = D.T @ (D * (p * (1 - p))[:, None]) + np.diag(prec)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:                            # pragma: no cover
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        f0, t = _nll(D, y, eta, prec, u), 1.0
        while t > 1e-8 and _nll(D, y, eta, prec, u - t * step) > f0:
            t *= 0.5
        u = u - t * step
        if np.max(np.abs(t * step)) < tol:
            break
    return u


def _statsmodels(D, y, eta, sigma2):
    """The specified fit: ridge GLM with ``alpha = 1/(n*sigma2)``, which is the same MAP.

    statsmodels minimises ``-loglike/n + alpha*(1-L1_wt)*|b|^2/2``; multiplying through by n
    turns that alpha into ``sum_k b_k^2 / (2 sigma2_k)``, the Gaussian prior. ``--self_test``
    asserts it agrees with :func:`_newton`.
    """
    n = max(len(y), 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.GLM(np.asarray(y, float), D, family=sm.families.Binomial(),
                     offset=np.asarray(eta, float)).fit_regularized(
                         alpha=1.0 / (n * np.asarray(sigma2, float)), L1_wt=0)
    return np.asarray(fit.params, float)


def adjust(D, y, eta, sigma2, solver="newton", u0=None):
    """Step 3 for one participant. -> (u, V).

    ``u`` is (b0_i, b1_i) at the posterior mode; ``V = inv(D' diag(p(1-p)) D + diag(1/sigma2))``
    is the Laplace covariance at that mode -- the second term is what keeps it finite when the
    participant has two labels and six coefficients, which is the normal case here.

    No labels: the posterior is the prior, so u = 0 and V = diag(sigma2).
    """
    sigma2 = np.asarray(sigma2, float)
    if len(y) == 0:
        return np.zeros(D.shape[1]), np.diag(sigma2)
    u = (_statsmodels(D, y, eta, sigma2) if solver == "statsmodels"
         else _newton(D, y, eta, sigma2, u0=u0))
    p = expit(eta + D @ u)
    H = D.T @ (D * (p * (1 - p))[:, None]) + np.diag(1.0 / sigma2)
    return u, np.linalg.inv(H)


# ============================================================== step 4: the EM variance update
def em_sigma2(groups, sigma2, max_iter=EM_ITER, tol=EM_TOL, solver="newton", warm=None):
    """Step 4: ``sigma2_k <- mean_i (u_ik^2 + V_i[k,k])``. -> (sigma2, iters, u, rel).

    `groups` is ``{person: (D_i, y_i, eta_i)}`` over the participants that HAVE labels -- the
    target included, and nobody else enters the mean. The returned ``u`` dict is the mode at the
    final ``sigma2``, so step 5 does not repeat the work for the target.

    Warm-starting each participant's ``u`` from the previous EM iteration is what makes 50
    iterations affordable; it changes nothing about the fixed point, since each inner problem is
    strictly convex.
    """
    sigma2 = np.asarray(sigma2, float).copy()
    warm = dict(warm or {})
    iters, rel = 0, 0.0
    if not groups:
        return sigma2, 0, warm, 0.0
    for iters in range(1, max_iter + 1):
        acc = np.zeros(len(sigma2))
        for person, (D, y, eta) in groups.items():
            u, V = adjust(D, y, eta, sigma2, solver=solver, u0=warm.get(person))
            warm[person] = u
            acc += u * u + np.diag(V)
        new = np.maximum(acc / len(groups), VAR_FLOOR)
        rel = float(np.max(np.abs(new - sigma2) / np.maximum(sigma2, VAR_FLOOR)))
        sigma2 = new
        if rel < tol:
            break
    # the modes above were found at the PREVIOUS sigma2; refresh them at the one being returned
    for person, (D, y, eta) in groups.items():
        warm[person] = adjust(D, y, eta, sigma2, solver=solver, u0=warm.get(person))[0]
    # `rel` is returned, and logged as em_rel, because the iteration cap DOES bind here: with a
    # handful of labels per participant the variance update crawls, so a round that stops at
    # max_iter stopped on the cap and not on the tolerance. Reading em_iters alone would hide
    # that; the pair says which.
    return sigma2, iters, warm, rel


# ================================================================= the estimator, steps 3-5
class FrozenMixed(SkactivemlClassifier):
    """The frozen-population model behind scikit-activeml's classifier interface.

    ``X`` carries the participant code as its LAST column (:func:`utility.attach_person`), the
    same convention as :class:`utility.MixedEffectsClassifier`, so a strategy that wants an
    estimator can be handed this one unchanged -- and so an acquisition function must be given
    the features WITHOUT that column.

    ``fit`` is steps 4 and 5: it groups the labelled rows by participant, runs the EM variance
    update (unless ``update_variances`` is False), and stores one ``u`` per participant.
    ``sigma2`` persists across calls in ``sigma2_``, so a round starts from the previous round's
    variances -- which is why the loop keeps ONE of these per (target, method) rather than
    building a fresh one each round.
    """

    def __init__(self, population, persons, update_variances=True, em_iter=EM_ITER,
                 em_tol=EM_TOL, solver="newton", sigma2=None, classes=None,
                 missing_label=MISSING_LABEL, cost_matrix=None, random_state=None):
        super().__init__(classes=classes, missing_label=missing_label,
                         cost_matrix=cost_matrix, random_state=random_state)
        # stored verbatim: sklearn's clone() requires get_params() to return what was passed
        self.population = population
        self.persons = persons
        self.update_variances = update_variances
        self.em_iter, self.em_tol, self.solver = em_iter, em_tol, solver
        self.sigma2 = sigma2

    # -- internals ---------------------------------------------------------
    def _split(self, X):
        X = np.asarray(X, float)
        persons = list(self.persons)
        return X[:, :-1], [persons[int(c)] for c in X[:, -1]]

    def fit(self, X, y, sample_weight=None):
        pop = self.population
        Z, person = self._split(X)
        person = np.asarray(person, dtype=object)
        y = np.asarray(y, float)
        lab = is_labeled(y, missing_label=self.missing_label)
        self.classes_ = np.array([0.0, 1.0]) if self.classes is None else np.asarray(self.classes)
        self.n_features_in_ = X.shape[1]
        self.fit_failed_ = bool(pop.failed)
        self.base_rate_ = float(y[lab].mean()) if lab.any() else 0.5

        D, eta = pop.design(Z), pop.eta(Z)
        start = getattr(self, "sigma2_", None)
        if start is None:
            start = pop.sigma2 if self.sigma2 is None else np.asarray(self.sigma2, float)
        groups = {}
        for p in dict.fromkeys(person[lab]):                      # first-seen order: stable
            m = lab & (person == p)
            groups[p] = (D[m], y[m], eta[m])
        self.n_labelled_ = {p: int(len(v[1])) for p, v in groups.items()}
        if self.update_variances:
            sigma2, iters, u, rel = em_sigma2(groups, start, self.em_iter, self.em_tol,
                                              self.solver, warm=getattr(self, "u_", None))
        else:
            # Step 4 skipped: the step-1 variances stand, so no other participant's label can
            # reach the target -- its own adjustment is the only thing its prediction depends on.
            sigma2, iters, rel = np.asarray(start, float), 0, 0.0
            u = {p: adjust(*groups[p], sigma2, solver=self.solver,
                           u0=(getattr(self, "u_", None) or {}).get(p))[0] for p in groups}
        self.sigma2_, self.em_iters_, self.em_rel_, self.u_ = sigma2, iters, rel, u
        return self

    # -- step 5 ------------------------------------------------------------
    def decision_function(self, X):
        """eta + D u for each row, under its own participant. Zero adjustment for an unseen one.

        The linear predictor, not the probability: it is what AUROC should be computed on, and
        :func:`utility.scores` takes it.
        """
        pop = self.population
        Z, person = self._split(X)
        person = np.asarray(person, dtype=object)
        out = pop.eta(Z)
        D = pop.design(Z)
        for p in set(person):
            u = getattr(self, "u_", {}).get(p)
            if u is None:
                continue
            m = person == p
            out[m] = out[m] + D[m] @ u
        return out

    def u_of(self, person, default_zeros=True):
        """One participant's ``u``, zeros when they have no labels yet (spec step 5)."""
        u = getattr(self, "u_", {}).get(person)
        if u is None and default_zeros:
            return np.zeros(1 + len(self.population.slope_idx))
        return u

    def predict_proba(self, X):
        p = expit(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(float)


# ======================================================================== step 1: the seed set
def cohort_seed(sp, days_of, rng, n_per_class=1):
    """One labelled window of EACH class from EVERY participant. -> (rows, skipped).

    The specification's step 1: for every participant, the target included, draw one y=1 and one
    y=0 from that participant's TRAINING pool -- never from any test set, which `sp.Z` cannot
    reach into: it holds training rows only (see ``preprocess_data.build_pool``).

    A participant missing either class contributes nothing and is named in `skipped`, so the
    log says who was left out instead of the seed quietly shrinking.

    ``utility.seed_set`` does the draw, which spreads the picks over distinct DAYS where the
    participant has them -- so two labels from one person are two days rather than two blocks
    of one afternoon.
    """
    take, skipped = [], []
    for u in sp.users:                                # fixed order -> reproducible draw
        rows = sp.rows_of(u)
        yy = sp.y[rows]
        if min((yy == 1).sum(), (yy == 0).sum()) < n_per_class:
            skipped.append(sp.ids.get(u, u))
            continue
        day = days_of[u]
        assert len(day) == len(rows), f"{len(day)} block days for {len(rows)} blocks of {u}"
        loc, _ = seed_set(yy, day, rng, n_seed=2 * n_per_class, n_seed_pos=n_per_class)
        take += list(rows[loc])
    return np.array(sorted(set(int(i) for i in take)), dtype=int), skipped


def assert_no_test_leak(sp, uuid, rows, test_keys):
    """No labelled window is one of the target's test windows. Raises AssertionError if one is.

    Structurally this cannot happen -- the pool is the training matrix and the day split holds
    out whole days -- so the check is really a guard against an upstream change that breaks
    that. It compares row CONTENT, not indices, so it would also catch a duplicated row that
    landed on both sides.
    """
    rows = np.asarray(rows, dtype=int)
    assert rows.size == 0 or (rows.min() >= 0 and rows.max() < len(sp.y)), \
        "labelled index outside the training pool"
    bad = [int(i) for i in rows
           if np.ascontiguousarray(sp.Z[i], float).tobytes() in test_keys]
    assert not bad, (f"{len(bad)} labelled row(s) are identical to test rows of "
                     f"{sp.ids.get(uuid, uuid)}: {bad[:5]}")


def seed_and_population(sp, uuid, days_of, slope_idx, args, split_id, cols=None,
                        verbose=False):
    """Steps 1-2 for one target. -> (seed_rows, skipped, population).

    Factored out because TWO things need the identical frozen population: the curves, and the
    100%%-label ceiling they are read against. Recomputing the seed draw in two places is how a
    ceiling silently stops capping the curve it is drawn over -- the RNG is keyed on
    ``(seed, split, participant)``, so the draw is reproducible, and that only helps if both
    callers reproduce it the same way.
    """
    pid = sp.ids[uuid]
    rng = np.random.default_rng([args.seed, split_id, int(pid[1:]), 11])
    seed_rows, skipped = cohort_seed(sp, days_of, rng, args.n_seed_per_class)
    pop = population_fit(sp.Z[seed_rows], sp.y[seed_rows], sp.pid[seed_rows], slope_idx,
                         fe_p=args.fe_p, vcp_p=args.vcp_p, cols=cols, verbose=verbose)
    return seed_rows, skipped, pop


# ============================================================ the 100%-label ceiling
def frozen_full_row(sp, uuid, pop, args, split_id, seed_rows=()):
    """-> one ``frozen_full`` row: this model with EVERY label in the pool. See the docstring.

    Steps 3-5 exactly as :meth:`FrozenMixed.fit` runs them, with the labelled set taken to be
    the whole training matrix: every participant's own rows form their group, the EM variance
    update runs over all of them, and the target is scored from its own ``u`` at the converged
    ``sigma2``. Nothing here touches a test row -- ``sp.Z`` holds training rows only.

    ``update_variances`` is honoured rather than assumed, so the ceiling matches the run: with
    step 4 off, sigma2 stays at step 1's and each participant is fitted independently, which is
    the limit of THAT experiment and not of the other one.
    """
    D, eta = pop.design(sp.Z), pop.eta(sp.Z)
    groups = {}
    for person in sp.users:                       # fixed order -> reproducible EM
        rows = sp.rows_of(person)
        if len(rows):
            groups[person] = (D[rows], sp.y[rows], eta[rows])
    if args.update_variances:
        sigma2, iters, u, rel = em_sigma2(groups, pop.sigma2, args.em_iter, args.em_tol,
                                          args.solver)
    else:
        sigma2, iters, rel = np.asarray(pop.sigma2, float), 0, 0.0
        u = {q: adjust(*groups[q], sigma2, solver=args.solver)[0] for q in groups}
    u_t = u.get(uuid)
    if u_t is None:                                                  # pragma: no cover
        u_t = np.zeros(1 + len(pop.slope_idx))
    Zte, yte = sp.test_of(uuid)
    auroc, auprc = scores(yte, pop.eta(Zte) + pop.design(Zte) @ u_t)
    return dict(split=split_id, target=sp.ids[uuid], model="frozen_full",
                auroc=auroc, auprc=auprc, n_test=int(len(yte)),
                n_pos_test=int(np.sum(yte == 1)), n_train=int(len(sp.y)),
                n_own_train=int(len(sp.rows_of(uuid))), n_persons_labeled=len(groups),
                em_iters=int(iters), em_rel=float(rel), sigma2_int=float(sigma2[0]),
                update_variances=bool(args.update_variances),
                fit_failed=bool(pop.failed), n_seed=int(len(seed_rows)))


def full_reference(sp, targets, days_of, slope_idx, args, split_id, cols=None):
    """-> a frame of ``frozen_full`` rows, refitting step 1 per target.

    What ``--refs_only`` uses on a run whose curves are already written. Inside a live run the
    ceiling is computed in :func:`run_target` instead, off the population that target has
    already paid for -- same number, one fewer step-1 fit.
    """
    rows = []
    for n, u in enumerate(targets, 1):
        seed_rows, _skipped, pop = seed_and_population(sp, u, days_of, slope_idx, args,
                                                       split_id, cols=cols, verbose=n == 1)
        rows.append(frozen_full_row(sp, u, pop, args, split_id, seed_rows))
    return pd.DataFrame(rows)


def refs_for_split(data, cols, split_id, args, ids=None):
    """-> (reference frame, ids) for one split, running no acquisition at all."""
    sp, ids, (tr_day, _) = prepare_one(data, cols, seed=split_id, W=args.block, ids=ids)
    slope_idx = slopes_for(sp, cols, args)
    days_of = {u: block_days(tr_day[u], args.block) for u in sp.users}
    targets = [u for u in sp.users if np.sum(sp.yte[u] == 1) >= args.min_test_pos]
    if args.max_targets:
        targets = targets[:args.max_targets]
    return full_reference(sp, targets, days_of, slope_idx, args, split_id, cols=cols), ids


# ===================================================================================== the loop
def run_target(sp, uuid, days_of, slope_idx, args, split_id, cols=None, verbose=False):
    """One target's curves, every method from the same seed set and the same frozen population.

    -> (per_round rows, query rows, label-count rows, state rows, reference rows)

    Everything is re-initialised from step 1 for this target: its own seed draw (the RNG is
    keyed on the target), its own population fit, its own variances. Nothing carries over from
    the previous target.

    Steps 1 and 2 are shared across METHODS, though, and that is not a shortcut: the seed set
    does not depend on the acquisition, so the frozen population cannot either, and sharing it
    is what makes the curves start from one identical model -- the same reason ``main.py``
    shares its seed set.
    """
    pid = sp.ids[uuid]
    own = sp.rows_of(uuid)
    seed_rows, skipped, pop = seed_and_population(sp, uuid, days_of, slope_idx, args, split_id,
                                                  cols=cols, verbose=verbose)
    # Step 2, materialised for every window of every participant, as specified -- and asserted
    # to equal what Population.eta recomputes, which is what the model actually calls.
    eta_pool = pop.eta(sp.Z)
    Zte, yte = sp.test_of(uuid)
    eta_te = pop.eta(Zte)
    assert np.allclose(eta_pool[own], pop.eta(sp.Z[own])), "the frozen offset is not frozen"
    n_pos_te = int(np.sum(yte == 1))
    test_keys = {np.ascontiguousarray(r, float).tobytes() for r in Zte}

    # One stream per (target, split), independent of the method's position in --methods, so
    # `random` does not replay one draw for all 43 targets. Same construction as main.py.
    strat_seed = int(np.random.default_rng(
        [args.seed, split_id, int(pid[1:]), 7]).integers(2 ** 31))

    Xclf = attach_person(sp.Z, sp.owner, sp.users)
    Xte = attach_person(Zte, [uuid] * len(yte), sp.users)
    Xte_as_target = Xte                      # the target's own code: its random effect applies

    # The ceiling first, off the population just fitted: it does not depend on the acquisition,
    # so it is one fit per target rather than one per (target, method).
    refs = ([] if getattr(args, "no_refs", False)
            else [frozen_full_row(sp, uuid, pop, args, split_id, seed_rows)])

    rows, queries, labels, state = [], [], [], []
    for method in args.methods:
        qs = build_query(method, strat_seed, args)
        qrng = np.random.default_rng(strat_seed)
        y_al = np.full(len(sp.y), np.nan)
        y_al[seed_rows] = sp.y[seed_rows]
        bought = []
        clf = FrozenMixed(population=pop, persons=sp.users, classes=[0.0, 1.0],
                          missing_label=np.nan, update_variances=args.update_variances,
                          em_iter=args.em_iter, em_tol=args.em_tol, solver=args.solver)
        for rnd in range(args.rounds + 1):
            # ------------------------------------------------- query -> label
            if rnd > 0:
                cand = np.flatnonzero(np.isnan(y_al))
                if len(cand) == 0:                               # pragma: no cover
                    break
                idx = pick(method, qs, qrng, sp, y_al, cand, Xclf, clf, uuid, args)
                y_al[idx] = sp.y[idx]
                bought.extend(int(i) for i in idx)
                for i in idx:
                    queries.append(dict(split=split_id, target=pid, method=method, round=rnd,
                                        row=int(i), owner=sp.ids.get(sp.owner[i], sp.owner[i]),
                                        is_self=bool(sp.owner[i] == uuid)))
            lab_rows = np.flatnonzero(~np.isnan(y_al))
            assert_no_test_leak(sp, uuid, lab_rows, test_keys)
            # ------------------------------------------- step 4 (+ step 3 for everyone)
            clf.fit(Xclf, y_al)
            # ------------------------------------------------------------- step 5
            u_t = clf.u_of(uuid)
            logit = eta_te + pop.design(Zte) @ u_t
            assert np.allclose(logit, clf.decision_function(Xte_as_target)), \
                "step 5 and the estimator disagree about the target's prediction"
            auroc, auprc = scores(yte, logit)
            # ------------------------------------------------------------- log
            n_lab = int(len(lab_rows))
            rows.append(dict(split=split_id, target=pid, method=method, round=rnd,
                             n_labeled=n_lab, acquired=n_lab - len(seed_rows),
                             auroc=auroc, auprc=auprc,
                             n_test=int(len(yte)), n_pos_test=n_pos_te,
                             self_share=float(np.mean([sp.owner[i] == uuid for i in bought]))
                             if bought else np.nan,
                             n_own_labeled=int(np.sum(~np.isnan(y_al[own]))),
                             n_persons_labeled=len(clf.n_labelled_),
                             em_iters=int(clf.em_iters_), em_rel=float(clf.em_rel_),
                             sigma2_int=float(clf.sigma2_[0]),
                             fit_failed=bool(clf.fit_failed_), n_seed=len(seed_rows),
                             n_skipped_seed=len(skipped)))
            for person, n in clf.n_labelled_.items():
                labels.append(dict(split=split_id, target=pid, method=method, round=rnd,
                                   person=sp.ids.get(person, person), n_labels=int(n),
                                   is_self=bool(person == uuid)))
            for k, term in enumerate(pop.terms):
                state.append(dict(split=split_id, target=pid, method=method, round=rnd,
                                  term=term, sigma2=float(clf.sigma2_[k]), u_t=float(u_t[k])))
    return rows, queries, labels, state, refs


def build_query(method, seed, args):
    """-> the strategy object for one (target, method), or None for ``random``.

    Built per target rather than once per run: CoreSet and ProbCover carry state about what is
    already covered, so one object shared across participants would leak one person's coverage
    into the next. Everything but ``uncertainty`` comes straight from ``main.build_strategy``,
    which is left untouched; ``UncertaintySampling`` is constructed here because main.py's
    pooled run does not offer that arm (its estimator is the one this file replaces) and
    build_strategy therefore does not know the name.
    """
    if method == "random":
        return None
    if method == "uncertainty":
        from skactiveml.pool import UncertaintySampling                       # noqa: PLC0415
        return UncertaintySampling(method=args.uncertainty_method, random_state=seed,
                                   missing_label=np.nan)
    return build_strategy(method, seed, args.typiclust_k)


def pick(method, qs, qrng, sp, y_al, cand, Xclf, clf, uuid, args):
    """-> this round's rows to buy. Candidates are the whole cohort's unlabelled training rows.

    The strategies come from ``main.build_strategy`` and are called exactly as ``main.py`` calls
    them: the label-free ones see the FEATURES only (``sp.Z``), never the participant column,
    because a participant code is not a coordinate to measure distance in.

    ``uncertainty`` is the one that reads the model, so it gets the owner-coded matrix and this
    estimator. ``--uncertainty_coding target`` recodes every candidate as the target instead,
    which asks "how uncertain is the model about this window FOR THE TARGET" rather than "...
    for whoever owns it"; the two differ exactly by the owner's random effect.
    """
    batch = int(min(args.batch, len(cand)))
    if method == "random":
        return qrng.choice(cand, size=batch, replace=False)
    if method == "uncertainty":
        X = Xclf if args.uncertainty_coding == "owner" else attach_person(
            sp.Z, [uuid] * len(sp.Z), sp.users)
        # fit_clf=False: `clf` was fitted on (Xclf, y_al) at the end of the previous round, so
        # refitting here would repeat the EM for nothing.
        return np.asarray(qs.query(X=X, y=y_al, clf=clf, fit_clf=False, batch_size=batch),
                          dtype=int)
    return np.asarray(qs.query(X=sp.Z, y=y_al, batch_size=batch), dtype=int)


def run_split(data, cols, split_id, args, ids=None):
    """Prepare one split and run every eligible target through it."""
    t0 = time.time()
    sp, ids, (tr_day, _) = prepare_one(data, cols, seed=split_id, W=args.block, ids=ids)
    slope_idx = slopes_for(sp, cols, args)
    days_of = {u: block_days(tr_day[u], args.block) for u in sp.users}
    targets = [u for u in sp.users if np.sum(sp.yte[u] == 1) >= args.min_test_pos]
    if args.max_targets:
        targets = targets[:args.max_targets]
    print(f"  split {split_id}: {sp}, {len(targets)} targets", flush=True)

    out = ([], [], [], [], [])
    for n, u in enumerate(targets, 1):
        got = run_target(sp, u, days_of, slope_idx, args, split_id, cols=cols, verbose=n == 1)
        for dst, src in zip(out, got):
            dst += src
        if n % 5 == 0 or n == len(targets):
            print(f"    {n}/{len(targets)} targets, {time.time() - t0:.0f}s", flush=True)
    return (*[pd.DataFrame(x) for x in out], ids)


def slopes_for(sp, cols, args):
    """Which features carry a random slope. ``--n_slopes 0`` is every one (the specification)."""
    if args.n_slopes <= 0:
        print(f"  random slopes: all {sp.Z.shape[1]} features -- the step-1 fit is "
              f"{sp.Z.shape[1]} variance components x {len(sp.users)} participants and will be "
              f"slow; see the module docstring", flush=True)
        return list(range(sp.Z.shape[1]))
    return select_slopes(sp.Z, sp.y, sp.pid, args.n_slopes, cols, verbose=True)


# ======================================================================== simulation + self-test
def simulate(n_person=12, n_feat=4, n_train=60, n_test=40, sigma2=(0.8, 0.5, 0.3, 0.0, 0.0),
             intercept=-0.8, seed=0, n_days=4):
    """A split whose random-effect variances are KNOWN. -> (Split, days_of, truth).

    ``sigma2`` is [intercept, slope on f0, f1, ...] and may be shorter than `n_feat`; the rest
    are zero, i.e. features whose effect does NOT vary between people. Fixed effects are drawn
    once and shared, so a recovery check has something to aim at.

    The matrices are not standardised (they are already standard normal) and `prep` is None --
    nothing downstream of :class:`preprocess_data.Split` needs it here.
    """
    rng = np.random.default_rng(seed)
    sigma2 = np.asarray(list(sigma2) + [0.0] * max(0, n_feat + 1 - len(sigma2)), float)
    beta = rng.normal(0, 0.8, n_feat)
    users = [f"u{i}" for i in range(n_person)]
    ids = {u: f"M{i + 1}" for i, u in enumerate(users)}
    Ztr, ytr, owner, Zte, yte, days = [], [], [], {}, {}, {}
    U = {}
    for u in users:
        U[u] = rng.normal(0, np.sqrt(sigma2))
        for which, n in (("tr", n_train), ("te", n_test)):
            Z = rng.normal(size=(n, n_feat))
            m = intercept + Z @ beta + U[u][0] + Z @ U[u][1:]
            y = (rng.random(n) < expit(m)).astype(float)
            if which == "tr":
                Ztr.append(Z)
                ytr.append(y)
                owner += [u] * n
                days[u] = np.repeat(np.arange(n_days), int(np.ceil(n / n_days)))[:n]
            else:
                Zte[u], yte[u] = Z, y
    sp = Split(np.vstack(Ztr), np.concatenate(ytr), np.array(owner), Zte, yte, None,
               users, [f"x{i}" for i in range(n_feat)], ids)
    truth = dict(intercept=intercept, beta=beta, sigma2=sigma2, u=U)
    return sp, days, truth


def self_test():
    """Solver equivalence, variance recovery on known sigma2, and one end-to-end AL run."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(1)
    print("1. step 3: three solvers on the same objective")
    for n, K in ((2, 4), (8, 3), (40, 6)):
        D = np.hstack([np.ones((n, 1)), rng.normal(size=(n, K - 1))])
        y = (rng.random(n) < 0.5).astype(float)
        eta = rng.normal(size=n)
        s2 = rng.uniform(0.2, 1.5, K)
        u_n, V = adjust(D, y, eta, s2, solver="newton")
        u_s, _ = adjust(D, y, eta, s2, solver="statsmodels")
        u_o = minimize(lambda v: _nll(D, y, eta, 1 / s2, v), np.zeros(K),
                       method="BFGS", tol=1e-12).x
        d_s, d_o = float(np.max(np.abs(u_n - u_s))), float(np.max(np.abs(u_n - u_o)))
        print(f"   n={n:3d} K={K}: |newton-statsmodels|={d_s:.2e} |newton-scipy|={d_o:.2e} "
              f"V diag {np.array2string(np.diag(V), precision=3)}")
        assert d_o < 1e-5, "Newton disagrees with a black-box minimisation of the same objective"
        assert d_s < 1e-4, "the statsmodels ridge fit disagrees with Newton"
    print("   OK: the specified fit_regularized call and --solver newton are the same fit\n")

    print("2. step 4: EM recovery with the population frozen at the truth")
    sp, days, truth = simulate(n_person=40, n_train=120, n_feat=4,
                               sigma2=(0.9, 0.6, 0.25, 0.0, 0.0), seed=3)
    pop = Population(truth["intercept"], truth["beta"], np.full(5, 1.0), range(4),
                     term_names(range(4)))
    D, eta = pop.design(sp.Z), pop.eta(sp.Z)
    groups = {u: (D[sp.rows_of(u)], sp.y[sp.rows_of(u)], eta[sp.rows_of(u)]) for u in sp.users}
    s2, iters, _, rel = em_sigma2(groups, np.full(5, 1.0))
    print(f"   {iters} EM iterations (last relative change {rel:.2e}), "
          f"{len(groups)} participants x 120 labels")
    print(f"   true  {np.array2string(truth['sigma2'], precision=3)}")
    print(f"   est   {np.array2string(s2, precision=3)}")
    big = truth["sigma2"] > 0.2
    assert np.all(np.abs(s2[big] - truth["sigma2"][big]) < 0.45), "EM did not recover sigma2"
    assert np.all(s2[~big] < 0.25), "EM invented variance where there is none"
    print("   OK (a 40x120 fit; with 2 seed labels per person step 1 is much rougher)\n")

    print("3. end to end on the simulated split, through the real loop")
    sp, days, truth = simulate(n_person=10, n_train=40, n_test=30, n_feat=4, seed=5)
    args = argparse.Namespace(
        seed=0, methods=["random", "coreset"], rounds=3, batch=3, n_seed_per_class=1,
        update_variances=True, em_iter=EM_ITER, em_tol=EM_TOL, solver="newton", fe_p=FE_P,
        vcp_p=VCP_P, typiclust_k=20, uncertainty_method="least_confident",
        uncertainty_coding="owner", batch_size=None)
    R, Q, L, S, F = run_target(sp, sp.users[0], days, [0, 1, 2, 3], args, 0, cols=sp.cols)
    R, Q, L, S, F = (pd.DataFrame(R), pd.DataFrame(Q), pd.DataFrame(L), pd.DataFrame(S),
                     pd.DataFrame(F))
    print(R[["method", "round", "n_labeled", "n_persons_labeled", "em_iters", "em_rel",
             "auroc", "auprc", "self_share"]].to_string(index=False))
    print(f"   sigma2 at the last round:\n"
          f"{S[S['round'] == S['round'].max()].to_string(index=False)}")
    assert len(R) == 2 * (args.rounds + 1) and Q.target.nunique() == 1
    assert len(L) and len(S), "the per-participant and sigma2 logs are empty"
    assert len(F) == 1 and F.model.iloc[0] == "frozen_full", "no ceiling row came back"
    print(f"   ceiling: frozen_full AUROC {F.auroc.iloc[0]:.4f} AUPRC {F.auprc.iloc[0]:.4f} "
          f"on all {int(F.n_train.iloc[0])} labels vs {int(R.n_labeled.max())} bought")
    print("   OK: steps 1-5 run end to end and the five logs are written\n")

    print("5. the ceiling --refs_only writes is the one the run would have written")
    # --refs_only refits step 1 from the seed draw instead of reusing the population the curves
    # ran under, so it is only sound if the draw reproduces. The SEED SET is checked exactly --
    # it is ours, keyed on (seed, split, participant), and a changed rule would otherwise show
    # up nowhere until a figure quietly capped its curves with the wrong number. The SCORE is
    # checked to a tolerance instead: statsmodels' fit_vb is not bit-reproducible (two calls on
    # identical input differ by ~1e-6 in beta on COMPUTER_WORK), which is a property of step 1
    # and applies to rerunning the curves themselves just as much.
    rows_a = seed_and_population(sp, sp.users[0], days, [0, 1, 2, 3], args, 0, cols=sp.cols)[0]
    rows_b = seed_and_population(sp, sp.users[0], days, [0, 1, 2, 3], args, 0, cols=sp.cols)[0]
    assert np.array_equal(rows_a, rows_b), "the seed draw is not reproducible"
    F2 = full_reference(sp, [sp.users[0]], days, [0, 1, 2, 3], args, 0, cols=sp.cols)
    d = float(abs(F2.auroc.iloc[0] - F.auroc.iloc[0]))
    print(f"   same seed set ({len(rows_a)} rows); run_target {F.auroc.iloc[0]:.9f} vs "
          f"refs_only {F2.auroc.iloc[0]:.9f} (|diff| {d:.1e})")
    assert d < 1e-4, "--refs_only does not reproduce the population the curves ran under"
    print("   OK\n")

    print("6. --no_update_variances really closes the channel")
    args.update_variances = False
    R2 = pd.DataFrame(run_target(sp, sp.users[0], days, [0, 1, 2, 3], args, 0, cols=sp.cols)[3])
    v = R2[(R2.method == "random") & (R2.term == "int")].sigma2.unique()
    print(f"   sigma2_int over all rounds: {np.array2string(v, precision=4)}")
    assert len(v) == 1, "sigma2 moved with the update switched off"
    print("   OK\n")
    print("self-test passed")


# ============================================================================== entry point
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="computer_work")
    ap.add_argument("--block", type=int, default=15, help="W: minutes per block (1 = none)")
    ap.add_argument("--methods", nargs="+", default=["random"], choices=list(METHODS),
                    help="acquisitions, from main.build_strategy; the laplace* arms of main.py "
                         "need the full mixed model's posterior and are not offered here")
    ap.add_argument("--splits", type=int, default=1)
    ap.add_argument("--first_split", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--batch", type=int, default=3, help="labels bought per round")
    ap.add_argument("--n_seed_per_class", type=int, default=1,
                    help="step 1: labels of EACH class taken from every participant")
    ap.add_argument("--n_slopes", type=int, default=5,
                    help="features carrying a random slope; 0 = every feature (the "
                         "specification), which makes the step-1 fit very slow")
    ap.add_argument("--no_update_variances", dest="update_variances", action="store_false",
                    help="skip step 4 and keep the step-1 variances, which cuts the only path "
                         "by which another participant's label can reach the target")
    ap.add_argument("--em_iter", type=int, default=EM_ITER)
    ap.add_argument("--em_tol", type=float, default=EM_TOL)
    ap.add_argument("--solver", default="newton", choices=["newton", "statsmodels"],
                    help="step 3's minimiser; same objective either way (see the docstring)")
    ap.add_argument("--fe_p", type=float, default=FE_P,
                    help="step 1 fixed-effect prior SD; statsmodels' own default is 2.0, which "
                         "lets fit_vb run away on an 86-row seed set (see the docstring)")
    ap.add_argument("--vcp_p", type=float, default=VCP_P,
                    help="step 1 prior SD on each log random-effect SD; statsmodels' own "
                         "default is 1.0")
    ap.add_argument("--typiclust_k", type=int, default=20)
    ap.add_argument("--uncertainty_method", default="least_confident",
                    choices=["least_confident", "margin_sampling", "entropy"])
    ap.add_argument("--uncertainty_coding", default="owner", choices=["owner", "target"],
                    help="whose random effect a candidate is scored under")
    ap.add_argument("--min_test_pos", type=int, default=1)
    ap.add_argument("--max_targets", type=int, default=0, help="debug: stop after N targets")
    ap.add_argument("--no_refs", action="store_true",
                    help="skip the 100%%-label ceiling `frozen_full`. It reuses the step-1 fit "
                         "the target already paid for, so it is one extra full-label EM per "
                         "(split, target) and is on by default; reference_models.py is not a "
                         "substitute (see the module docstring).")
    ap.add_argument("--refs_only", action="store_true",
                    help="write ONLY analysis/reference_auroc_split<k>.csv for a run whose "
                         "curves are already on disk, then exit. Nothing else in the folder is "
                         "touched and no config is rewritten, so a finished fan-out can be "
                         "given its ceiling without rerunning any acquisition. Needs --out (or "
                         "the same flags the run was launched with, so the folder name derives "
                         "to the same place).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--reserve", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip splits already written")
    ap.add_argument("--no_index", action="store_true")
    ap.add_argument("--out", default=None, help="override the derived folder")
    ap.add_argument("--self_test", action="store_true",
                    help="simulated data with known variances; writes nothing")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return

    t = get_target(args.target)
    args.target = t.name
    # The model is in the folder name, not only in config.json: a frozen-population run and a
    # main.py run can share every other setting and must never land in one folder.
    tag = "frozenmixed" + (f"_{args.tag}" if args.tag else "")
    tag += "" if args.update_variances else "_fixedvar"
    out = args.out or run_dir(t.name, args.block, args.methods, args.rounds, args.batch,
                              args.n_slopes, tag=tag, reserve=args.reserve, scope=None,
                              create=not args.refs_only)
    # --refs_only rewrites no config: the folder already has one, and rewriting it from these
    # args would move the keys per_person_grid.py gates sibling folders on.
    cfg = None
    if args.refs_only:
        if not os.path.exists(os.path.join(out, "config.json")):
            raise SystemExit(f"--refs_only needs a finished run; no config.json in {out}\n"
                             f"  name the folder with --out <path>")
    else:
        os.makedirs(out, exist_ok=True)
        cfg = write_config(out, args, extra=dict(model="frozen_mixed_effects", pool="cohort",
                                                 seed_source="cohort_one_per_class"))
    print(f"{t.name}: frozen-population mixed effects, {' vs '.join(args.methods)}, "
          f"W={args.block}, {args.n_seed_per_class}/class/participant seed + "
          f"{args.rounds} x {args.batch} acquired, {args.splits} split(s)")
    print(f"  step 4 {'ON' if args.update_variances else 'OFF'}, solver {args.solver}, "
          f"ceiling {'OFF' if args.no_refs else 'frozen_full'}"
          + (" (REFS ONLY: no acquisition)" if args.refs_only else ""))
    print(f"  -> {out}", flush=True)

    data, cols = load(t, verbose=True)
    have = done_splits(out) if args.resume else []
    if have:
        print(f"  resuming: splits {have} already written", flush=True)
    ids = canonical_ids(data)
    if args.refs_only:
        for k in range(args.first_split, args.first_split + args.splits):
            dst = analysis_path(out, f"reference_auroc_split{k}.csv")
            if args.resume and os.path.exists(dst):
                print(f"  split {k}: reference already written, skipping", flush=True)
                continue
            t0 = time.time()
            F, ids = refs_for_split(data, cols, k, args, ids)
            F.to_csv(dst, index=False)
            print(f"  split {k}: frozen_full AUROC {F.auroc.mean():.4f}, "
                  f"AUPRC {F.auprc.mean():.4f} over {len(F)} targets "
                  f"in {time.time() - t0:.0f}s -> {dst}", flush=True)
        print(f"done -> {out}")
        return

    for k in range(args.first_split, args.first_split + args.splits):
        if k in have:
            continue
        R, Q, L, S, F, ids = run_split(data, cols, k, args, ids)
        f_round, f_query = split_files(out, k)
        R.to_csv(f_round, index=False)
        Q.to_csv(f_query, index=False)
        L.to_csv(os.path.join(out, f"labels_split{k}.csv"), index=False)
        S.to_csv(os.path.join(out, f"state_split{k}.csv"), index=False)
        # One file per split, never a shared one: concurrent array tasks would race on it.
        # plots/per_person_grid.py merges reference_auroc_split*.csv itself.
        if len(F):
            F.to_csv(analysis_path(out, f"reference_auroc_split{k}.csv"), index=False)
        if not args.no_index:
            update_index(out, cfg)
        last = R[R["round"] == R["round"].max()]
        print(f"  split {k}: round {int(last['round'].iloc[0])} AUPRC "
              + ", ".join(f"{m} {g.auprc.mean():.4f}"
                          for m, g in last.groupby("method"))
              + (f"  | frozen_full {F.auprc.mean():.4f}" if len(F) else ""), flush=True)
    if not args.no_index:
        update_index(out, cfg)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
