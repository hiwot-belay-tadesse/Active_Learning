"""The two-phase acquisition: Target Calibration (Phase 1), then Target-Weighted Fixed-Effect Gain.

Both phases read the SAME object -- the Laplace posterior of the mixed-effects logistic model,
``Sigma = (Xt' W Xt + Lam)^-1`` over theta = (beta, b) with Xt = [X, Z] -- and differ only in
which part of it they are trying to shrink and over which candidates. That is the whole design:
one posterior, two objectives, a schedule between them.

    Phase 1  D-optimality on the target's random-effect block b_T, over the TARGET's own blocks.
    Phase 2  A-optimality on the FIXED effects beta, measured under the target's density, over
             the WHOLE pool.

Why a schedule and not one criterion
------------------------------------
In a GLMM the target's prediction is beta-driven plus b_T. A stranger's label carries no
information about b_T at all (it enters only through beta), so while b_T is still essentially at
its prior the personalised prediction is mis-centred no matter how good beta gets -- and a
criterion that mixes the two objectives spends its first labels trading a large, cheap gain in
b_T against a diffuse gain in beta. Phase 1 therefore buys b_T outright from the only source
that has it, in a number of queries set by its dimension (K+1 random-effect coordinates: ~5-10
labels at K=5), and Phase 2 then spends the rest of the budget where the pool is large and
labels are cheap, which is everyone else's blocks.

Phase 1: D-optimality on b_T
----------------------------
The target's random-effect coordinates are a sub-block of the posterior. Labelling a candidate c
adds w_c xt_c xt_c' to the precision, so by Sherman-Morrison

    Sigma' = Sigma - gamma_c sigma_c sigma_c',    sigma_c = Sigma xt_c,
             gamma_c = w_c / (1 + w_c xt_c' Sigma xt_c)

and, restricting to the target's block B (rows/cols of b_T), det Sigma'_BB factorises exactly:

    det Sigma'_BB = det Sigma_BB * (1 - gamma_c a_c' Sigma_BB^-1 a_c),   a_c = sigma_c[B]

so maximising the D-optimal gain -log det Sigma'_BB + log det Sigma_BB is *exactly* maximising

    U_dopt(c) = gamma_c * a_c' Sigma_BB^-1 a_c                                      (Phase 1)

-- a closed form, no label simulation, no refit. This is the posterior version of D-optimality
on the random-effects design matrix Z_T: the classical criterion maximises det(Z_T' W Z_T + Lam_b)
over designs, which is the same objective with the cross-information between b_T and beta set to
zero. Keeping the cross term is free here and is what makes a seed set's borrowed strength count.

Candidates are restricted to the target's own unlabelled blocks, which is not a heuristic: rows
owned by anyone else have a structural zero in the target's random-effect columns, so a_c = 0
and U_dopt(c) = 0 identically. The restriction is the criterion, made explicit.

``criterion="kmedoids"`` is the geometric alternative the proposal contrasts: greedy
facility-location medoids of the target's own blocks (minimise the mean distance from every
target block to its nearest selected block). It anchors b_T from representative rather than
informative points, needs no model, and is the honest "do you need the posterior for this?"
ablation.

Phase 2: Target-Weighted Fixed-Effect Gain
------------------------------------------
Let x~ = [1, x, 0...0] be the design row that uses the FIXED effects only, and let pi be an
evaluation measure on feature space concentrated on the target's operational manifold --
pi ∝ p_T(x)/p_pool(x) over the pooled blocks (see ``density_ratio.py``). Define

    A_T = sum_e pi_e x~_e x~_e'        (the target-weighted second moment, (1+p) x (1+p))

The quantity Phase 2 reduces is the pi-weighted posterior variance of the population-level
predictor, tr(A_T Sigma_beta,beta). One label's reduction is again exact:

    U_twfeg(c) = w_c * (xt_c' Sigma_.beta A_T Sigma_beta. xt_c) / (1 + w_c xt_c' Sigma xt_c)
               = w_c * sum_e pi_e (x~_e' Sigma xt_c)^2 / (1 + w_c xt_c' Sigma xt_c)      (Phase 2)

Three things follow from the form, and they are the reasons to prefer it to the geometric
criteria it is benchmarked against:

* **It is population-level by construction.** The evaluation rows x~_e carry zeros in the random
  block, so nothing in U_twfeg rewards shrinking b of any individual -- including the target's.
  Phase 1 has already dealt with b_T; Phase 2 cannot be diverted into re-buying it.
* **It is target-weighted, not target-restricted.** The cross term xt_c' Sigma_.beta couples a
  stranger's candidate to the target's manifold through the shared beta, so a block owned by
  someone else is valued exactly insofar as it informs beta *where the target operates*. Pool
  size is an asset rather than a dilution.
* **It cannot be bought by an outlier.** A candidate far from the target's manifold has a small
  A_T-weighted cross term however unusual it is; a noisy or anomalous row carries a small w_c
  only if the model already predicts it confidently, but its geometric distinctiveness -- the
  thing CoreSet and ProbCover reward -- enters nowhere. The denominator additionally discounts
  candidates whose own predictive variance is enormous, which is the self-limiting behaviour an
  extrapolation point should have.

Implementation notes
--------------------
* Both phases score the whole candidate set in closed form (one Sigma per round, then matrix
  products), so neither needs ``SubSamplingWrapper``. A_T does not depend on the round and is
  factored once per target; the within-batch greedy update is rank-one on Sigma and on the
  pre-multiplied ``Sigma Xt'``, so a batch of 3 costs one full product and two cheap updates.
* W is evaluated at the current mode and the mode is refitted only between rounds, exactly as in
  the parent repo's ``laplace_variance``. Candidates keep their OWNER's participant code -- the
  provenance term is the point, and re-coding every candidate as the target would delete it.
* When the model could not be fitted (a single-class labelled set, which happens at round 0 for
  a target whose seed found no positive) there is no posterior; utilities fall back to uniform so
  the round degenerates to random rather than raising. Runs count those rounds.
* A participant with no labels yet has no random-effect columns in the current fit, so their
  rows are scored on the fixed-effect term alone. For Phase 2 that is not an approximation to
  apologise for -- the fixed-effect term is the entire criterion.
"""
import os
import sys

import numpy as np
from skactiveml.base import SingleAnnotatorPoolQueryStrategy
from skactiveml.utils import MISSING_LABEL, is_labeled, simple_batch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from laplace_variance import _blocks, laplace_cov, stack_design, vc_spec   # noqa: E402

__all__ = ["TargetCalibration", "TWFEG", "TwoStage", "target_block", "chol_psd",
           "singleton_attenuation"]


# --------------------------------------------------------------------------- reading the fit
def target_block(res, target):
    """-> column indices (into the stacked design) of the random-effect block of one participant.

    Empty when the participant has no random-effect columns in this fit, which is what happens
    before anyone has bought them a label.
    """
    k_fep = res.model.k_fep
    return np.array([k_fep + j for j, (_, person) in enumerate(vc_spec(res))
                     if person == target], dtype=int)


def singleton_attenuation(Z_cand, who, clf, res, w):
    """-> effective weights w_eff, discounting a NEW participant's first label as beta-information.

    The problem it fixes. ``stack_design`` gives a candidate whose owner holds no labels yet a
    zero random-effect block, because that participant has no random-effect columns in the current
    fit. The criterion then treats their observation as if it were a pure fixed-effect
    observation -- which **overstates** its value, and overstates it in exactly the direction that
    matters here. In a GLMM the new participant's own random effect
    $b_c \sim \mathcal{N}(0, D)$ will absorb part of that observation's residual, so a singleton
    label from a stranger says much less about $\boldsymbol{\beta}$ than the same features
    recorded by someone whose random effect is already pinned down.

    The correction is exact for a group with no prior labels. Marginalising $b_c$ out of the joint
    information of one observation,

        I = [[ w x x',  w x z' ], [ w z x',  w z z' + D^-1 ]],

    the Schur complement in the beta block is, by Sherman-Morrison,

        I_beta|marg = w x x' ( 1 - w z'(w z z' + D^-1)^-1 z ) = x x' * w / (1 + w z' D z),

    so the observation behaves like one with weight ``w / (1 + w z' D z)`` rather than ``w``.
    Here ``z = [1, x_k1, ..., x_kK]`` is the candidate's random-effect design row and
    ``D = diag(s^2)`` holds the fitted random-effect SDs. The factor is 1 when the random effects
    are tight (D -> 0) and tends to 0 as they grow -- i.e. the more individual variation the model
    has estimated, the less a stranger's single label can say about the population.

    The returned weights equal ``w`` unchanged for candidates whose owner already has
    random-effect columns in the fit: their partial identification is already represented in
    ``Sigma``, so applying this on top would double-count it.
    """
    spec = vc_spec(res)
    w = np.asarray(w, float)
    if not spec:
        return w
    _, _, _, sd = _blocks(res)
    comp_sd = {}
    for j, (comp, _person) in enumerate(spec):
        comp_sd.setdefault(comp, float(sd[j]))
    known = {person for _comp, person in spec}
    slope_idx = list(getattr(clf, "slope_idx", ()) or ())
    Z_cand = np.asarray(Z_cand, float)
    z = np.column_stack([np.ones(len(Z_cand))] + [Z_cand[:, k] for k in slope_idx])
    d = np.array([comp_sd.get("intercept", 0.0) ** 2]
                 + [comp_sd.get(f"f{k}", 0.0) ** 2 for k in slope_idx])
    zDz = (z ** 2 * d).sum(axis=1)
    is_new = np.array([p not in known for p in who])
    return np.where(is_new, w / (1.0 + w * zDz), w)


def chol_psd(A, eps=1e-10):
    """Cholesky factor L with A = L L', repairing a numerically indefinite A. -> L.

    A_T is a weighted second-moment matrix, hence PSD in exact arithmetic, but with 124
    standardised features and a heavily concentrated weight vector its smallest eigenvalues sit
    at round-off. Eigenvalue clipping is used rather than a growing ridge so the repair is
    bounded and visible.
    """
    A = 0.5 * (np.asarray(A, float) + np.asarray(A, float).T)
    try:
        return np.linalg.cholesky(A + eps * np.eye(len(A)))
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(A)
        return V * np.sqrt(np.clip(w, eps, None))


# --------------------------------------------------------------------------- the shared greedy
def _weights(Xp, clf):
    """w_c = mu_c (1 - mu_c) at the current posterior mode."""
    p = np.asarray(clf.predict_proba(Xp), float)[:, 1]
    return np.clip(p * (1.0 - p), 1e-12, None)


def _cross_dopt(SX, bidx, Sigma):
    """a_c' Sigma_BB^-1 a_c for every candidate, with a_c = (Sigma xt_c)[B]."""
    A = SX[bidx, :]                                     # (|B|, n_cand)
    bb = Sigma[np.ix_(bidx, bidx)]
    try:
        sol = np.linalg.solve(bb, A)
    except np.linalg.LinAlgError:
        sol = np.linalg.pinv(bb) @ A
    return np.einsum("ij,ij->j", sol, A)


def _cross_twfeg(SX, k_fep, L):
    """sum_e pi_e (x~_e' Sigma xt_c)^2 = || L' (Sigma xt_c)[beta] ||^2 for every candidate."""
    return ((L.T @ SX[:k_fep, :]) ** 2).sum(axis=0)


def _greedy(Sigma, Xt, w, cross_fn, batch_size, rng):
    """Greedy batch under a rank-one posterior update. -> (picks, utility rows).

    ``cross_fn(SX, Sigma)`` returns the criterion's numerator for every candidate; the
    denominator ``1 + w q`` and the Sherman-Morrison bookkeeping are shared. ``SX = Sigma Xt'``
    and ``q = diag(Xt Sigma Xt')`` are maintained incrementally: after picking b,

        Sigma <- Sigma - g_b s_b s_b',   SX <- SX - g_b outer(s_b, Xt s_b),  q <- q - g_b (Xt s_b)^2

    which is what keeps a batch from costing a full (d x n) product per pick.
    """
    S = np.array(Sigma, float, copy=True)
    SX = S @ Xt.T                                       # (d, n_cand)
    q = np.einsum("ij,ji->i", Xt, SX)
    picked, rows = [], []
    for _ in range(batch_size):
        util = w * cross_fn(SX, S) / (1.0 + w * q)
        util[picked] = -np.inf
        rows.append(np.where(np.isfinite(util), util, np.nan))
        finite = np.where(np.isfinite(util), util, np.nan)
        if np.all(np.isnan(finite)):                    # nothing left to pick
            break
        top = np.flatnonzero(finite == np.nanmax(finite))
        b = int(rng.choice(top))                        # ties broken at random, not by index
        picked.append(b)
        s_b, g_b = SX[:, b], w[b] / (1.0 + w[b] * q[b])
        t = Xt @ s_b
        S -= g_b * np.outer(s_b, s_b)
        SX -= g_b * np.outer(s_b, t)
        q -= g_b * t ** 2
    return picked, rows


def _fallback(n_cand, mapping, n_total, rng, batch_size, return_utilities):
    """Uniform utilities -- used when there is no posterior to work with."""
    u = np.zeros(n_cand) if mapping is None else _scatter(np.zeros(n_cand), mapping, n_total)
    return simple_batch(u, rng, batch_size=batch_size, return_utilities=return_utilities)


def _scatter(v, mapping, n):
    out = np.full(n, np.nan)
    out[mapping] = v
    return out


def _design(Xp, clf, res):
    """Split the participant code off the last column and stack the full design."""
    Xp = np.asarray(Xp, float)
    persons = list(clf.persons)
    who = [persons[int(c)] for c in Xp[:, -1]]
    return stack_design(Xp[:, :-1], who, res)


# --------------------------------------------------------------------------- Phase 1
class TargetCalibration(SingleAnnotatorPoolQueryStrategy):
    """Phase 1: anchor the target's random effects b_T from the target's own blocks.

    Parameters
    ----------
    criterion : {"dopt", "kmedoids"}, default="dopt"
        ``"dopt"``     D-optimality on the posterior block of b_T (module docstring).
        ``"kmedoids"`` greedy facility-location medoids of the target's own blocks -- model-free.
    cov_method, jitter
        Passed to :func:`laplace_variance.laplace_cov`.
    missing_label, random_state
        Standard scikit-activeml arguments.

    ``query`` takes ``target`` (the participant id) and expects ``candidates`` to be the target's
    own unlabelled row indices. It does not silently widen the candidate set: a caller that
    hands it the whole pool gets utilities that are zero off the target's rows, which is correct
    but wasteful.
    """

    def __init__(self, criterion="dopt", cov_method="hessian", jitter=0.0,
                 missing_label=MISSING_LABEL, random_state=None):
        super().__init__(missing_label=missing_label, random_state=random_state)
        self.criterion = criterion
        self.cov_method = cov_method
        self.jitter = jitter

    def query(self, X, y, clf=None, target=None, candidates=None, batch_size=1,
              return_utilities=False):
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True)
        X_cand, mapping = self._transform_candidates(candidates, X, y)
        n_total = len(X)
        if self.criterion not in ("dopt", "kmedoids"):
            raise ValueError(f"criterion must be 'dopt' or 'kmedoids', got {self.criterion!r}")

        if self.criterion == "kmedoids":
            lab = is_labeled(y, missing_label=self.missing_label_)
            u = self._medoid_utilities(X, X_cand, mapping, lab, target, clf)
            util = u if mapping is None else _scatter(u, mapping, n_total)
            return simple_batch(util, self.random_state_, batch_size=batch_size,
                                return_utilities=return_utilities)

        res = getattr(clf, "res_", None)
        if clf is None or getattr(clf, "fit_failed_", False) or res is None:
            return _fallback(len(X_cand), mapping, n_total, self.random_state_, batch_size,
                             return_utilities)
        bidx = target_block(res, target)
        if len(bidx) == 0:      # nobody has bought the target a label yet: no block to calibrate
            return _fallback(len(X_cand), mapping, n_total, self.random_state_, batch_size,
                             return_utilities)
        Sigma = laplace_cov(res, method=self.cov_method, jitter=self.jitter)
        Xt = _design(X_cand, clf, res)
        w = _weights(X_cand, clf)
        picks, rows = _greedy(Sigma, Xt, w, lambda SX, S: _cross_dopt(SX, bidx, S),
                              batch_size, self.random_state_)
        return self._finish(picks, rows, mapping, n_total, return_utilities)

    # -- helpers -------------------------------------------------------------
    def _medoid_utilities(self, X, X_cand, mapping, lab, target, clf):
        """Greedy facility-location gain over the target's own rows (model-free Phase 1).

        Utility of candidate c = how much it reduces the mean distance from each of the target's
        blocks to its nearest *selected* block, where the already-labelled target blocks count as
        selected. One pass, so a batch is the top-k of a single gain vector rather than a true
        greedy sequence -- within a batch of 3 on one participant's ~200 blocks the difference is
        negligible, and it keeps the ablation simple.
        """
        from scipy.spatial.distance import cdist
        persons = list(clf.persons) if clf is not None else None
        Zc = np.asarray(X_cand, float)[:, :-1]
        who = (np.array([persons[int(c)] for c in np.asarray(X, float)[:, -1]])
               if persons is not None else None)
        own = np.where(who == target)[0] if who is not None else np.arange(len(X))
        Zown = np.asarray(X, float)[own][:, :-1]
        sel = np.asarray(X, float)[own][lab[own]][:, :-1]
        D = cdist(Zown, Zc)                                       # (n_own, n_cand)
        base = (cdist(Zown, sel).min(axis=1) if len(sel)
                else np.full(len(Zown), np.inf))
        gain = np.maximum(base[:, None] - D, 0.0)
        g = gain.mean(axis=0) if np.isfinite(base).all() else (-D.mean(axis=0))
        return g

    @staticmethod
    def _finish(picks, rows, mapping, n_total, return_utilities):
        idx = np.asarray(picks if mapping is None else mapping[picks], dtype=int)
        if return_utilities:
            U = np.vstack([r if mapping is None else _scatter(r, mapping, n_total) for r in rows])
            return idx, U
        return idx


# --------------------------------------------------------------------------- Phase 2
class TWFEG(SingleAnnotatorPoolQueryStrategy):
    """Phase 2: Target-Weighted Fixed-Effect Gain over the whole multi-subject pool.

    Parameters
    ----------
    cov_method, jitter
        Passed to :func:`laplace_variance.laplace_cov`.
    missing_label, random_state
        Standard scikit-activeml arguments.

    marginalise : bool, default=False
        Discount a candidate owned by a participant who holds NO labels yet, by the exact factor
        for marginalising that participant's own random effect out of the observation
        (:func:`singleton_attenuation`). Off by default so that the criterion as proposed is what
        gets benchmarked; the refinement is its own arm.

    ``query`` needs ``A_chol`` -- the Cholesky factor of the target-weighted second moment A_T,
    built once per target by ``density_ratio.second_moment`` + :func:`chol_psd`. Passing the
    factor rather than the measure is deliberate: A_T is round-independent, and recomputing it
    per round would dominate the cost of the criterion it feeds.
    """

    def __init__(self, cov_method="hessian", jitter=0.0, marginalise=False,
                 missing_label=MISSING_LABEL, random_state=None):
        super().__init__(missing_label=missing_label, random_state=random_state)
        self.cov_method = cov_method
        self.jitter = jitter
        self.marginalise = marginalise

    def query(self, X, y, clf, A_chol=None, candidates=None, batch_size=1,
              return_utilities=False):
        X, y, candidates, batch_size, return_utilities = self._validate_data(
            X, y, candidates, batch_size, return_utilities, reset=True)
        X_cand, mapping = self._transform_candidates(candidates, X, y)
        n_total = len(X)
        res = getattr(clf, "res_", None)
        if getattr(clf, "fit_failed_", False) or res is None:
            return _fallback(len(X_cand), mapping, n_total, self.random_state_, batch_size,
                             return_utilities)
        if A_chol is None:
            raise ValueError("TWFEG.query needs A_chol: the Cholesky factor of A_T "
                             "(density_ratio.second_moment -> chol_psd)")
        k_fep = res.model.k_fep
        if A_chol.shape[0] != k_fep:
            raise ValueError(f"A_chol is {A_chol.shape[0]}x{A_chol.shape[1]} but the fit has "
                             f"{k_fep} fixed effects")
        Sigma = laplace_cov(res, method=self.cov_method, jitter=self.jitter)
        Xt = _design(X_cand, clf, res)
        w = _weights(X_cand, clf)
        if self.marginalise:
            persons = list(clf.persons)
            Xc = np.asarray(X_cand, float)
            who = [persons[int(c)] for c in Xc[:, -1]]
            w = singleton_attenuation(Xc[:, :-1], who, clf, res, w)
        picks, rows = _greedy(Sigma, Xt, w, lambda SX, S: _cross_twfeg(SX, k_fep, A_chol),
                              batch_size, self.random_state_)
        return TargetCalibration._finish(picks, rows, mapping, n_total, return_utilities)


# --------------------------------------------------------------------------- the framework
class TwoStage(SingleAnnotatorPoolQueryStrategy):
    """Phase 1 for the first ``n_calib`` labels, Phase 2 afterwards -- the proposed framework.

    Parameters
    ----------
    n_calib : int, default=6
        Labels spent on Target Calibration before switching. The natural scale is the dimension
        of b_T (one intercept + ``n_slopes`` slopes), so 5-10 at the repo's default of 5 slopes.
        ``0`` disables Phase 1 (pure TW-FEG, the ablation).
    calib : {"dopt", "kmedoids"}, default="dopt"
    cov_method, jitter, missing_label, random_state
        As above.

    ``query(..., n_acquired=k)`` is told how many labels this target has already bought, rather
    than inferring it from ``y``: the seed set is also in ``y``, and a run that changes ``n_seed``
    would otherwise silently change where the phase boundary falls. A batch that straddles the
    boundary is split between the phases.
    """

    def __init__(self, n_calib=6, calib="dopt", cov_method="hessian", jitter=0.0,
                 marginalise=False, missing_label=MISSING_LABEL, random_state=None):
        super().__init__(missing_label=missing_label, random_state=random_state)
        self.n_calib = n_calib
        self.calib = calib
        self.cov_method = cov_method
        self.jitter = jitter
        self.marginalise = marginalise

    def query(self, X, y, clf, target=None, own_candidates=None, A_chol=None, n_acquired=0,
              candidates=None, batch_size=1, return_utilities=False):
        """-> indices into ``X``.

        ``own_candidates`` are the target's own unlabelled rows (Phase 1's candidate set);
        ``candidates`` are the pool's unlabelled rows (Phase 2's). Both are indices into ``X``.
        """
        if return_utilities:
            raise NotImplementedError("TwoStage reports no combined utility surface: the two "
                                      "phases are measured in different units (log-det of b_T "
                                      "vs weighted variance of beta) and stacking them would "
                                      "invite exactly the comparison the schedule avoids")
        rs = self._validate_data(X, y, candidates, batch_size, False, reset=True)[2:]
        candidates, batch_size = rs[0], rs[1]
        n1 = int(np.clip(self.n_calib - int(n_acquired), 0, batch_size))
        if own_candidates is not None:
            n1 = min(n1, len(own_candidates))
        out = []
        if n1 > 0:
            qs = TargetCalibration(criterion=self.calib, cov_method=self.cov_method,
                                   jitter=self.jitter, missing_label=self.missing_label,
                                   random_state=self.random_state_)
            out += list(qs.query(X=X, y=y, clf=clf, target=target,
                                 candidates=np.asarray(own_candidates, int), batch_size=n1))
        n2 = batch_size - len(out)
        if n2 > 0:
            cand2 = np.asarray(candidates, int) if candidates is not None else None
            if cand2 is not None and len(out):
                cand2 = np.setdiff1d(cand2, np.asarray(out, int))
            qs = TWFEG(cov_method=self.cov_method, jitter=self.jitter,
                       marginalise=self.marginalise, missing_label=self.missing_label,
                       random_state=self.random_state_)
            out += list(qs.query(X=X, y=y, clf=clf, A_chol=A_chol, candidates=cand2,
                                 batch_size=n2))
        return np.asarray(out, dtype=int)
