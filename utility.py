"""Model-side helpers: the mixed-effects classifier and everything the AL loop needs around it.

Acquisition functions are NOT here -- scikit-activeml supplies those. What is here is the
estimator they are evaluated against, plus the small pieces of bookkeeping an active-learning
run needs: seed sets, random-slope selection, scoring.

The model
---------
A Bayesian mixed-effects logistic regression (``statsmodels.BinomialBayesMixedGLM``, MAP fit)
with

    logit P(y=1) = b0 + x'B  +  u_p  +  sum_k v_pk x_k

i.e. population-level fixed effects shared by everyone, a per-participant random intercept
``u_p``, and per-participant random slopes ``v_pk`` on a few selected features. That is the
point of the model: a personalised prediction for participant *p* that still borrows strength
from everybody else's labels, which is exactly what makes "whose labels did we buy" a meaningful
question.

Two warnings carried over from the original implementation, both learned the hard way:

* Use ``fit_map()``, never ``fit_vb()``. On the small labelled sets an AL round produces, the
  variational fit drives its own variance components upward without bound -- bigger allowance
  raises the likelihood, which licenses a bigger allowance -- and diverges (gradients ~1e14,
  negative variance SDs).
* Tighten the priors. statsmodels' defaults (``fe_p=2``, ``vcp_p=1``) are permissive enough to
  do the same thing when hundreds of parameters are fitted on tens of rows. ``fe_p=1.0``,
  ``vcp_p=0.5`` are the values the existing COMPUTER_WORK results use.

The scikit-activeml wrapper
---------------------------
:class:`MixedEffectsClassifier` presents that model through scikit-activeml's
``SkactivemlClassifier`` interface, so it can be handed to any strategy that wants an estimator
and used for evaluation each round. The one wrinkle: the model needs to know WHICH PARTICIPANT
each row belongs to, which a plain ``(X, y)`` interface has nowhere to put. The participant
index therefore travels as the last column of ``X`` and is split off inside ``fit``/
``predict_proba`` -- see :func:`attach_person` and keep that column away from any acquisition
function, which should see the features only.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

try:                                             # the wrapper is optional for the plain helpers
    from skactiveml.base import SkactivemlClassifier
    from skactiveml.utils import MISSING_LABEL, is_labeled
except ImportError:                              # pragma: no cover
    SkactivemlClassifier, MISSING_LABEL = object, np.nan

    def is_labeled(y, missing_label=np.nan):
        return ~pd.isna(np.asarray(y, dtype=float))

RIDGE_PERSONAL = 1e-3        # personal GLMs are ridge-penalised; one participant separates easily
FE_P, VCP_P = 1.0, 0.5       # fixed-effect prior SD; prior SD on the LOG of each random-effect SD


# ----------------------------------------------------------------------------- design frames
def feature_names(n_feat):
    """Formula-safe names. statsmodels parses a formula, so columns cannot be called 'audio:x'."""
    return [f"f{i}" for i in range(n_feat)]


def frame(Z, y, person):
    """Design frame with f0..fn, the label, and a `person` factor."""
    df = pd.DataFrame(np.asarray(Z, float), columns=feature_names(np.shape(Z)[1]))
    df["y"] = np.asarray(y, float)
    df["person"] = list(person)
    return df


def vc_formulas(slope_idx):
    """Variance components: a random intercept per person, plus a random slope per chosen feature.

    ``"0 + C(person)"`` means "one coefficient per participant, no shared intercept" -- the shared
    part is already in the fixed effects.
    """
    vc = {"person": "0 + C(person)"}
    vc.update({f"slope_f{k}": f"0 + C(person):f{k}" for k in slope_idx})
    return vc


# ----------------------------------------------------------------------------- fits
def fit_global(df, feats):
    """Pooled logistic regression: one model for everybody, no random effects."""
    mod = sm.GLM.from_formula("y ~ " + " + ".join(feats), data=df, family=sm.families.Binomial())
    try:
        return mod.fit()
    except Exception as e:                       # separation on wide feature sets
        print(f"    global GLM.fit() failed ({type(e).__name__}); ridge fit_regularized instead")
        return mod.fit_regularized(alpha=1e-4, L1_wt=0.0)


def fit_personal(df, feats, person):
    """Ridge logistic regression on ONE participant's rows, or None if they lack both classes."""
    d = df[df.person == person]
    if d.y.nunique() < 2:
        return None
    mod = sm.GLM.from_formula("y ~ " + " + ".join(feats), data=d, family=sm.families.Binomial())
    try:
        return mod.fit_regularized(alpha=RIDGE_PERSONAL, L1_wt=0.0)
    except Exception:
        return None


def fit_mixed(df, feats, vc, vcp_p=VCP_P, fe_p=FE_P, scale_fe=False):
    """MAP fit of the mixed-effects logistic model. See the module docstring for why MAP."""
    mod = BinomialBayesMixedGLM.from_formula("y ~ " + " + ".join(feats), vc, df,
                                             vcp_p=vcp_p, fe_p=fe_p)
    return mod.fit_map(scale_fe=scale_fe)


# ----------------------------------------------------------------------------- reading a fit
def parse_vc(res):
    """-> {(component, person): posterior mean}. Component is 'intercept' or 'f<k>'.

    statsmodels names these after the factor expression, not the vc_formulas key:
    ``C(person)[M4]`` for an intercept and ``C(person)[M4]:f11`` for a slope. Parsing on the key
    silently returns zeros, which looks like a model with no random effects at all.
    """
    names = list(getattr(res.model, "vc_names", []) or [])
    out = {}
    for nm, val in zip(names, np.asarray(res.vc_mean, dtype=float)):
        i, j = nm.find("["), nm.find("]")
        pid = nm[i + 1:j] if i >= 0 else nm
        tail = nm[j + 1:]
        out[(tail[1:] if tail.startswith(":") else "intercept", pid)] = float(val)
    return out


def logit_fixed(res, Z):
    """Population-level part of the linear predictor."""
    fe = np.asarray(res.fe_mean, dtype=float)
    return fe[0] + np.asarray(Z, float) @ fe[1:]


def random_part(vc, person, Z, slope_idx=()):
    """Participant-specific part: u_p + sum_k v_pk x_k. Zero for an unseen participant."""
    Z = np.asarray(Z, float)
    v = np.full(len(Z), vc.get(("intercept", person), 0.0))
    for k in slope_idx:
        v = v + vc.get((f"f{k}", person), 0.0) * Z[:, k]
    return v


def predict_logit(res, Z, person, slope_idx=()):
    """Full linear predictor for one participant's rows."""
    return logit_fixed(res, Z) + random_part(parse_vc(res), person, Z, slope_idx)


# ----------------------------------------------------------------------------- slope selection
def between_person_var(coefs, n_feat):
    """Variance across participants of each feature's personal-GLM coefficient.

    A random slope models heterogeneity in a feature's EFFECT, which an |average coefficient|
    ranking cannot see: a feature the pooled model leans on hard may carry the same slope for
    everyone, and is then wasted as a random slope.
    """
    B = np.vstack(list(coefs.values()))
    assert B.shape[1] == n_feat, f"personal GLM has {B.shape[1]} coefficients, expected {n_feat}"
    return B.var(axis=0, ddof=1)


def select_slopes(Z, y, person, n_slopes=5, cols=None, verbose=False):
    """Indices of the `n_slopes` features whose effect varies most between participants.

    Fitted on the FULL training rows of the split, once, before the AL loop -- not on the
    acquired labels -- so the random-effect structure is fixed across rounds and the comparison
    between acquisition strategies is not confounded by a moving model specification.
    """
    feats = feature_names(Z.shape[1])
    df = frame(Z, y, person)
    coefs = {}
    for p in sorted(set(person)):
        r = fit_personal(df, feats, p)
        if r is not None:
            coefs[p] = np.asarray(r.params, float)[1:]
    if len(coefs) < 2:
        raise RuntimeError("random-slope selection needs two usable personal fits")
    order = [int(i) for i in np.argsort(-between_person_var(coefs, Z.shape[1]))]
    idx = order[:n_slopes]
    if verbose and cols is not None:
        print(f"  {len(idx)} random slopes from {len(coefs)} personal fits: "
              f"{', '.join(cols[i] for i in idx)}")
    return idx


# ----------------------------------------------------------------------------- seed sets
def seed_set(y, day, rng, n_seed=3, n_seed_pos=1):
    """`n_seed` of one participant's own labels: `n_seed_pos` positives, the rest negatives.

    Spread over distinct days first, then filled at random. Starting every target from a seed
    containing at least one positive is what makes round 0 comparable across participants --
    AUROC is undefined for an all-negative labelled set.

    -> (indices into the participant's rows, flag set when they had too few positives)
    """
    y = np.asarray(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    flagged = len(pos) < n_seed_pos
    take = []
    n_pos = min(n_seed_pos, len(pos))
    for idx, n in ((pos, n_pos), (neg, n_seed - n_pos)):
        if n <= 0 or len(idx) == 0:
            continue
        dd = np.asarray(day)[idx]
        chosen = []
        for d in rng.permutation(np.unique(dd)):          # one per day first
            chosen.append(rng.choice(idx[dd == d]))
            if len(chosen) >= min(n, len(np.unique(dd))):
                break
        rest = np.setdiff1d(idx, chosen)
        if n > len(chosen) and len(rest):
            chosen += list(rng.choice(rest, min(n - len(chosen), len(rest)), replace=False))
        take += list(chosen)
    return np.array(sorted(set(take)), dtype=int), flagged


# ----------------------------------------------------------------------------- scoring
def scores(y, logit):
    """AUROC / AUPRC from a linear predictor. NaN when the test rows are single-class."""
    y = np.asarray(y)
    if not 0 < y.sum() < len(y):
        return np.nan, np.nan
    p = np.clip(expit(logit), 1e-9, 1 - 1e-9)
    return roc_auc_score(y, logit), average_precision_score(y, p)


# ----------------------------------------------------------------------------- skactiveml wrapper
def attach_person(Z, person, order):
    """Append the participant index as a last column, for the classifier only.

    scikit-activeml's estimator interface is ``fit(X, y)``, with no room for a grouping vector,
    so the group rides along inside X. `order` fixes the code for each participant; pass the
    same list to :class:`MixedEffectsClassifier` so the codes agree.

    Acquisition strategies must be given the features WITHOUT this column -- a participant index
    is not a coordinate anything should measure distances in.
    """
    code = {p: i for i, p in enumerate(order)}
    return np.column_stack([np.asarray(Z, float),
                            np.array([code[p] for p in person], float)])


class MixedEffectsClassifier(SkactivemlClassifier):
    """The mixed-effects model behind scikit-activeml's classifier interface.

    Parameters
    ----------
    persons : list
        Participant ids, in the order their codes were assigned by :func:`attach_person`.
    slope_idx : sequence of int
        Feature indices carrying a random slope.
    fe_p, vcp_p : float
        Prior SDs. Defaults are the tightened values; see the module docstring.
    classes, missing_label, random_state
        Standard scikit-activeml arguments.

    ``X`` must carry the participant code as its LAST column (:func:`attach_person`). ``y`` uses
    ``missing_label`` (NaN by default) for unlabelled rows, which is scikit-activeml's
    convention: the pool is one array, and acquiring a label fills a slot rather than moving a
    row between two sets.

    When the labelled rows carry only one class -- which happens at round 0 for a target whose
    seed set found no positive -- no model is identified, and ``predict_proba`` falls back to the
    labelled base rate rather than raising. ``self.fit_failed_`` records this so a run can count
    it instead of silently averaging a degenerate round into its curve.
    """

    def __init__(self, persons, slope_idx=(), fe_p=FE_P, vcp_p=VCP_P, classes=None,
                 missing_label=MISSING_LABEL, cost_matrix=None, random_state=None):
        super().__init__(classes=classes, missing_label=missing_label, cost_matrix=cost_matrix,
                         random_state=random_state)
        # stored verbatim: sklearn's clone() (used by scikit-activeml strategies that refit
        # the model, e.g. expected error reduction) requires get_params() to return exactly
        # what was passed, so no list()/copy conversions here -- do them where they are used
        self.persons = persons
        self.slope_idx = slope_idx
        self.fe_p, self.vcp_p = fe_p, vcp_p

    # -- internals -----------------------------------------------------------
    def _split(self, X):
        X = np.asarray(X, float)
        persons = list(self.persons)
        return X[:, :-1], [persons[int(c)] for c in X[:, -1]]

    def fit(self, X, y, sample_weight=None):
        Z, person = self._split(X)
        y = np.asarray(y, float)
        lab = is_labeled(y, missing_label=self.missing_label_ if hasattr(self, "missing_label_")
                         else self.missing_label)
        self.classes_ = np.array([0.0, 1.0]) if self.classes is None else np.asarray(self.classes)
        self.n_features_in_ = X.shape[1]
        self.fit_failed_, self.res_, self.vc_ = False, None, {}
        yl = y[lab]
        self.base_rate_ = float(yl.mean()) if len(yl) else 0.5
        if len(yl) == 0 or len(np.unique(yl)) < 2:
            self.fit_failed_ = True                     # single-class: nothing is identified
            return self
        df = frame(Z[lab], yl, np.asarray(person, dtype=object)[lab])
        feats = feature_names(Z.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.res_ = fit_mixed(df, feats, vc_formulas(list(self.slope_idx)),
                                      vcp_p=self.vcp_p, fe_p=self.fe_p)
                self.vc_ = parse_vc(self.res_)
            except Exception:
                self.fit_failed_ = True
        return self

    def decision_function(self, X):
        """Linear predictor -- what AUROC should be computed on (monotone in the probability)."""
        Z, person = self._split(X)
        if self.fit_failed_ or self.res_ is None:
            return np.full(len(Z), float(np.log(max(self.base_rate_, 1e-9)
                                                / max(1 - self.base_rate_, 1e-9))))
        lo = logit_fixed(self.res_, Z)
        for p in set(person):
            m = np.array([q == p for q in person])
            lo[m] = lo[m] + random_part(self.vc_, p, Z[m], list(self.slope_idx))
        return lo

    def predict_proba(self, X):
        p = expit(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(float)
