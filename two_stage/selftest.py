"""Verify the algebra of both phases against explicit re-inversion. Not the science -- the maths.

Four things can break silently here while still producing plausible-looking rankings, and all
four are checkable without an oracle:

  1. **Phase 1 is D-optimality.** The claimed identity
     ``log det Sigma_BB - log det Sigma'_BB == -log(1 - gamma a' Sigma_BB^-1 a)`` must hold
     against an explicit ``(H + w xt xt')^-1``, where B is the target's random-effect block. If
     the block indices are wrong (e.g. picking another participant's columns, which the
     statsmodels naming makes easy) the criterion still returns numbers, just not these ones.
  2. **Phase 2 is the weighted fixed-effect variance drop.** ``U_twfeg`` must equal
     ``tr(A_T Sigma_bb,beta) - tr(A_T Sigma'_beta,beta)`` exactly.
  3. **The incremental greedy bookkeeping is exact.** ``SX`` and ``q`` are updated rank-one
     inside a batch rather than recomputed; that must agree with recomputation to round-off.
  3b. **The singleton attenuation is the right Schur complement.** The factor applied to a
     stranger's first label must equal what marginalising that stranger's random effect out of
     the observation does to its beta-information.
  4. **The schedule routes.** The first ``n_calib`` labels must all come from the target's own
     rows, and the rest must be free to come from anywhere.

What this does NOT check: whether the Laplace Gaussian approximates the true posterior well, or
whether shrinking either quantity improves Macro PR-AUC. Those are what the experiment is for.

    python two_stage/selftest.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from acquisitions import (TWFEG, TargetCalibration, TwoStage, _cross_dopt,  # noqa: E402
                          _cross_twfeg, _greedy, chol_psd, target_block)
from density_ratio import eval_weights, second_moment                        # noqa: E402
from laplace_variance import _blocks as _blocks_of                           # noqa: E402
from laplace_variance import laplace_cov                                     # noqa: E402
from utility import MixedEffectsClassifier, attach_person                    # noqa: E402

TOL = 1e-7


def _fixture(seed=0, n=400, G=6, n_feat=4, slopes=(0, 1)):
    """A small pooled GLMM problem, fitted exactly as the AL loop fits it."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, G, n)
    Z = rng.normal(size=(n, n_feat))
    b0 = rng.normal(0, 0.8, G)
    eta = -0.3 + Z @ np.array([1.2, -0.7, 0.4, 0.0]) + b0[g]
    y = rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(float)
    persons = [f"P{i}" for i in range(G)]
    owner = np.array([f"P{i}" for i in g])
    X = attach_person(Z, owner, persons)
    clf = MixedEffectsClassifier(persons=persons, slope_idx=list(slopes), classes=[0.0, 1.0],
                                 missing_label=np.nan)
    clf.fit(X, y)
    return dict(Z=Z, y=y, X=X, owner=owner, persons=persons, clf=clf, rng=rng,
                target="P0", slopes=list(slopes))


def main():
    f = _fixture()
    clf, res, X, Z, owner, target = f["clf"], f["clf"].res_, f["X"], f["Z"], f["owner"], f["target"]
    assert not clf.fit_failed_, "fixture fit failed"
    k_fep = res.model.k_fep
    Sigma = laplace_cov(res)
    H = np.linalg.inv(Sigma)
    bidx = target_block(res, target)
    print(f"fit: {k_fep} fixed + {Sigma.shape[0] - k_fep} random columns; "
          f"b_{target} block = {list(bidx)}")
    assert len(bidx) == len(f["slopes"]) + 1, "target block should be intercept + one per slope"

    from acquisitions import _design, _weights
    Xt = _design(X, clf, res)
    w = _weights(X, clf)
    pi = eval_weights(Z, owner, target, measure="ratio", seed=0)
    L = chol_psd(second_moment(Z, pi))
    A_T = L @ L.T

    # --- 1 & 2: both criteria against explicit re-inversion -----------------
    worst_d, worst_t = 0.0, 0.0
    SX = Sigma @ Xt.T
    q = np.einsum("ij,ji->i", Xt, SX)
    u_dopt = w * _cross_dopt(SX, bidx, Sigma) / (1 + w * q)
    u_twfeg = w * _cross_twfeg(SX, k_fep, L) / (1 + w * q)
    own = np.where(owner == target)[0]
    probe = np.r_[f["rng"].choice(own, 4, replace=False), f["rng"].choice(len(X), 8, replace=False)]
    for c in probe:
        S2 = np.linalg.inv(H + w[c] * np.outer(Xt[c], Xt[c]))
        # 1. D-optimal gain on the target's block
        dlog = (np.linalg.slogdet(Sigma[np.ix_(bidx, bidx)])[1]
                - np.linalg.slogdet(S2[np.ix_(bidx, bidx)])[1])
        claim = -np.log1p(-u_dopt[c]) if u_dopt[c] < 1 else np.inf
        worst_d = max(worst_d, abs(dlog - claim))
        # 2. target-weighted fixed-effect variance drop
        drop = np.trace(A_T @ (Sigma[:k_fep, :k_fep] - S2[:k_fep, :k_fep]))
        worst_t = max(worst_t, abs(drop - u_twfeg[c]))
    print(f"\n1. Phase 1  D-optimal log-det identity      max|diff| = {worst_d:.2e}")
    print(f"2. Phase 2  tr(A_T dSigma_beta) identity     max|diff| = {worst_t:.2e}")

    # --- 3: incremental greedy vs recomputation -----------------------------
    rng = np.random.default_rng(1)
    picks, _ = _greedy(Sigma, Xt, w, lambda S_X, S: _cross_twfeg(S_X, k_fep, L), 3, rng)
    S = Sigma.copy()
    for b in picks:                                     # replay the picks the slow way
        s = S @ Xt[b]
        S = S - np.outer(s, s) * (w[b] / (1 + w[b] * Xt[b] @ s))
    SXf, qf = S @ Xt.T, np.einsum("ij,ji->i", Xt, S @ Xt.T)
    u_fast = None
    S_fast = Sigma.copy()
    SXi = S_fast @ Xt.T
    qi = np.einsum("ij,ji->i", Xt, SXi)
    for b in picks:
        s_b, g_b = SXi[:, b], w[b] / (1 + w[b] * qi[b])
        t = Xt @ s_b
        S_fast -= g_b * np.outer(s_b, s_b)
        SXi -= g_b * np.outer(s_b, t)
        qi -= g_b * t ** 2
    print(f"3. batch bookkeeping  SX  max|diff| = {np.abs(SXi - SXf).max():.2e}"
          f"   q max|diff| = {np.abs(qi - qf).max():.2e}  ({len(picks)} picks)")
    worst_b = max(np.abs(SXi - SXf).max(), np.abs(qi - qf).max())

    # --- 3b: the singleton attenuation identity -----------------------------
    # For a group with no prior labels, marginalising its random effect out of ONE observation
    # must leave beta-information w/(1 + w z'Dz) * x x'. Checked by building the 2x2-block
    # information matrix explicitly and taking its Schur complement.
    from acquisitions import singleton_attenuation
    _, _, _, sd = _blocks_of(res)
    spec_sd = {}
    for j, (comp, _p) in enumerate(__import__("laplace_variance").vc_spec(res)):
        spec_sd.setdefault(comp, float(sd[j]))
    D = np.diag([spec_sd.get("intercept", 0.0) ** 2]
                + [spec_sd.get(f"f{k}", 0.0) ** 2 for k in f["slopes"]])
    worst_a = 0.0
    rng2 = np.random.default_rng(3)
    for c in rng2.choice(len(X), 6, replace=False):
        xb = np.r_[1.0, Z[c]]                              # fixed-effect design row
        zc = np.r_[1.0, Z[c, f["slopes"]]]                 # random-effect design row
        wc = float(w[c])
        I = np.block([[wc * np.outer(xb, xb), wc * np.outer(xb, zc)],
                      [wc * np.outer(zc, xb), wc * np.outer(zc, zc) + np.linalg.inv(D)]])
        nb = len(xb)
        schur = I[:nb, :nb] - I[:nb, nb:] @ np.linalg.solve(I[nb:, nb:], I[nb:, :nb])
        claim = wc / (1.0 + wc * zc @ D @ zc) * np.outer(xb, xb)
        worst_a = max(worst_a, np.abs(schur - claim).max())
    # and the helper must leave a participant WITH columns untouched, discount one without
    persons_known = {p for _c, p in __import__("laplace_variance").vc_spec(res)}
    who_all = list(owner)
    w_eff = singleton_attenuation(Z, who_all, clf, res, w)
    known_mask = np.array([p in persons_known for p in who_all])
    untouched = np.allclose(w_eff[known_mask], w[known_mask])
    print(f"3b. singleton marginalisation identity   max|diff| = {worst_a:.2e}; "
          f"fitted participants untouched = {untouched} "
          f"({int(known_mask.sum())}/{len(w)} rows have random-effect columns)")

    # --- 4: the schedule ----------------------------------------------------
    y_al = np.full(len(f["y"]), np.nan)
    seed_rows = own[:3]
    y_al[seed_rows] = f["y"][seed_rows]
    clf2 = MixedEffectsClassifier(persons=f["persons"], slope_idx=f["slopes"],
                                  classes=[0.0, 1.0], missing_label=np.nan).fit(X, y_al)
    qs = TwoStage(n_calib=6, random_state=0)
    bought, phase1_self = [], []
    for rnd in range(4):
        unl = np.where(np.isnan(y_al))[0]
        own_unl = np.intersect1d(unl, own)
        idx = qs.query(X=X, y=y_al, clf=clf2, target=target, own_candidates=own_unl,
                       A_chol=L, n_acquired=len(bought), candidates=unl, batch_size=3)
        assert len(idx) == 3 and len(set(idx)) == 3, f"round {rnd}: got {idx}"
        if len(bought) < 6:
            phase1_self.append(bool(np.all(owner[idx] == target)))
        y_al[idx] = f["y"][idx]
        bought += list(idx)
        clf2 = MixedEffectsClassifier(persons=f["persons"], slope_idx=f["slopes"],
                                      classes=[0.0, 1.0], missing_label=np.nan).fit(X, y_al)
    self_share_late = float(np.mean(owner[np.array(bought[6:])] == target))
    print(f"4. schedule: first 6 picks target-owned = {all(phase1_self)}; "
          f"self-share of picks 7-12 = {self_share_late:.2f}")

    # Phase 1 standalone, k-medoids variant: must also stay on the target's rows
    km = TargetCalibration(criterion="kmedoids", random_state=0)
    y_k = np.full(len(f["y"]), np.nan)
    y_k[seed_rows] = f["y"][seed_rows]
    own_unl = np.setdiff1d(own, seed_rows)
    idx_km = km.query(X=X, y=y_k, clf=clf, target=target, candidates=own_unl, batch_size=3)
    km_ok = bool(np.all(owner[idx_km] == target)) and len(set(idx_km)) == 3
    print(f"   k-medoids variant returns 3 distinct target-owned rows = {km_ok}")

    ok = (worst_d < 1e-6 and worst_t < 1e-8 and worst_b < 1e-8 and worst_a < 1e-8
          and untouched and all(phase1_self) and km_ok)
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
