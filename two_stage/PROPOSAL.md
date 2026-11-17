# Two-Stage Personalised Active Learning for Generalised Linear Mixed-Effects Models on Multi-User Sensor Streams

**Target task.** COMPUTER_WORK detection from ExtraSensory phone-and-watch features, one
personalised model per participant, labels purchasable from anyone in the cohort, severe label
budget (33 labels per personalised model).

**Proposed claim.** Under a low budget, geometric acquisition on a multi-subject wearable pool is
not merely suboptimal but *systematically mis-targeted*, and it buys that mis-targeting with a
severely unequal labelling burden across participants. A two-phase schedule that first anchors
the target's random effects by D-optimal self-querying and then buys population-level
fixed-effect information *weighted by the target's own operational density* should recover the
personalised accuracy without concentrating the burden.

**What the experiment found.** The framework beats equal-budget random sampling over the pool by
**+0.060 Macro PR-AUC** (+0.097 AUPRC, *p* < 1e-4), reaches random's full-budget accuracy with
**6 of 33 labels**, and does so at the same cohort-level burden inequality as random. The
diagnosis of the geometric baselines holds and is sharper than proposed. But the framework
**does not beat a one-line control**: uniform sampling restricted to the target's own blocks
captures 89% of the best arm's advantage, and neither information criterion improves on it
significantly (§5.3). The reason is measurable rather than incidental — at 33 labels the random
effects this framework is built to exploit are prior-dominated, and spreading labels across
participants collapses them outright (§2.4) — which is also what makes the negative result
informative about the estimator rather than about these two criteria.

**Status of this document.** Sections 1–4 are the proposal as specified, with its claims left
standing and marked where the evidence later contradicted them. Sections 5–7 report the
experiment that was run in this subdirectory, code, logs and figures alongside. The benchmark is
scoped to the two-phase strategy against random acquisition.

---

## 1. Problem setting

### 1.1 The estimator, and what "personalised" means in it

A participant's activity is predicted by a Bayesian generalised linear mixed-effects model
(GLMM): for participant *i* and 15-minute block *j* with standardised feature vector
$\mathbf{x}_{ij} \in \mathbb{R}^{p}$ ($p = 124$),

$$\operatorname{logit}\Pr(y_{ij}=1) \;=\; \beta_0 + \mathbf{x}_{ij}^{\top}\boldsymbol{\beta}
\;+\; u_i \;+\; \sum_{k \in \mathcal{K}} v_{ik}\, x_{ij,k},
\qquad
\mathbf{b}_i = (u_i, \{v_{ik}\}_{k\in\mathcal{K}}) \sim \mathcal{N}(\mathbf{0}, \operatorname{diag}(\mathbf{s}^2)),$$

with $|\mathcal{K}| = K = 5$ features carrying a per-participant random slope, selected before the
run by between-participant variance of personal-GLM coefficients. $\boldsymbol{\beta}$ is
population-level and shared; $\mathbf{b}_i$ is individual. Fitting is MAP (`fit_map`), so the
posterior is Laplace:

$$p(\boldsymbol{\theta}\mid\mathcal{D}) \approx
\mathcal{N}(\hat{\boldsymbol{\theta}}, \boldsymbol{\Sigma}),
\qquad
\boldsymbol{\Sigma} = \big(\tilde{\mathbf{X}}^{\top}\mathbf{W}\tilde{\mathbf{X}} + \boldsymbol{\Lambda}\big)^{-1},
\qquad
\boldsymbol{\theta} = (\boldsymbol{\beta}, \mathbf{b}),$$

where $\tilde{\mathbf{X}} = [\mathbf{X}, \mathbf{Z}]$ stacks the fixed- and random-effect designs,
$\mathbf{W} = \operatorname{diag}(\mu(1-\mu))$, and $\boldsymbol{\Lambda}$ is the prior precision
(fixed-effect prior SD 1.0, prior SD 0.5 on each log random-effect SD).

Personalisation is therefore a *decomposition*, not a separate model: the prediction for target
$T$ is driven by $\boldsymbol{\beta}$, which every participant's labels inform, plus
$\mathbf{b}_T$, which **only the target's own labels inform**. That asymmetry is the hinge of
everything below.

### 1.2 The acquisition problem

For a fixed target $T$ we hold a pool of ~11 000 unlabelled blocks contributed by 43
participants, a seed set of 3 of the target's own labels (1 positive, 2 negative, spread over
days), and a budget of 10 rounds × 3 labels. Each purchased label may come from *any*
participant. We are scored on $T$'s held-out **days**, and we are accountable for *whose* labels
we bought.

### 1.3 Why both axes are necessary

A personalised-accuracy-only evaluation licenses an acquisition that repeatedly interrogates
whichever two or three participants happen to be most geometrically interesting, which in a real
deployment is how a study loses those participants. Conversely, equity alone is trivially
maximised by never asking anyone anything useful. The two axes together define the problem:

* **Axis 1 — predictive performance on the target.** Threshold-free, macro-averaged over the two
  classes: Macro PR-AUC (primary) and Macro ROC-AUC on the target's held-out days.
* **Axis 2 — participant burden equity.** The Gini coefficient $G_q$ and the max-to-mean ratio of
  the per-participant label counts, pooled over every personalised model in the cohort (each
  participant takes a turn as target), against the *pool-proportional* reference rather than
  against zero.

---

## 2. Diagnosis: why geometric acquisition fails here

Everything in this section is measured on the same 10 day splits and the same 43 targets as the
benchmark, from the runs' own query records (`diagnose.py`, `probcover_delta.py`,
`vc_diagnostic.py`, `manifold_check.py`). Where the evidence contradicts the premise this
proposal started from, §2.6 says so.

### 2.1 What the two criteria optimise, and why that is misaligned

**CoreSet** is k-centre greedy: it buys the point that minimises the maximum distance from any
pool point to its nearest labelled point, which means that at every step it buys *the point
currently furthest from everything already labelled*. In 124 z-scored wearable features that is a
sensor anomaly, a dropout-and-recovery artefact, or a block belonging to a participant whose
signature resembles nobody else's. The worst-case covering guarantee this comes from is a
sensible thing to want when labels are plentiful; at 30 labels over ~11 000 blocks the guarantee
is vacuous and all that survives of it is a preference for extremes. Measured: the blocks CoreSet
buys sit at a mean distance of **23.5** from their 20 nearest neighbours, against a pool mean of
**5.63** — an effect size of **d = +6.3** against random sampling on the same split.

**ProbCover** inverts the preference: it covers the pool with balls of radius $\delta$ and buys
the candidate covering the most still-uncovered points, which favours *dense* regions. Its
$\delta$ is the largest value on a grid whose balls stay "pure" against a k-means partition, and
scikit-activeml's default grid is `np.arange(0.2, 2.2, 0.2)` — it tops out at 2.0. That grid is
calibrated for the setting ProbCover was published in: self-supervised embeddings, usually
L2-normalised, where the interesting radii are $O(1)$. Here the median pairwise distance is
**13.4** and the mean 20-NN distance **5.65**, so the selected $\delta$ is **2.0 — the grid
maximum, i.e. the purity constraint never binds, the grid simply runs out**. At that radius:

* **0.023%** of all pairs are within $\delta$ of each other;
* **78.4%** of blocks have a *completely empty* ball (the median in-ball degree is **0**);
* the three participants with the tightest clusters hold **42.6%** of all in-ball coverage mass.

The covering objective has degenerated. Nothing is being covered: 78% of candidates cover only
themselves, and all the coverage mass that exists belongs to a handful of participants who
recorded long stationary stretches. What remains of the criterion is a density ranking, which is
why ProbCover's purchases sit at a mean 20-NN distance of **1.37** (d = −1.5), buy blocks whose
neighbourhoods agree on the label **95.7%** of the time (pool: 82.1%), and include a
label-boundary block **1.7%** of the time (pool: 9.9%). It buys the most redundant blocks in the
pool. This failure is a scaling mismatch rather than a flaw in the idea — passing `deltas=`
scaled to the data would fix it — but the default is what a practitioner gets, and it is what the
published comparison measures.

**What the two share matters more than how they differ.** Neither criterion can see participant
identity: it is not a coordinate in the space they measure distances in, and rightly so, since a
participant index is not a direction in which distance means anything. But in a mixed-effects
model *the owner of a block determines which parameters its label can inform* — a stranger's
label speaks only to $\boldsymbol{\beta}$, the target's own label speaks to $\boldsymbol{\beta}$
and to $\mathbf{b}_T$. A subject-blind acquisition on a multi-subject pool is therefore not
merely uninformed about the estimator; it is structurally unable to represent the one distinction
that decides what a label is worth. §2.4 puts a number on what that costs.

### 2.2 What each method buys, measured

Effect size $d$ of the queried blocks against the pool, paired against random sampling on the
same split (10 splits, ~11 600 purchases per arm; `diagnosis/row_profile.csv`,
`fig_row_profile.png`):

| arm | isolation (20-NN dist) | extremity for its owner | phase-shift impurity | transition block | neighbourhood purity | positive |
|---|---|---|---|---|---|---|
| `coreset` | **+6.28** | **+5.07** | −0.02 | +0.01 | +0.26 | −0.26 |
| `bald` | **+3.63** | **+3.28** | +0.02 | +0.07 | +0.06 | −0.08 |
| `twfeg` | +1.40 | +1.27 | +0.12 | +0.14 | +0.02 | −0.04 |
| `twostage` | +1.32 | +1.19 | +0.08 | +0.12 | −0.03 | −0.03 |
| `calib` | +0.07 | −0.22 | +0.03 | +0.10 | **−0.31** | **+0.12** |
| `probcover` | **−1.50** | −0.44 | −0.22 | −0.26 | **+0.61** | −0.29 |

Three readings:

* **The anomaly-seeking is real and large for CoreSet, and for BALD.** CoreSet's purchases are
  five standard deviations more extreme *for their own owner* than that owner's typical block —
  this is the subject-specific noise axis, and it is the single biggest effect in the table.
  BALD is not far behind, which is worth stating plainly: it is posterior-aware, and being
  posterior-aware does not help, because posterior variance is largest exactly where the data
  are sparse. "Model-based versus geometric" is not the axis that matters.
* **Neither geometric method hunts positives; `calib` is the only arm that does.** At a 16.2%
  base rate, CoreSet buys 6.4% positives and ProbCover 5.3%, against random's 16.1%. D-optimality
  on the target's own blocks buys **20.5%** positives and is the only arm with a *negative*
  neighbourhood-purity effect — it buys the target's own boundary cases, which is what a design
  criterion should do.
* **TW-FEG is mildly outlier-seeking, not immune.** §3.3 claimed geometric distinctiveness
  "enters nowhere" in $U_{\text{twfeg}}$. That is true of the formula and false of the behaviour:
  isolated blocks tend to have large feature norms, which inflates the $\mathbf{A}_T$-weighted
  cross term. The effect is a quarter of CoreSet's (+1.4 vs +6.3) but it is not zero, and the
  claim should be weakened to that.

### 2.3 The burden consequence

Cohort Gini $G_q$ and max-to-mean ratio of per-participant label counts, pooled over all 43
personalised models, at the full budget (10 splits; `diagnosis/burden_curve.csv`,
`fig_burden.png`, `fig_lorenz.png`):

| arm | $G_q$ | max-to-mean | participants ever asked (of 43) | over-buying from the 3 most isolated participants |
|---|---|---|---|---|
| `calib` | **0.004** | **1.01** | 42.9 | 1.5× |
| `twfeg_own` | 0.159 | 1.70 | 43.0 | 1.4× |
| *pool-proportional reference* | *0.275* | *1.94* | *43* | *1.0×* |
| `random` | 0.287 | 2.04 | 43.0 | 1.0× |
| `twostage` | 0.293 | 4.02 | 43.0 | 3.3× |
| `twfeg` | 0.347 | 4.46 | 42.9 | 3.5× |
| `bald` | 0.465 | 4.28 | 42.5 | **5.0×** |
| `coreset` | 0.657 | 5.75 | **30.7** | 4.5× |
| `probcover` | **0.676** | 5.71 | **20.7** | 0.0× |

The reference line is the point: participants contribute between 36 and 475 blocks, so the pool
itself has $G_q = 0.275$ and no unbiased sampler removes that. Random lands on it (0.287), which
also validates the seeding. Against that reference:

* CoreSet and ProbCover run at **2.3× the pool's own inequality**, 2.8× its max-to-mean ratio,
  and **exclude 12 and 22 participants entirely** — ProbCover asks fewer than half the cohort
  anything at all.
* BALD over-buys from the three most isolated participants by **5.0×** their pool share, the
  worst in the table; CoreSet by 4.5×. ProbCover is the mirror image (0.0× — it never asks them),
  which is its own kind of failure: those participants' models are built entirely from strangers.
* The proposed arms sit at or below random. `calib` reaches $G_q = 0.004$ — every participant
  answers 30 questions about their own model and nobody else's. Note that this is slightly
  *more* than proportional for small contributors, and deliberately so: burden should scale with
  people, not with how much data someone already donated.
* `twostage` matches random's Gini (0.293 vs 0.287) while its max-to-mean is higher (4.02 vs
  2.04). Both numbers are correct and they say different things: Phase 1 spreads perfectly, and
  Phase 2's 24 labels still land somewhat unevenly. This is exactly why both statistics are
  reported.

### 2.4 Why a mixed-effects model pays for subject-blindness

This is the measurement that reframes the diagnosis (`vc_diagnostic.py`). Fit the model on 33
labels drawn in three ways and read off what it identified:

| labels drawn | participants holding a label | fitted random-effect SD | $\|\hat{\mathbf{b}}_T\|_\infty$ | random-effect share of the target's predictive spread |
|---|---|---|---|---|
| all from the target | 1 | **0.78** | 0.24 | 7.1% |
| half target, half pool | 13.6 | **0.035** | 0.0004 | 0.01% |
| 20% target, 80% pool | 18.5 | **0.011** | 0.0000 | 0.00% |

Spreading a small budget across many participants gives each of them one or two labels. Singleton
groups carry no within-person replication, so the likelihood cannot separate between-person
variance from binomial noise, and the MAP estimate of every variance component collapses towards
zero. **The personalised model silently stops being personalised**: at 33 labels spread over ~18
people the random effects contribute nothing, and what is left is a pooled GLM with 125
coefficients fitted on 33 rows.

Two consequences, and the second is the deeper one:

1. A pool-wide acquisition is not buying "population-level information that the target can borrow
   from". It is buying away the structure that made the model personalised in the first place.
2. Even in the favourable case — 33 labels all from the target — the random effects supply only
   **7.1%** of the target's predictive spread, and the posterior SD of $\mathbf{b}_T$ is **0.763**
   against a prior of 0.779. So $\mathbf{b}_T$ is barely identified either, and the gain from
   self-querying cannot be mostly "anchoring $\mathbf{b}_T$", which is what §3.2 assumed. The
   operative mechanism must be simpler: the target's own blocks are the training rows whose
   distribution matches the rows the model will be scored on. Phase 1 works, but for a reason
   closer to covariate-shift matching than to random-effect identification. §6.1 takes this up.

### 2.5 A correction that follows — and its measured size

§2.4 implies a specific flaw in Phase 2 as specified. `stack_design` gives a candidate whose owner
holds no labels a zero random-effect block, so $U_{\text{twfeg}}$ treats a stranger's first label
as a pure fixed-effect observation — which **overstates** its value, because that stranger's own
random effect will absorb part of its residual. Marginalising $b_c \sim \mathcal{N}(0, D)$ out of
one observation gives, exactly,

$$I_{\boldsymbol{\beta}|\text{marg}} = \frac{w_c}{1 + w_c \mathbf{z}_c^{\top}\mathbf{D}\mathbf{z}_c}\,\tilde{\mathbf{x}}_c^{\text{fix}}\tilde{\mathbf{x}}_c^{\text{fix}\top},$$

so the candidate should carry the attenuated weight $w_c/(1 + w_c\mathbf{z}_c'\mathbf{D}\mathbf{z}_c)$
(derived and verified against the explicit Schur complement to $5.6\times10^{-17}$ in
`selftest.py`). The discount is not small: at 33 self-labels it averages **0.77** across
candidates (SD 0.17, minimum 0.15). Whether it *reorders* anything is a separate question —
`twfeg_marg` and `twostage_marg` in §5.6 answer it.

### 2.6 What the evidence does not support

The premise this proposal began from was that the geometric methods fail *because* they
over-query subject-specific noise and phase-shift anomalies. The first half of that survives as a
description of what they buy (§2.2). As a causal account of why they score badly, it does not.

Within each arm and split — so that every between-method difference is removed, and the variation
is only "this arm bought more anomalous blocks for this target than it usually does" — the
correlation between the anomaly of what was bought and the final AUROC is (3 861 cells):

| attribute of the purchased blocks | Pearson r | p |
|---|---|---|
| isolation (20-NN distance) | −0.010 | 0.55 |
| extremity for its owner (own_z) | −0.034 | 0.033 |
| phase-shift impurity | **+0.079** | <1e-4 |
| transition block | **+0.080** | <1e-4 |
| neighbourhood label purity | −0.016 | 0.33 |
| positive label | **+0.111** | <1e-4 |

Isolation does not predict performance at all. Owner-extremity predicts it weakly and in the
expected direction. **Phase-shift blocks predict it in the *opposite* direction**: a block whose
15 minutes straddle an activity boundary is, if anything, mildly *useful* — unsurprising once
stated, since at a 16% base rate a mixed block usually contains some positive minutes, and
positives are the scarce commodity (the strongest correlate in the table).

So the diagnosis has to be stated differently from how it was posed. The geometric criteria fail
because they are **subject-blind**, not because they are noise-prone:

* they cannot tell whose label they are buying, and in a GLMM that is what determines which
  parameters a label informs (§2.1);
* spreading the budget across strangers destroys the random-effect structure the personalised
  model depends on (§2.4);
* and the same blindness concentrates the human cost on a few individuals while excluding half
  the cohort (§2.3).

The anomaly-seeking is a *symptom* of the same blindness — the objective has no notion of whose
distribution a block is extreme with respect to — and it is worth reporting as such, but it is
not the channel through which the accuracy is lost.

---

## 3. The proposed framework

Two phases, one posterior, an explicit schedule between them.

| | Phase 1 — Target Calibration | Phase 2 — Target-Weighted Fixed-Effect Gain |
|---|---|---|
| objective | $\mathbf{b}_T$ | $\boldsymbol{\beta}$ |
| criterion | D-optimality on the posterior block of $\mathbf{b}_T$ | A-optimality on $\boldsymbol{\beta}$ under the target's density |
| candidate set | the target's own unlabelled blocks | the whole multi-subject pool |
| budget | $n_{\text{calib}} \approx \dim \mathbf{b}_T$, i.e. 5–10 labels | the remainder |

### 3.1 Why a schedule rather than one combined criterion

Write the target's predictive linear predictor as
$\eta_T(\mathbf{x}) = \beta_0 + \mathbf{x}^{\top}\boldsymbol{\beta} + \mathbf{z}(\mathbf{x})^{\top}\mathbf{b}_T$.
Its posterior variance decomposes into a $\boldsymbol{\beta}$ part, a $\mathbf{b}_T$ part and a
cross term. At the start of a run the $\mathbf{b}_T$ part is essentially at its prior — the target
has 3 labels — and only the target's own blocks can reduce it. A single criterion that sums the
two parts will trade a large, cheap, *exclusively self-supplied* reduction against a diffuse
reduction available from 11 000 candidates, and the relative scaling of that trade is an artefact
of the prior SDs rather than a statement about the task. Separating the phases makes the trade a
declared budget ($n_{\text{calib}}$) instead of an accident of parameterisation, and it is also
what makes the method's burden profile predictable: Phase 1 is self-querying by construction,
Phase 2 is pool-wide by construction.

### 3.2 Phase 1 — active Target Calibration

Let $B$ index the columns of $\mathbf{b}_T$ inside $\boldsymbol{\theta}$. Labelling candidate $c$
adds $w_c \tilde{\mathbf{x}}_c \tilde{\mathbf{x}}_c^{\top}$ to the posterior precision, so by
Sherman–Morrison

$$\boldsymbol{\Sigma}' = \boldsymbol{\Sigma} - \gamma_c\,\boldsymbol{\sigma}_c\boldsymbol{\sigma}_c^{\top},
\qquad \boldsymbol{\sigma}_c = \boldsymbol{\Sigma}\tilde{\mathbf{x}}_c,
\qquad \gamma_c = \frac{w_c}{1 + w_c \tilde{\mathbf{x}}_c^{\top}\boldsymbol{\Sigma}\tilde{\mathbf{x}}_c},$$

and the determinant of the target's block factorises **exactly**:

$$\det \boldsymbol{\Sigma}'_{BB} = \det \boldsymbol{\Sigma}_{BB}\,
\big(1 - \gamma_c\, \mathbf{a}_c^{\top}\boldsymbol{\Sigma}_{BB}^{-1}\mathbf{a}_c\big),
\qquad \mathbf{a}_c = \boldsymbol{\sigma}_c[B].$$

Maximising the D-optimal gain $\log\det\boldsymbol{\Sigma}_{BB} - \log\det\boldsymbol{\Sigma}'_{BB}$
is therefore *exactly* maximising

$$\boxed{\;U_{\text{dopt}}(c) \;=\; \gamma_c\,\mathbf{a}_c^{\top}\boldsymbol{\Sigma}_{BB}^{-1}\mathbf{a}_c\;}$$

a closed form with no label simulation and no refit per candidate. (Verified against explicit
re-inversion to $2.6\times10^{-15}$ in `selftest.py`.)

Three properties:

* **The candidate restriction is the criterion, not a heuristic.** A block owned by anyone else
  has a structural zero in the target's random-effect columns, hence $\mathbf{a}_c = \mathbf{0}$
  and $U_{\text{dopt}}(c) = 0$ identically. Phase 1 queries the target because nothing else can
  answer the question it is asking.
* **It is the posterior form of D-optimality on $\mathbf{Z}_T$.** The classical criterion
  maximises $\det(\mathbf{Z}_T^{\top}\mathbf{W}\mathbf{Z}_T + \boldsymbol{\Lambda}_b)$ over
  designs; the expression above is the same objective with the cross-information between
  $\mathbf{b}_T$ and $\boldsymbol{\beta}$ retained, which is free to keep here and is what lets
  borrowed strength from the pool count against the calibration budget.
* **The budget has a principled scale.** $\dim\mathbf{b}_T = K+1 = 6$. Fewer than that many
  informative rows cannot identify the block; many more spends pool-sized budget on a 6-parameter
  problem. Hence $n_{\text{calib}} \in [5, 10]$, with $n_{\text{calib}} = 6$ as the default and a
  sweep reported in §5.4.
  *This reasoning assumes the budget can identify $\mathbf{b}_T$ at all. §2.4 measures that it
  cannot — 33 self-labels move the posterior SD of $\mathbf{b}_T$ from 0.779 to 0.763 — so the
  dimension of $\mathbf{b}_T$ is the wrong scale for sizing Phase 1, and the sweep in §5.4 is
  the question rather than a robustness check.*

**Ablation (`twostage_kmed`).** The same schedule with Phase 1 replaced by *target-restricted
k-medoids*: greedy facility-location medoids of the target's own blocks, minimising the mean
distance from every target block to its nearest selected block. Model-free, anchors
$\mathbf{b}_T$ from *representative* rather than *informative* rows, and answers whether the
posterior is doing any work in Phase 1.

### 3.3 Phase 2 — Target-Weighted Fixed-Effect Gain (TW-FEG)

Let $\tilde{\mathbf{x}}^{\text{fix}} = [1, \mathbf{x}, \mathbf{0}]$ be the design row that uses the
**fixed effects only**, and let $\pi$ be a probability measure on feature space concentrated on
the target's operational manifold,

$$\pi(\mathbf{x}) \;\propto\; r(\mathbf{x}) \;=\; \frac{p_T(\mathbf{x})}{p_{\text{pool}}(\mathbf{x})},$$

estimated over the pooled blocks. Define the target-weighted second moment and the quantity to be
reduced:

$$\mathbf{A}_T = \sum_e \pi_e\, \tilde{\mathbf{x}}^{\text{fix}}_e \tilde{\mathbf{x}}^{\text{fix}\top}_e
\in \mathbb{R}^{(1+p)\times(1+p)},
\qquad
\mathcal{V}(\boldsymbol{\Sigma}) = \operatorname{tr}\!\big(\mathbf{A}_T\,\boldsymbol{\Sigma}_{\beta\beta}\big)
= \mathbb{E}_{\pi}\!\left[\operatorname{Var}\big(\beta_0 + \mathbf{x}^{\top}\boldsymbol{\beta}\big)\right].$$

$\mathcal{V}$ is the posterior variance of the **population-level** predictor, averaged over where
the target actually operates. One label's reduction is again exact:

$$\boxed{\;U_{\text{twfeg}}(c) \;=\; \frac{w_c \sum_e \pi_e \big(\tilde{\mathbf{x}}^{\text{fix}\top}_e \boldsymbol{\Sigma}\, \tilde{\mathbf{x}}_c\big)^2}{1 + w_c\, \tilde{\mathbf{x}}_c^{\top}\boldsymbol{\Sigma}\tilde{\mathbf{x}}_c}\;}$$

(Verified against $\operatorname{tr}(\mathbf{A}_T\Delta\boldsymbol{\Sigma}_{\beta\beta})$ to
$4.7\times10^{-16}$.) Four properties, each of which is a difference from a baseline rather than a
restatement:

1. **Population-level by construction.** The evaluation rows carry zeros in the random block, so
   nothing in $U_{\text{twfeg}}$ rewards shrinking any individual's $\mathbf{b}$ — including the
   target's. Phase 1 has already bought $\mathbf{b}_T$; Phase 2 cannot be diverted into
   re-buying it, which a criterion on the full predictive variance (e.g. the parent repo's
   `laplace` arm) can and does.
2. **Target-weighted, not target-restricted.** The cross term
   $\tilde{\mathbf{x}}^{\text{fix}\top}_e\boldsymbol{\Sigma}\tilde{\mathbf{x}}_c$ couples a
   stranger's candidate to the target's manifold *through the shared $\boldsymbol{\beta}$*. A
   block owned by someone else is valued exactly insofar as it informs $\boldsymbol{\beta}$ where
   the target operates. Pool size becomes an asset rather than a dilution.
3. **Not purchasable by an anomaly.** Geometric distinctiveness — the thing CoreSet maximises —
   enters nowhere. An isolated block scores well only if its *direction* carries
   $\mathbf{A}_T$-weighted information, and the denominator actively discounts candidates whose
   own predictive variance is large, which is the self-limiting behaviour an extrapolation point
   ought to have.
   *Measured, this holds only partially (§2.2): TW-FEG's purchases are mildly isolated
   (d = +1.4, against CoreSet's +6.3), because an isolated block tends to have a large feature
   norm and that inflates the $\mathbf{A}_T$-weighted cross term. The claim is true of the
   formula and should be weakened to "a quarter of CoreSet's anomaly-seeking" as a statement
   about behaviour.*
4. **Label-free.** $U_{\text{twfeg}}$ contains no $y_c$: $w_c$ uses the current fitted $\mu_c$
   only. One posterior per round, then matrix products — unlike expected error reduction, which
   needs two refits per candidate and a 50-candidate subsample to be runnable at all. TW-FEG
   scores all ~11 000 candidates every round.

**The density ratio.** $r$ is estimated by probabilistic classification (Sugiyama et al., 2012):
an L2-regularised logistic discriminator of "is this row the target's?" with balanced class
weights, read out as $r \propto q/(1-q)$, with **out-of-fold** probabilities (124 features against
one participant's 40–400 rows separates in-sample, which would collapse $r$ to an indicator of literal
membership) and clipping at the 99th percentile before normalisation (one row with $r=400$ would
turn a weighted expectation into a single-point evaluation). It never touches $y$.

**Ablations.** `twfeg_own` puts $\pi$ uniformly on the target's own blocks (does the ratio do
anything?); `twfeg` runs Phase 2 from round 1 (is Phase 1 needed?); `calib` runs Phase 1 for the
whole budget (is Phase 2 needed?).

### 3.4 Burden properties by construction

The framework's burden profile is not an empirical hope; it is implied by the two candidate sets.
Phase 1 is self-querying, which across a cohort in which every participant takes a turn as target
is *perfectly* equitable (each participant answers for their own model). Phase 2 is valued through
a $\boldsymbol{\beta}$ cross term that is maximised by spread rather than by repetition: once a
direction of feature space has been labelled, $\boldsymbol{\Sigma}$ has shrunk along it and the
next-best candidate lies elsewhere. Neither phase contains a term that rewards returning to the
same participant. The experiment tests whether that argument survives contact with the data.

### 3.5 Cost

Per round and per target: one GLMM MAP refit (shared by every arm), one $d\times d$ inversion for
$\boldsymbol{\Sigma}$ with $d = (1+p) + K_{\text{cols}}$ — 125 fixed-effect columns plus six
random-effect columns for each participant who currently holds a label, so $d$ grows from ~131 at
round 0 to ~300 at the full budget — one $(d\times n)$ product
$\boldsymbol{\Sigma}\tilde{\mathbf{X}}^{\top}$, and $O(p^2 n)$ for the criterion. $\mathbf{A}_T$ is
round-independent and is Cholesky-factored once per target; within a batch the greedy update is
rank-one on $\boldsymbol{\Sigma}$, on $\boldsymbol{\Sigma}\tilde{\mathbf{X}}^{\top}$ and on the
quadratic form, so a batch of 3 costs one full product plus two cheap updates. Measured: ~2–3 s
per target per arm for an 11-round run, against ~4 300 s per split for expected error reduction
in the parent repo — i.e. the criterion is not the bottleneck, the model refits are.

---

## 4. Experimental protocol

### 4.1 Data and splits

ExtraSensory, 61 participants, target label `label:COMPUTER_WORK`; 45 participants carry ≥20
minutes of both classes, 43 survive the day-split eligibility filter in every fold. Features: the
phone+watch sensor groups minus the context and artefact columns that predict the label without
sensing the activity (124 features). Minute rows are averaged into **15-minute blocks** (majority
label), because adjacent minutes of one activity episode are near-duplicates and treating them as
independent observations overstates what a label buys. Splits hold out whole **days** at random
(30%), drawn on minute rows with each side blocked afterwards, redrawn until both sides carry ≥5
rows of each class. Imputation and z-scoring are fitted on the pooled training blocks only.
**10 independent day splits**; the pool is ~10 900–11 400 blocks at ~16% positive.

Every one of these choices, and the code implementing it, is the parent repo's — not re-derived
here — so an arm in this subdirectory is comparable to an arm there by construction.

### 4.2 Budget and seeding

3 seed labels from the target (1 positive, 2 negative, spread over days), then 10 rounds × 3
labels = **33 labels per personalised model**. The seed set is drawn from
`default_rng([seed, split, participant, 11])`, so it is *identical across arms and identical to
the parent repo's runs*; the strategy RNG is keyed separately. Every arm within a split therefore
starts from the same model and sees the same pool, which makes the (split, target) cell a valid
pairing unit.

### 4.3 Arms

**The benchmark is `twostage` against `random`** — equal-budget uniform sampling over the pool —
on 43 targets × 10 splits × 11 rounds, plus one control and one sweep:

* **`self_random`** (added after the first results were inspected, and reported as such in §5.3):
  uniform sampling restricted to the target's own unlabelled blocks. This is the control that
  separates the framework's two simultaneous claims — that the target is the right person to ask,
  and that its criteria pick the right blocks of that person. Without it the headline comparison
  cannot distinguish the method from the premise.
* **the $n_{\text{calib}}$ sweep** (§5.4): the same strategy at Phase 1 budgets
  0, 3, 6, 9, 12, 18, 24, 30, each a full 10 splits.

Run on the identical splits, seeds and cells, and used for the diagnosis in §2 and the ablations:
`calib`, `twfeg`, `twfeg_own`, `twostage_kmed`, `twfeg_marg`, `twostage_marg`, `coreset`,
`probcover`, `bald`. They are tabulated in each run's `analysis/` but are not the comparison this
document makes.

**BALD.** Standard BALD needs a distribution over models, which `predict_proba` cannot supply, and
scikit-activeml's `GreedyBALD` needs a committee. The committee here is the Laplace posterior
itself: $\boldsymbol{\theta}_s \sim \mathcal{N}(\hat{\boldsymbol{\theta}}, \boldsymbol{\Sigma})$,
$S = 20$ draws, exposed as `sample_proba` (`bald.py`). This is deliberate: it makes the comparison
*posterior-aware vs posterior-aware*, so a TW-FEG win cannot be attributed to "model-based beats
geometric". BALD and TW-FEG differ in **what they are informative about** — the whole parameter
vector including every stranger's random effect, versus $\boldsymbol{\beta}$ under the target's
density.

### 4.4 Metrics

*Axis 1*: **Macro PR-AUC** (mean of the per-class average precision) is primary — at 16%
prevalence the positive-class PR-AUC alone is moved a long way by a handful of test blocks (the
median target has 18 positive test blocks, the thinnest has 3), and the macro form is what stops
an arm that buys only easy negatives from looking good. Macro ROC-AUC, AUROC and AUPRC are
reported alongside; Macro ROC-AUC is numerically identical to AUROC in the binary case and is
carried for reporting symmetry only.

Two caveats on that choice, since it drives the headline numbers. First, "macro" normally means
averaging over $K$ classes in a multi-class problem; in a binary problem it amounts to averaging
average precision with the average precision of the complementary ranking task, which is a
defensible stabiliser rather than a convention. Second, it **dilutes**: the negative-class curve
is nearly saturated (mean AP$^-$ = 0.884 at the full budget, against AP$^+$ = 0.391) and barely
responds to acquisition, so averaging it in halves the measured effect. Measured over the arms,
the macro gaps are a median **0.51×** the plain-AUPRC gaps while the arm ordering is *identical*
(Spearman $\rho = 1.00$ against AUPRC, 0.98 against AUROC). Every table in §5 therefore reports
all four columns, and no conclusion depends on which is called primary.

*Axis 2*: Gini coefficient $G_q$ and max-to-mean ratio of per-participant query counts over the
**fixed 43-participant cohort with zeros included**, pooled over all 43 personalised models;
coverage (participants ever asked), top-participant share, and Lorenz curves. The reference is
*pool-proportional* sampling: participants contribute between 43 and ~400 blocks, so even a
perfectly unbiased sampler inherits the pool's own inequality, and reading $G_q$ against 0 would
score that inherited inequality as a fairness failure.

### 4.5 Inference

Every number is computed per (split, target) cell and reduced in two ways, because they answer
different questions:

* **Wilcoxon signed-rank over all 430 cells** — is the method better for the typical personalised
  model? No normality assumption, but cells within a split share a pool and are not independent.
* **One-sample t-test over the 10 per-split means** — would this replicate on a new day split?
  Ten independent numbers; the conservative reading, and the one quoted in the text.

Win rate (fraction of cells improved) is reported because a mean gain concentrated in three
participants is a different finding from a uniform one.

### 4.6 Diagnosis instrumentation

Every pooled block carries method-independent attributes (`pool_attributes.py`), computed once per
split and cached: mean distance to its 20 nearest neighbours (**isolation**), the share of those
neighbours carrying the same label (**local label purity**), its distance to its *owner's* centroid
in units of that owner's spread (**own_z**, subject-specific extremity), the mixedness of the 15
minute labels averaged into it (**impurity** — the *phase-shift* blocks, an activity boundary
smeared across one window whose majority label is near a coin toss), whether its label differs
from a temporally adjacent block of the same participant (**transition**), and per participant the
mean distance from their blocks to the nearest block belonging to anyone else (**isolation of the
individual** — the outlier-participant axis). Block-level attributes are recovered by re-running
the parent repo's own split machinery on a row-index proxy and are asserted to reproduce the block
labels the runs actually used, so a silent mis-alignment cannot pass as a result.

---

## 5. Results: the two-phase strategy against equal-budget random sampling

The comparison reported here is the framework (`twostage`: 6 labels of D-optimal Target
Calibration, then TW-FEG) against **equal-budget random sampling**, on 10 day splits × 43 target
participants × 33 labels. Every cell is paired: both arms saw the same split, the same target,
the same seed set and the same pool, so the difference within a cell is the acquisition and
nothing else. §5.8 says what else is on disk.

### 5.1 Axis 1 — predictive performance on the target

At the full budget (33 labels), 429 (split × target) cells, 10 splits
(`analysis_twostage_vs_random/final_*.csv`):

| | `twostage` | `random` | paired gap | SE (over splits) | win rate | Wilcoxon *p* | split-level *p* |
|---|---|---|---|---|---|---|---|
| **Macro PR-AUC** | **0.6660** | 0.6061 | **+0.0598** | 0.0069 | 61.8% | <1e-4 | <1e-4 |
| AUPRC | 0.4268 | 0.3295 | +0.0973 | 0.0101 | 63.6% | <1e-4 | <1e-4 |
| AUROC / Macro ROC-AUC | 0.7069 | 0.6356 | +0.0713 | 0.0129 | 60.6% | <1e-4 | 0.0004 |

The gap is significant on both readings — the per-cell Wilcoxon signed-rank test and the
conservative t-test over 10 independent per-split means — and the sign is the same on every
metric, as §4.4 said it would be.

**Label efficiency.** `twostage` reaches random's *full-budget* Macro PR-AUC (0.6061) after
**6 labels**, against random's 33: a 5.5× reduction in labelling effort for the same accuracy
(`label_efficiency.csv`).

**Per participant** (`fig_curves_per_person.png`, `who_benefits.py`,
`diagnosis/who_benefits.csv`). A mean over 43 personalised models can be produced by a uniform
small gain or by a handful of large ones, and here it is closer to the second: 33 of 43
participants improve, median **+0.033**, with a long right tail (M8 +0.235, M34 +0.218,
M2 +0.196) and ten small negatives (worst M27, −0.047).

Those ten are not failures, and the distinction matters:

* **No participant is significantly hurt.** Each participant's gap is averaged over 10 splits
  with a per-participant SE of 0.015–0.047, and every negative point estimate sits inside its
  own noise (*p* from 0.27 to 0.99). Ten participants are significantly *helped*; the remaining
  33 are indistinguishable from zero, which includes both genuine small gains and genuine nulls.
* **But the heterogeneity is partly real.** The split-half rank correlation of the
  per-participant gap is **+0.49**, implying a Spearman-Brown reliability of **0.66** for the
  10-split average: about two thirds of the between-participant variance is signal. So "who it
  helps" is a legitimate question, while any individual participant's rank is not worth much.

**What predicts it.** One variable survives a standardised OLS against block count, positive
rate and how often random happened to buy the target's own blocks: **how isolated the
participant's own blocks are in feature space** (mean distance to their 20 nearest neighbours;
*r* = +0.40, *p* = 0.008; *t* = +2.47 adjusted). The model explains $R^2 = 0.21$ of the raw
variance, i.e. about a third of the *reliable* variance. The reading is the one §6.1 argues
from the other direction: if a participant's sensor signature is unlike the cohort's, the
population fit is wrong for them and labels bought from strangers do not transfer, so spending
the budget on their own blocks pays; if they are typical, the pooled model already fits them and
self-querying adds little. Participants with more mixed, boundary-heavy blocks also gain more
(*r* = +0.38, *p* = 0.013), consistent with §2.6's finding that such blocks are informative
rather than noise.

**Two mechanisms that are not the explanation**, tested rather than assumed. *"Pool-wide random
already buys the target's own blocks, so large contributors have less to gain"* — false: random
buys between 0.1 and 1.7 of its 30 labels from the target even for the largest contributor
(492 of ~11 000 blocks), so there is nothing to crowd out, and the correlation with the gap is
−0.16 (*p* = 0.32). *"Participants with almost no positives of their own cannot benefit from
self-querying"* — also false as a general rule: the seven participants under an 8% positive rate
average +0.052 against +0.061 for the rest (*p* = 0.81), and two of them (M41, M43) are among
the largest gainers. The single worst participant, M30, does have a 0.6% positive rate, but one
case is not a mechanism.

**The shape of the curves** (`fig_curves.png`) is as informative as the endpoint. `twostage`
rises steeply through its calibration phase (0.600 → 0.636 over the first 6 purchased labels) and
then more slowly through Phase 2. Random **falls** over its first 6 labels — 0.5997 → 0.5872 — and
does not regain its 3-label starting point until roughly 30 labels. The dip appears in **8 of the
10 splits** (mean −0.011), though on its own it is only suggestive — a t-test over the per-split
means gives *p* = 0.09. Read together with §2.4 it is the expected signature: the first handful
of labels bought from scattered strangers leaves the personalised model no better than the
3-label seed it started from, because they dissolve the random-effect structure without buying
enough population information to compensate.

### 5.2 Axis 2 — participant burden equity

Cohort-level burden at the full budget, pooled over all 43 personalised models:

| | `twostage` | `random` | pool-proportional reference |
|---|---|---|---|
| Gini $G_q$ | 0.293 (SE 0.005) | 0.287 (SE 0.007) | 0.275 |
| max-to-mean query ratio | **4.02** (SE 0.08) | 2.04 (SE 0.05) | 1.94 |
| participants ever asked | 43.0 of 43 | 43.0 of 43 | 43 |
| heaviest participant's share | 9.4% | 4.7% | 4.5% |
| share of labels from the target | 31.3% | 2.2% | — |

Two findings, and the second is a cost rather than a benefit:

* **On the Gini coefficient the framework is indistinguishable from random**, and both sit at the
  pool's own inequality (0.275), which no unbiased sampler can remove. The accuracy gain in §5.1
  is therefore *not* paid for in cohort-level burden concentration: it asks every participant,
  and it asks them about as unevenly as uniform sampling does.
* **On the max-to-mean ratio it is worse than random** — 4.02 against 2.04. The heaviest-burdened
  participant carries four times the average load rather than twice. The two statistics
  disagreeing is the point of reporting both: Phase 1 spreads perfectly (every participant
  answers exactly 6 questions about their own model), while Phase 2's 24 pool-wide labels land
  unevenly enough to double the extreme. If a deployment cares about the worst-off individual
  rather than the distribution as a whole, that is the number to watch, and §7 lists the
  constrained variant that would address it directly.

### 5.3 The control: how much of this is the framework, and how much is "ask the target"?

`twostage` beats pool-wide random sampling, but it differs from it in *two* ways at once: it
restricts most of its queries to the target participant, and it chooses which blocks by a design
criterion. A second random arm separates them — **`self_random`**, uniform sampling restricted to
the target's own unlabelled blocks, same budget, same cells, same seeds:

| arm | what it buys | gap vs pool `random` | gap vs `self_random` |
|---|---|---|---|
| `random` | uniform over the pool | — | −0.0709 |
| **`self_random`** | **uniform over the *target's* blocks** | **+0.0709** (SE 0.0085) | — |
| `calib` | D-optimal over the target's blocks | +0.0794 (SE 0.0076) | +0.0085 (*p* = 0.20) |
| `twostage` | 6 D-optimal self + 24 TW-FEG | +0.0598 (SE 0.0069) | −0.0111 (*p* = 0.19) |

This is the most important table in the document, and it is not flattering to the framework.

* **89% of the best arm's entire advantage is available from "label the target's own blocks, at
  random".** No posterior, no design criterion, no density ratio — just the decision to ask the
  target rather than the pool.
* **The D-optimal criterion adds +0.0085 Macro PR-AUC over picking self-blocks at random, and
  that is not significant** (Wilcoxon *p* = 0.16, split-level *p* = 0.20, win rate 51%). On
  AUPRC it is +0.0222 (*p* = 0.05); on AUROC it is **−0.0066**. So Phase 1's value is almost
  entirely in *whom* it asks, not in *which* of their blocks it picks.
* **The full two-phase strategy is not better than the trivial control**: −0.0111 Macro PR-AUC
  (*p* = 0.19) and −0.0216 AUROC (Wilcoxon *p* = 0.005) against `self_random`. Phase 2's 24
  pool-wide labels cost more than they return at this budget.

The honest summary of the benchmark is therefore: *the framework beats the acquisition baseline
it was designed against (+0.060 over pool-random, p < 1e-4), and loses to a one-line baseline
nobody would call an acquisition function.* Whatever is to be salvaged here is in §5.4's sweep
and in the budget regime §7 identifies — not in the 33-label result.

### 5.4 Where the gain comes from: the phase boundary

After its 6 calibration labels, `twostage` is already **+0.0486** ahead of random at the same
budget — **81% of its final advantage of +0.0598**, bought with 20% of the budget. The remaining
24 labels of Phase 2 add +0.011.

**The sweep** (`fig_ncalib.png`, `ncalib_curve.csv`). Eight values of $n_{\text{calib}}$ from 0
(TW-FEG only) to 30 (calibration only), each a full 10 splits × 43 targets, each paired against
random on the identical cells:

| Phase 1 budget | gain vs random | Gini $G_q$ | max-to-mean | self-share |
|---|---|---|---|---|
| 0 (`twfeg`) | +0.0490 (SE 0.0061) | 0.347 | 4.46 | 0.17 |
| 3 | +0.0591 (0.0062) | 0.318 | 4.15 | 0.24 |
| **6 (`twostage`, as proposed)** | **+0.0598 (0.0069)** | 0.293 | 4.02 | 0.31 |
| 9 | +0.0626 (0.0085) | 0.267 | 3.83 | 0.39 |
| 12 | +0.0644 (0.0075) | 0.242 | 3.67 | 0.47 |
| 18 | +0.0720 (0.0065) | 0.182 | 3.22 | 0.64 |
| 24 | +0.0757 (0.0080) | 0.112 | 2.57 | 0.82 |
| 30 (`calib`) | +0.0794 (0.0076) | 0.004 | 1.01 | 1.00 |

The curve is **monotone over the entire range, on both axes at once**. Every label moved from
Phase 2 to Phase 1 improves the target's accuracy *and* flattens the cohort's burden; there is no
interior optimum and no accuracy-equity trade-off to tune. Two consequences:

* The proposal's prescription — 5 to 10 calibration queries, sized by $\dim\mathbf{b}_T = 6$ —
  sits **at the bottom of the tested range**: only the settings that spend *even less* on Phase 1
  (0 and 3) do worse, and the gap between the prescribed setting and the best one is +0.0196,
  more than three times its own standard error. The dimension of the random-effect block is not
  the right scale for the phase boundary, for the reason §2.4 measures: that block is not being
  identified at this budget, so there is nothing for 6 labels to finish.
* The schedule only overtakes the `self_random` control (+0.0709, dashed line in
  `fig_ncalib.png`) at $n_{\text{calib}} \approx 18$ — that is, only once it has become mostly
  Phase 1 — and even at 30 the margin over the control is +0.0085 and not significant (§5.3).



### 5.5 Does Phase 2 do what it was designed to do?

Yes — which is what makes §5.3 and §5.4 a finding about the problem rather than a bug in the
criterion. For every target, rank
all ~11 000 pooled blocks by the estimated density ratio $r(x) = p_T(x)/p_{\text{pool}}(x)$ and
ask what percentile each arm's purchases sit at, excluding self-queries so the answer is about
the *pool-wide* phase only (`manifold_check.py`, splits 0–1, 43 targets):

| arm | percentile of the target's density ratio (non-self purchases) |
|---|---|
| `twostage` | **76.4** (SE 1.1) |
| `random` | 48.4 (SE 0.6) |

Phase 2's purchases sit three quarters of the way up the target's own density manifold while
random's sit exactly at the median, which is what the criterion was built to do. The density
re-weighting is working; it is the *value of pool labels at this budget* that is small, for the
reasons measured in §2.4 and argued in §6.1 — not the targeting.

### 5.6 The singleton-marginalisation refinement

§2.5 derived an exact discount for a stranger's *first* label — the factor
$1/(1 + w\,\mathbf{z}'\mathbf{D}\mathbf{z})$ by which that observation's $\boldsymbol{\beta}$
information is attenuated once the stranger's own random effect is marginalised out — and noted
that its size (mean 0.77, SD 0.17) says nothing about whether it *reorders* anything.

Benchmarked on the same 10 splits, it does not. `twostage_marg` against `twostage` is
**+0.0006** Macro PR-AUC (SE 0.0022, *p* = 0.80); `twfeg_marg` against `twfeg` is **−0.0028**
(*p* = 0.37). The discount is exact, its level is substantial, but its *spread across candidates*
is small next to the orders of magnitude the TW-FEG cross term spans, so it almost never changes
which candidate is the argmax. A correct derivation with no effect is worth reporting as such:
the mechanism in §2.4 is real, but this is not the lever that exploits it — §7's clustered-spending
proposal is the one that would.

### 5.7 Summary of what held

| claim | verdict |
|---|---|
| The framework beats equal-budget random on the target's threshold-free performance | **held** — +0.060 Macro PR-AUC, +0.097 AUPRC, *p* < 1e-4 on both tests |
| ...without concentrating the cohort's labelling burden | **held on $G_q$** (0.293 vs 0.287, both at the pool's own 0.275); **not held on the max-to-mean ratio** (4.02 vs 2.04) |
| Phase 1 anchors the target in 5–10 queries | **held as a budget, not as a mechanism** — 6 labels deliver 81% of the final gain, but §2.4 shows they are not identifying $\mathbf{b}_T$ |
| The D-optimal criterion is what makes Phase 1 work | **not held** — +0.0085 over uniform sampling of the same participant's blocks, *p* = 0.20 (§5.3) |
| Phase 2 targets the fixed effects along the target's density manifold | **held mechanically** (76th percentile vs random's 48th), **negative in payoff against the right control** (−0.011 Macro PR-AUC, −0.022 AUROC vs `self_random`) |
| The phase boundary belongs at 5–10 labels | **not held** — see §5.4 |
| The framework as a whole is worth its complexity at this budget | **not held** — it does not beat uniform sampling restricted to the target (§5.3) |

### 5.8 Scope of this comparison

The benchmark reported above is `twostage` against `random`, as scoped. The same runs also
contain CoreSet, ProbCover, BALD, and the framework's own ablations (`calib`, `twfeg`,
`twfeg_own`, `twostage_kmed`), all on the identical splits, seeds and cells; the diagnosis in §2
draws on their query records, and the full nine-arm tables are in
`<run>/analysis/final_*.csv` alongside this focused read-out. One result from that set bears on
how the framework should be configured and is therefore not left in a file: **Phase 1 alone
(`calib`, all 30 labels D-optimally chosen from the target) scores +0.0794 against random — above
`twostage`'s +0.0598 — at a Gini of 0.004.** §5.3 and §6.2 take that seriously rather than
burying it.

---

## 6. Discussion

### 6.1 Information is the wrong currency when the binding error is bias

Both phases of this framework — and BALD, expected error reduction, and the parent repo's
`laplace` criterion (not re-run here; its 3-split numbers are in the parent `results/`) — are
**variance-reduction** criteria. Each asks how much a label would
shrink some functional of the posterior covariance. None can see bias, because a correctly
specified model has none to see.

At a 33-label budget on a multi-subject wearable pool, the model is not, for practical purposes,
correctly specified *for the target*. The population coefficients fitted on labels bought from
strangers estimate a population relationship that differs from the target's own, and the
mechanism meant to absorb that difference — the random effects — is prior-dominated and, once the
budget is spread, collapses outright (§2.4). A stranger's label does reduce the posterior variance
of $\boldsymbol{\beta}$, exactly as TW-FEG computes, while *increasing* the bias of the prediction
it was bought to improve. The criterion is not wrong about what it measures; it measures the
component of error that is not binding.

Four independent observations line up behind that reading, and none of them is decisive alone:

1. **Random's curve falls before it rises** (§5.1), in 8 of 10 splits. The first six labels
   bought from scattered strangers leave the personalised model no better than its 3-label seed
   — suggestive on its own (*p* = 0.09), and pointed in the same direction as the rest.
2. **The variance components collapse when the budget is spread** (§2.4): fitted random-effect SD
   0.78 → 0.035 → 0.011 as the labels reach 1, 13.6 and 18.5 participants.
3. **Phase 2 aims correctly and pays little** (§5.5): its purchases sit at the 76th percentile of
   the target's density manifold, and 24 of them are worth +0.011.
4. **Across the nine arms on disk, the share of the budget spent on the target predicts the gain**
   (Pearson $r = +0.744$, $p = 0.022$, $n = 9$ arms). This one is the weakest of the four: the
   same correlation computed *within* an arm, across cells, is null ($r = -0.015$), though that
   test has almost no leverage because an arm's self-share is fixed by its schedule and barely
   varies across cells. It is consistent with (1)–(3), not independent evidence for them.

The target's own labels reduce both components: they shrink variance and they move the fitted
model towards the distribution it will be scored on. That is a statement about *the estimator and
the budget*, not about acquisition functions, which is what makes it likely to generalise beyond
this comparison.

### 6.2 What survives of the framework, and what does not

The control in §5.3 divides the proposal into a part that is right and a part that is not
carrying its weight.

**What is right is the premise about *whom* to ask.** Restricting a low budget to the target
participant is worth +0.071 Macro PR-AUC over spending it on the pool — and that is the whole
finding. It is also exactly equitable at the cohort level, since every participant answers only
for their own model. The diagnosis in §2 explains why it works and why the pool-wide alternatives
cannot: a subject-blind sampler spreads singletons that collapse the random-effect structure and
pull the fit towards a population the target does not belong to.

**What is not carrying its weight is both information criteria.** D-optimality on
$\mathbf{b}_T$ adds +0.0085 over picking the same participant's blocks uniformly, which does not
reach significance on the primary metric and is negative on AUROC; §2.4 already showed why, since
$\mathbf{b}_T$ is barely identified at this budget and a criterion that optimises its
determinant is optimising a quantity the data cannot pin down. TW-FEG is worse than that: it
aims correctly (§5.5) and still loses 0.011 Macro PR-AUC and 0.022 AUROC against the trivial
control, because the labels it aims *at* are the wrong currency (§6.1).

So the schedule parameter $n_{\text{calib}}$ is not a nuisance knob but the quantity the
experiment turned out to be about, and the answer in §5.4 is at the boundary of its range: at
this budget the framework should spend everything in Phase 1 — at which point what remains of it
is "ask the target, and use a design criterion if you like, it will not hurt much". That is a
useful result for a wearable study and a negative result for the framework as proposed.

### 6.3 Reading the two axes together

The two-axis framing earns its place, and it does not resolve into a single verdict. On accuracy
the framework is clearly ahead of equal-budget random (+0.060 Macro PR-AUC, p < 1e-4 on both
tests, 5.5× label efficiency). On *distributional* equity it is indistinguishable from random and
from the pool's own inequality — the gain is not paid for by concentrating the cohort's work. But
on the *extreme* — the single worst-off participant — it is twice as demanding as random (4.0×
the mean load against 2.0×), and that cost comes entirely from Phase 2. A deployment that cares
about the distribution should regard the framework as free; one that cares about the heaviest
individual is paying something real for Phase 2's +0.011, and §7 names the variant that would fix
it.

## 7. Limitations and what follows

**The budget is the main boundary on every conclusion.** All of this is conditioned on 33 labels,
124 features and 125 fixed effects. The prediction that follows is specific and testable: Phase
2's value should grow with budget, because the target's own blocks run out (participants
contribute 36–475 blocks, several fewer than 100), because the variance components become
identifiable once several participants hold multiple labels each, and because the bias term that
defeats stranger labels shrinks as the random effects start absorbing between-subject
heterogeneity. Nothing here tests that.

**One target label, one dataset, one blocking width.** COMPUTER_WORK at W = 15 on ExtraSensory.
The parent repo's registry makes `in_a_meeting` and `walking` one flag away, and `walking` — 46
watch-accelerometer features, five named negatives, balanced classes — is a materially different
problem where the bias argument of §6.1 should bite less hard.

**The density ratio is estimated and unvalidated on real features.** The estimator is checked
against a known log-ratio on synthetic Gaussians and its *effect* is visible in §5.5, but there is
no ground truth for $p_T/p_{\text{pool}}$ on ExtraSensory. A mis-estimated ratio would weaken
Phase 2 with no symptom other than the payoff it failed to deliver.

**The Laplace posterior is an approximation, and the covariance used is conditional on the
variance components at the mode.** Given §2.4 — variance components pinned near their prior or
collapsed — that is not a small approximation at this budget. It applies to both phases equally.

**Equity is measured, not optimised**, and §5.2 shows where that bites: the max-to-mean ratio is
the one axis on which the framework is worse than random. A per-participant cap or a
burden-penalised utility in Phase 2 is the obvious next design, and the measurements above give it
a concrete target rather than a hope.

**The comparison reported is against random sampling, in two forms.** Uniform over the pool is
the reference for "is this acquisition worth running at all"; uniform over the target's own
blocks (§5.3) is the reference for "is the machinery worth its complexity", and it is the one the
framework fails. The other arms on disk (§5.8) have not been given the same write-up scrutiny.
A reader who wants the framework defended should note that no arm tested here — geometric,
entropy-based or information-theoretic — beat `self_random` by a significant margin either; the
closest was `calib`, at +0.022 AUPRC with *p* = 0.05.

**Three concrete next steps**, in the order the evidence recommends them:

1. **Budget sweep for the phase boundary**, 33 → 300 labels, `twostage` against `random` at
   several values of $n_{\text{calib}}$. The crossover, if it exists, is what would make Phase 2
   worth its labels.
2. **Bias-aware Phase 2.** Weight each labelled stranger row by its density ratio *in the
   likelihood*, not only in the acquisition — importance-weighted estimation, which turns the
   manifold argument from a selection rule into an estimator, and attacks §6.1 directly.
3. **Identifiability-aware spending.** Buy strangers' labels in small *clusters* per participant
   rather than singletons, so the variance components stay identified. §2.4 predicts a
   qualitative change in behaviour at 2–3 labels per participant, and a burden cap falls out of
   the same design.

---

## 8. Figures

Two directories, both under the repo root. `RUN` below is the nine-arm run folder,
`two_stage/results/computer_work/computer_work_w15_bald-calib-coreset-probcover-random-twfeg-twfeg_own-twostage-twostage_kmed_r10x3_s5`.

**The headline comparison** — `RUN/analysis_twostage_vs_random/`, the two-arm read-out:

| file | what it shows | section |
|---|---|---|
| `fig_curves.png` | Macro PR-AUC against labels bought, `twostage` vs `random`, SE band over splits. Random's early dip is visible here | §5.1 |
| `fig_curves_per_person.png` | **the same curves, one panel per participant**, ordered by final gap with the cohort mean last. The pooled curve hides how many *individuals* are helped: 33 of 43 here, median +0.033, best +0.235 (M8), worst −0.047 (M27) | §5.1 |
| `fig_ncalib.png` | **the sweep**: gain against the Phase 1 budget, 0 → 30, with the `self_random` control as a dashed line and the crossover at ~18 | §5.4 |
| `fig_gap.png` | final-budget paired gain per arm with split-level *p* in the labels | §5.1, §5.3 |
| `fig_plane.png` | the equity-performance plane: gain against Gini $G_q$, with the pool-proportional reference | §5.2 |
| `fig_selfshare.png` | each arm's self-share against its gain — the between-arm relationship of §6.1 | §6.1 |

**The diagnosis** — `two_stage/diagnosis/`:

| file | what it shows | section |
|---|---|---|
| `fig_row_profile.png` | what each method buys: effect size per attribute against the pool | §2.2 |
| `fig_burden.png` | Gini, max-to-mean and coverage against budget, all arms, with the pool-proportional line | §2.3 |
| `fig_lorenz.png` | Lorenz curves of labelling burden at the full budget | §2.3 |
| `fig_isolation.png` | per-participant queried share ÷ pool share against participant isolation | §2.3 |

`RUN/analysis/` holds the same five figures computed over all nine arms rather than two; the
`analysis_twostage_vs_random/` copies are the ones this document cites. Every figure has its
numbers beside it as a `.csv` in the same directory. Figures are regenerated by the commands in
§9 and are gitignored (`*.png`); the CSVs are kept.

## 9. Reproducibility

```bash
# the arms (one process per split; PAR is limited by RAM, see README -- the probcover arm holds
# an 11k x 11k distance matrix per target and PAR=6 swaps on a 16 GB machine)
SPLITS=10 PAR=3 bash two_stage/run_all.sh                        # the 9-arm run   (~1 h)
METHODS=self_random SPLITS=10 PAR=5 bash two_stage/run_all.sh    # the control     (~5 min)
METHODS="twfeg_marg twostage_marg" SPLITS=10 PAR=3 bash two_stage/run_all.sh
SPLITS=10 PAR=5 NCALIBS="3 9 12 18 24" bash two_stage/run_ncalib.sh   # the sweep  (~25 min)

# section 2
./.venv/bin/python two_stage/diagnose.py --runs <the 9-arm run folder>
./.venv/bin/python two_stage/probcover_delta.py --splits 0 1
./.venv/bin/python two_stage/vc_diagnostic.py --targets 8 --self_frac 1.0   # and 0.5, 0.2
./.venv/bin/python two_stage/manifold_check.py --splits 0 1 --runs <the 9-arm run folder>

# sections 5-6: the headline comparison, with the control and the sweep merged in
./.venv/bin/python two_stage/analyse.py --run <9-arm> --also <nc3> <nc9> <nc12> <nc18> <nc24> \
    <self_random> --methods twostage calib twfeg self_random

# the algebra of both criteria, the singleton identity, and the schedule's routing
./.venv/bin/python two_stage/selftest.py
```

Two notes for anyone re-running this. `diagnose.py` and `analyse.py` take **explicit run
folders** rather than globbing, because the sweep writes an arm called `twostage` once per
$n_{\text{calib}}$ and de-duplication keys on the method name — globbing would pool eight
different schedules into one row. And the parent repo's pre-seed-fix `random` arm replays a
single draw of 30 rows for all 43 targets, which leaves its attribute means unbiased but its
burden statistics meaningless; every number in §2 and §5 comes from the runs in this
subdirectory.

Tables and figures referenced above are written to `two_stage/diagnosis/` and
`<run>/analysis/`; see [README.md](README.md) for the file map and the output layout. Library
versions, every command-line argument and the timestamp of each run are recorded in that run's
`config.json`.

## References

* G. Hacohen, A. Dekel, D. Weinshall. *Active Learning on a Budget: Opposite Strategies Suit High
  and Low Budgets.* ICML 2022.
* N. Houlsby, F. Huszár, Z. Ghahramani, M. Lengyel. *Bayesian Active Learning for Classification
  and Preference Learning.* arXiv:1112.5745, 2011.
* A. Kirsch, J. van Amersfoort, Y. Gal. *BatchBALD.* NeurIPS 2019.
* O. Sener, S. Savarese. *Active Learning for Convolutional Neural Networks: A Core-Set Approach.*
  ICLR 2018.
* M. Sugiyama, T. Suzuki, T. Kanamori. *Density Ratio Estimation in Machine Learning.* Cambridge
  University Press, 2012.
* Y. Vaizman, K. Ellis, G. Lanckriet. *Recognizing Detailed Human Context in the Wild from
  Smartphones and Smartwatches.* IEEE Pervasive Computing, 2017. (ExtraSensory)
* O. Yehuda, A. Dekel, G. Hacohen, D. Weinshall. *Active Learning Through a Covering Lens.*
  NeurIPS 2022. (ProbCover)
