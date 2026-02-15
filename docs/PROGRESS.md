# PyStatsBio — Development Progress

**Package:** pystatsbio
**Version:** 0.1.0-dev
**Target:** Phase 1 — Wedge Release (~4,000 lines)

---

## Metrics

| Metric | Current |
|--------|---------|
| Source lines (pystatsbio/) | 4,689 |
| Test lines (tests/) | 1,959 |
| Total tests | 248 |
| Tests passing | 248 |
| Phase 1 modules | 3/4 complete |

---

## Phase 1 Modules

### 1. `power/` — Sample Size and Power Calculations ✅

**Status:** Complete
**Files:** 8 implementation + 7 test files
**Tests:** 95 passing

Covers the core trial-planning workflows: two-sample/one-sample/paired t-tests, proportion tests (chi-squared + Fisher exact), log-rank survival (Schoenfeld, Freedman, Lachin-Foulkes), one-way and factorial ANOVA, non-inferiority/equivalence/superiority for means and proportions, 2×2 crossover bioequivalence, and cluster-randomized trials.

Every function follows the "solve for any one" pattern — pass `None` for exactly one of `n`, `d`/`h`/`hr`, or `power` and the function solves for it.

| File | Description |
|------|-------------|
| `_common.py` | `PowerResult` frozen dataclass with `.summary()` |
| `_means.py` | `power_t_test`, `power_paired_t_test` — noncentral-t with normal fallback |
| `_proportions.py` | `power_prop_test`, `power_fisher_test` — Cohen's h, simulation-based Fisher |
| `_survival.py` | `power_logrank` — Schoenfeld / Freedman / Lachin-Foulkes |
| `_anova.py` | `power_anova_oneway`, `power_anova_factorial` — noncentral F |
| `_noninferiority.py` | `power_noninf_mean`, `power_noninf_prop`, `power_equiv_mean`, `power_superiority_mean` |
| `_crossover.py` | `power_crossover_be` — 2×2 crossover bioequivalence via TOST |
| `_cluster.py` | `power_cluster` — ICC-adjusted design effect |

**Notable decisions:**
- Noncentral-t CDF can return NaN for extreme parameters → fallback to normal approximation
- Crossover n forced even (subjects split across sequences)
- Cluster power delegates to power_t_test internally after inflating n by design effect

---

### 2. `doseresponse/` — Dose-Response Modeling ✅

**Status:** Complete
**Files:** 7 implementation + 4 test files
**Tests:** 70 passing

The primary GPU showcase. Five model functions matching R's `drc` package parameterization, single-curve and batch fitting, EC50/BMD analysis.

| File | Description |
|------|-------------|
| `_models.py` | `ll4`, `ll5`, `weibull1`, `weibull2`, `brain_cousens` + model registry |
| `_common.py` | `CurveParams`, `DoseResponseResult`, `BatchDoseResponseResult` |
| `_fit.py` | `fit_drm` — self-starting params + scipy `least_squares` (TRF) |
| `_potency.py` | `ec50` (delta method CI on log-scale), `relative_potency` (Fieller's theorem) |
| `_bmd.py` | `bmd` — benchmark dose with BMDL/BMDU via delta method |
| `_batch.py` | `fit_drm_batch` — CPU loop + GPU batched Levenberg-Marquardt in PyTorch |
| `__init__.py` | Public API exports |

**Notable decisions:**
- Hill convention: `hill > 0` = increasing curve (opposite of `drc`'s `b` parameter where `b < 0` = increasing)
- `dose=0` handled via IEEE 754: `log(0) = -inf` propagates correctly through all model formulas
- GPU path parameterizes EC50 on log-scale for unconstrained optimization
- MPS (Apple Silicon) gets float32 with adapted numerical constants (wider finite-difference step, relaxed convergence tolerance)
- Self-starting algorithm: interpolated EC50 on log-dose scale + logit-linear hill estimate
- Standard errors from Jacobian: `se = sqrt(diag((J'J)^{-1} * s²))`

**GPU batch algorithm:**
- Batched Levenberg-Marquardt across K compounds simultaneously
- 4 forward passes per iteration for finite-difference Jacobian
- Batched `torch.linalg.solve` for normal equations
- Per-compound lambda damping with accept/reject
- ~100 iterations max, convergence on relative RSS change

---

### 3. `diagnostic/` — Diagnostic Accuracy ✅

**Status:** Complete
**Files:** 6 implementation + 4 test files
**Tests:** 83 passing

ROC analysis with DeLong CIs (logit-transformed), DeLong test for comparing correlated ROC curves, diagnostic accuracy at fixed cutoff (sensitivity, specificity, PPV, NPV, LR+/LR−, DOR), optimal cutoff selection (Youden, closest-to-topleft, cost-based), and batch AUC for HTS biomarker panels (CPU + GPU).

| File | Description |
|------|-------------|
| `_common.py` | `ROCResult`, `DiagnosticResult` frozen dataclasses with `.summary()` |
| `_roc.py` | `roc()` — empirical ROC + AUC via Mann-Whitney + DeLong SE + logit CI; `roc_test()` — paired DeLong test via covariance of placement values |
| `_accuracy.py` | `diagnostic_accuracy()` — 2×2 table metrics: sens/spec (Clopper-Pearson or Wilson CI), PPV/NPV (Bayes prevalence adjustment), LR+/LR−, DOR (Woolf log-scale CI with Haldane correction) |
| `_cutoff.py` | `optimal_cutoff()` — Youden J, closest-to-topleft, cost-weighted |
| `_batch.py` | `batch_auc()` — column-wise rank-based AUC + DeLong SE; CPU via scipy.stats.rankdata, GPU via PyTorch batched argsort |
| `__init__.py` | Public API exports including `ROCTestResult`, `CutoffResult`, `BatchAUCResult` |

**Notable decisions:**
- AUC via rank-based Mann-Whitney U: `AUC = (sum_case_ranks - n1*(n1+1)/2) / (n1*n0)`
- DeLong placement values computed from rank differences: `V10[i] = (pooled_rank_i - within_rank_i) / n0`
- CI on logit-transformed scale (matching pROC default): `logit(AUC) ± z * SE/[AUC*(1-AUC)]`
- Direction auto-detection via median comparison of cases vs controls
- Diagnostic odds ratio uses Haldane-Anscombe 0.5 correction when any cell is zero
- GPU batch AUC: midrank tie correction via per-column sorted-value scan

---

### 4. `pk/` — Non-Compartmental PK Analysis 🔧

**Status:** Stubs only
**Files:** Stub files, 0 test files
**Tests:** 0

AUC (linear/log-linear/lin-up-log-down), Cmax/Tmax, half-life, clearance, Vz, PK summary statistics.

**R packages to match:** PKNCA, NonCompart

---

## Bugs Fixed

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| Noncentral-t NaN | `scipy.stats.nct.sf()` returns NaN for extreme df/nc | Fallback to normal approximation when NaN detected |
| MPS float64 crash | Apple's MPS doesn't support float64 tensors | Device-dependent dtype selection (float32 on MPS) |
| Weibull2 monotonicity test | W2.4 with positive hill is decreasing (sign convention) | Test checks all-diffs-same-sign instead of strictly increasing |
| GPU LM ascent direction | Residual Jacobian used with model-Jacobian update formula | Switched to model Jacobian `∂f/∂θ` so `θ + δ` descends correctly |
| GPU float32 convergence | Tolerance 1e-8 too tight for float32 (7 digits precision) | Adaptive constants: `eps_fd=1e-3`, `tol=1e-5`, wider lambda bounds |

---

## Architecture

```
SGC-Bio (Web App)       ← user-facing UI, tables, reports, GPU infra
    ↓
PyStatsBio (Package)    ← biotech/pharma statistical methods (this package)
    ↓
PyStatistics (Package)  ← general statistical computing
```

**Machines:**
- Powerhouse (Mac Studio M2 Max, 96GB) — primary development, MPS GPU
- Forge (AMD Ryzen 5 7600X, RTX 5070 Ti, Ubuntu 24.04) — CUDA GPU, CI target

---

*Last updated after diagnostic/ module completion — 3/4 Phase 1 modules done.*
