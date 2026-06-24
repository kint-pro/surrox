# ADR-026: Model-Agnostic Permutation Shapley Instead of Tree SHAP

## Status

Accepted

## Date

2026-06-24

## Context

The Analysis layer produces additive feature attributions so that kint's customers — who act on surrox recommendations to operate critical infrastructure — can trace why a recommendation was made. The original implementation used `shap.TreeExplainer`, creating one explainer per ensemble member and weight-averaging the results.

Two problems:

1. **Correctness.** The ensemble is heterogeneous. `EstimatorFamily` (ADR-012) already ships `gaussian_process` and `tabicl` alongside `xgboost` and `lightgbm`, and names neural networks as future families. `shap.TreeExplainer` only accepts tree models — it raises on a Gaussian Process or TabICL member. The per-member loop was silently restricted to tree-only ensembles, contradicting the ensemble design.

2. **Dependency weight.** `shap` declares an unconditional `numba>=0.54` dependency, which pulls in `llvmlite` (~112 MB of native build artifacts). surrox is a published library; this dominates the install footprint while only `TreeExplainer` was ever used.

### Options

**Option A — Per-family explainer dispatch.** Trees via native `pred_contribs`, GP/TabICL via a model-agnostic fallback, then weight-average. Still requires a model-agnostic path for non-tree members, so it adds a branch without removing the hard part.

**Option B — Model-agnostic Shapley on the aggregate prediction.** Treat the ensemble as one black box `f(x) = Σ wₖ·modelₖ(x)` (already exposed as `Ensemble.predict`) and compute Shapley values on it. One path for every family, present and future.

## Decision

Compute interventional Shapley values model-agnostically on `Ensemble.predict`, implemented in-tree (`src/surrox/analysis/shapley.py`). Drop the `shap` dependency.

- **Exact** by full coalition enumeration when `n_features ≤ shap_exact_threshold` (default 12).
- **Antithetic permutation sampling** above the threshold, reporting a per-value standard error.
- Both modes satisfy the efficiency property `base_value + Σφ = f(x)` exactly.

## Rationale

- **Correct for heterogeneous ensembles.** Shapley values are linear, so explaining the weighted aggregate equals weight-averaging exact per-member values for the tree case, while also covering GP, TabICL, and future families through a single code path.
- **No tree assumption, no dispatch.** Consistent with the family-agnostic training pipeline (ADR-012).
- **Removes `shap` → `numba` → `llvmlite` (~112 MB)** from the dependency tree. Like ADR-010 (conformal from scratch), the algorithm is small and standard; vendoring shap's C++ TreeSHAP would have added a build step and a tree-only constraint.
- **Honest uncertainty for critical infrastructure.** Exact below the threshold; above it, the reported standard error makes approximation explicit rather than silent. The threshold is user-configurable because exact cost grows as `2^n_features` and sampling cost as `2 · permutations · n_features`.

## Consequences

- New `AnalysisConfig` fields: `shap_exact_threshold`, `shap_sampling_permutations`.
- `ShapGlobalResult` and `ShapLocalResult` gain an optional `standard_error` field (`None` when exact).
- `shap` is removed from the `ml` optional dependencies; `numba`, `llvmlite`, `cloudpickle`, and `slicer` drop from the lockfile.
- Global SHAP over a large background with many features is more expensive than TreeSHAP was; cost is bounded by `shap_background_size` and `shap_exact_threshold`.
- A model-agnostic fast path using native `pred_contribs` for tree-only ensembles may be added later purely as an optimization, without changing the contract.
