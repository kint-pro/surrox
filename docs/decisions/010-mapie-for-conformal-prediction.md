# ADR-010: MAPIE for Conformal Prediction

## Status

Accepted

## Date

2026-02-26

## Context

The SurrogateManager needs calibrated prediction intervals with statistical coverage guarantees. Split conformal prediction is the chosen method. The question is whether to implement it from scratch (~15 lines) or use MAPIE, a scikit-learn-contrib library.

**Option A — From scratch:**
The core algorithm is simple: compute sorted absolute residuals on a calibration set, then add/subtract the quantile to point predictions. Implementation is ~15 lines of NumPy. No additional dependency.

**Option B — MAPIE (v1.3.0):**
`SplitConformalRegressor` with `prefit=True`. Requires a sklearn-compatible wrapper (~10 lines) for custom ensembles. Dependencies: numpy, scikit-learn, scipy — all already transitive dependencies of XGBoost/LightGBM.

### The subtle edge cases

The coverage guarantee of conformal prediction depends on getting the implementation details exactly right:

- **Finite-sample correction**: The quantile level must be `ceil((n+1)*(1-α))/n`, not `1-α`. Using the naive level breaks the coverage guarantee.
- **Quantile method**: Must use `method='higher'` (round up), not `'linear'` (interpolate). Interpolation can produce anti-conservative intervals.
- **Tied residuals**: Handled correctly by `method='higher'`, but easy to overlook.
- **Minimum sample validation**: Small calibration sets can push `q_level` to 1.0, making intervals impractically wide.

These are exactly the kind of bugs that pass unit tests but silently degrade coverage in production — the interval looks reasonable but doesn't deliver the claimed 90%.

## Decision

Use MAPIE for conformal prediction. Wrap the custom ensemble in a sklearn-compatible adapter.

## Rationale

- **Correctness is critical**: Conformal intervals are what users base trust decisions on. A subtle bug in the quantile computation silently breaks coverage guarantees without visible errors. MAPIE is peer-reviewed, tested, and maintained by the scikit-learn-contrib community.
- **Zero additional dependency cost**: numpy, scikit-learn, and scipy are already transitive dependencies of XGBoost and LightGBM. MAPIE adds no new dependency tree.
- **Multiple coverage levels**: MAPIE supports querying multiple coverage levels (e.g., 80%, 90%, 95%) in a single `predict_interval` call without recalibration. A from-scratch implementation would need to recompute quantiles per level.
- **Future extensibility**: MAPIE supports asymmetric conformity scores (residual-normalized, gamma), cross-conformal (jackknife+), and Mondrian conformal (conditional coverage by group). These are potential Phase 2 features that would require significant implementation effort from scratch.
- **The wrapper is minimal**: The sklearn adapter for the custom ensemble is ~10 lines — less than the from-scratch conformal implementation itself.

## Consequences

- MAPIE is added as a required dependency (not optional).
- The custom ensemble must be wrapped in a class inheriting from `sklearn.base.RegressorMixin, BaseEstimator` with `predict(X)` and `__sklearn_is_fitted__()`.
- `SplitConformalRegressor(prefit=True)` is used — MAPIE never trains the model, only calibrates intervals.
- The `ConformalCalibration` entity in the spec wraps MAPIE's `SplitConformalRegressor` rather than storing raw calibration scores directly.
