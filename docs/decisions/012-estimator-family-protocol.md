# ADR-012: EstimatorFamily as a Protocol for Extensible Estimator Support

## Status

Accepted

## Date

2026-02-26

## Context

The SurrogateManager trains models using XGBoost and LightGBM. The init prompt explicitly lists future estimator families: CatBoost, TabPFN, and neural networks. Each estimator family differs in three ways:

1. **Hyperparameter search space**: XGBoost has `max_depth`, `colsample_bytree`; LightGBM has `num_leaves`, `feature_fraction`; CatBoost would have `depth`, `l2_leaf_reg`.
2. **Monotonicity constraint format**: XGBoost takes a `dict[str, int]`, LightGBM takes a positional `list[int]`, CatBoost takes a different format entirely.
3. **Model creation**: Different constructors, different parameter names, different sklearn-compatibility levels (TabPFN has no standard sklearn interface).

### Options

**Option A — Conditional branches:**
`if family == "xgboost": ... elif family == "lightgbm": ...` in the objective function, constraint mapping, and model creation. Works for 2 families. Becomes unmaintainable at 5+.

**Option B — EstimatorFamily Protocol:**
Each family is a class implementing a common interface. The training pipeline calls protocol methods — it never knows which family it's working with. Adding a new family is adding a new class, not modifying existing code (Open-Closed Principle).

## Decision

`EstimatorFamily` is a Protocol with four methods:

- `name -> str`: unique identifier
- `suggest_hyperparameters(trial) -> dict`: define the Optuna search space
- `build_model(hyperparameters, monotonic_constraints, random_seed) -> BaseEstimator`: create a configured, unfitted model
- `map_monotonic_constraints(constraints, feature_names, categorical_features) -> dict[str, int]`: map from surrox representation to native format

Two built-in implementations: `XGBoostFamily` and `LightGBMFamily`. The list of families is configurable via `TrainingConfig.estimator_families`.

## Rationale

- **One new class per new estimator**: No existing code changes needed. The training pipeline's objective function, ensemble construction, and conformal calibration are all family-agnostic.
- **Testable in isolation**: Each family can be unit-tested independently — its search space, model creation, and constraint mapping.
- **Optuna integration is clean**: The family name becomes a `suggest_categorical` value. Optuna's TPE sampler handles conditional hyperparameter spaces natively.
- **Not premature abstraction**: The init prompt names 3 additional estimator families. The interface has exactly the methods needed by the training pipeline — no speculative methods.

## Consequences

- `EstimatorFamily` is a Protocol (typing.Protocol), not an ABC. No inheritance required — structural subtyping is sufficient.
- `TrainingConfig.estimator_families` replaces the implicit "always XGBoost + LightGBM" assumption. Defaults to `(XGBoostFamily(), LightGBMFamily())`.
- `TrialRecord.estimator_family` and `EnsembleMember.estimator_family` store the family name as `str`, not the Protocol instance.
- `EnsembleMember.model` is typed as `BaseEstimator` (sklearn), not as a union of specific types.
- Adding CatBoost in Phase 2 means: implement `CatBoostFamily`, add it to `estimator_families`. No pipeline changes.
