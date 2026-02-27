# Analysis Layer — Specification

## Purpose

The Analysis layer produces structured diagnostic data from the optimization results, surrogate models, and problem definition. It returns data structures only — no visualizations, no text interpretation. kint visualizes and interprets.

## Architecture

Two entry points, two responsibilities:

1. **`analyze()`** — Single-run analysis. Returns a frozen `AnalysisResult` (Summary only) and an `Analyzer` (stateful, holds lazy/cached detail analyses).
2. **`compare_scenarios()`** — Cross-run scenario comparison. Operates on multiple `OptimizationResult` objects.

### Why separate `AnalysisResult` and `Analyzer`

All other layers return frozen/immutable results (ADR-002). Making `AnalysisResult` mutable to hold caches would break this pattern and trigger known Pydantic bugs with `cached_property` on frozen models (ADR-019). Instead:

- `AnalysisResult` is a frozen Pydantic model containing only the Summary.
- `Analyzer` is a plain Python class that holds the inputs and a `dict`-based cache for detail analyses. It computes detail analyses lazily on first access and caches them. `Analyzer` is not thread-safe — each user/session gets its own instance.

## Inputs

| Input | Source | Description |
|-------|--------|-------------|
| ProblemDefinition | Problem layer | Variables, objectives, constraints, monotonic relations, scenarios |
| SurrogateManager | Surrogate layer | Trained ensembles, trial histories, uncertainty evaluation |
| OptimizationResult | Optimizer layer | Feasible/infeasible points, compromise index, hypervolume |
| BoundDataset | Problem layer | Training data for baseline comparison and SHAP background |

## Entry Points

### `analyze()`

```python
def analyze(
    optimization_result: OptimizationResult,
    surrogate_manager: SurrogateManager,
    bound_dataset: BoundDataset,
    config: AnalysisConfig | None = None,
) -> tuple[AnalysisResult, Analyzer]
```

The `ProblemDefinition` is accessed via `optimization_result.problem`.

Returns a tuple: the frozen result (Summary) and the stateful analyzer (detail analyses).

### `compare_scenarios()`

```python
def compare_scenarios(
    results: dict[str, OptimizationResult],
    problem: ProblemDefinition,
) -> ScenarioComparisonResult
```

Cross-run comparison. Operates on already-computed optimization results from multiple scenario runs. Does NOT run additional optimizations. Extracts recommended solutions (compromise for multi-objective, best for single) from each `OptimizationResult` and compares decision variable values across scenarios.

## Configuration

### AnalysisConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| shap_background_size | int | 100 | Max samples from training data for SHAP TreeExplainer background (interventional mode) |
| pdp_grid_resolution | int | 50 | Number of grid points for PDP/ICE computation |
| pdp_percentiles | tuple[float, float] | (0.05, 0.95) | Feature range percentiles for PDP grid |
| monotonicity_check_resolution | int | 50 | Grid points per variable for monotonicity validation |

**Validation rules:**
- shap_background_size >= 10
- pdp_grid_resolution >= 10
- monotonicity_check_resolution >= 10
- 0 < pdp_percentiles[0] < pdp_percentiles[1] < 1

## Domain Entities

### AnalysisResult

Frozen Pydantic model. Contains only the Summary — automatically computed at construction time.

| Field | Type | Description |
|-------|------|-------------|
| summary | Summary | Automatically computed at construction time |

### Analyzer

Plain Python class. Holds inputs and a `dict`-based cache. Computes detail analyses lazily. Not thread-safe.

| Method | Return Type | Cached | Description |
|--------|-------------|--------|-------------|
| feature_importance(column) | FeatureImportanceResult | Yes (delegates to shap_global) | SHAP-based importance per surrogate column |
| shap_global(column) | ShapGlobalResult | Yes | SHAP values over training data per column |
| shap_local(column, point_index) | ShapLocalResult | Yes | SHAP values for a specific solution point |
| pdp_ice(variable_name, column) | PDPICEResult | Yes | Partial dependence + ICE for one decision variable |
| trade_off() | TradeOffResult | Yes | Marginal trade-offs along Pareto front (multi-objective only) |
| what_if(variable_values) | WhatIfResult | No | Predict objectives + constraints for arbitrary inputs |

Detail analysis methods raise `AnalysisError` for invalid inputs (e.g., column not found, single-objective for trade-off).

**Caching:** Each cached detail analysis is computed on first call and stored in an internal `dict`. The cache key is derived from the method arguments. `what_if()` is not cached — each call has different variable values and the computation is cheap (single surrogate evaluation + extrapolation check).

**`feature_importance()` delegation:** `feature_importance(column)` internally calls `shap_global(column)` and derives importance from the SHAP values via `np.abs(shap_values).mean(axis=0)`. It does not maintain a separate cache — it reuses the cached `ShapGlobalResult`.

### Summary

Automatically computed at construction time.

| Field | Type | Description |
|-------|------|-------------|
| solution_summary | SolutionSummary | Best solution or Pareto front overview |
| baseline_comparison | BaselineComparison | Improvement vs. best historical data point |
| constraint_status | tuple[ConstraintStatus, ...] | Per-constraint status for the recommended solution |
| surrogate_quality | tuple[SurrogateQuality, ...] | Per-column model quality metrics |
| extrapolation_warnings | tuple[ExtrapolationWarning, ...] | Solutions flagged by extrapolation gate |
| monotonicity_violations | tuple[MonotonicityViolation, ...] | Empirical monotonicity check results |

### SolutionSummary

| Field | Type | Description |
|-------|------|-------------|
| n_feasible | int | Number of feasible solutions |
| n_infeasible | int | Number of infeasible solutions |
| best_objectives | dict[str, float] | Best value per objective across feasible solutions |
| compromise_objectives | dict[str, float] \| None | Compromise solution objective values (multi-objective, None if single or no feasible) |
| hypervolume | float \| None | From OptimizationResult |

### BaselineComparison

Compares the recommended solution (compromise for multi-objective, best for single) to the best historical data point in the training data.

| Field | Type | Description |
|-------|------|-------------|
| recommended_objectives | dict[str, float] | Objective values of the recommended solution |
| historical_best_per_objective | dict[str, float] | Best value per objective found in training data (not a single point — each objective's best independently) |
| improvement | dict[str, float] | Difference per objective (positive = improved, accounts for direction) |

For multi-objective: `historical_best_per_objective` represents a utopia point — the per-objective best values from historical data, which may come from different data points. The `improvement` shows per-objective gains relative to these individual bests. The comparison is surrogate predictions (recommended) vs. raw historical measurements (training data).

### ConstraintStatus

Enriches the existing `ConstraintEvaluation` from the Optimizer layer with analysis-specific fields.

| Field | Type | Description |
|-------|------|-------------|
| evaluation | ConstraintEvaluation | The original constraint evaluation from OptimizationResult |
| status | ConstraintStatusKind | Satisfied, active, or violated |
| margin | float | Distance to constraint boundary (positive = satisfied, negative = violated) |

The `evaluation` field provides access to `name`, `severity`, `violation`, `prediction`, `lower_bound`, `upper_bound` from the Optimizer layer without duplication.

### ConstraintStatusKind

```python
class ConstraintStatusKind(StrEnum):
    SATISFIED = "satisfied"
    ACTIVE = "active"
    VIOLATED = "violated"
```

**Status logic:**
- `SATISFIED`: violation ≤ 0 and `abs(margin) > 0.05 * max(abs(limit), 1e-8)`
- `ACTIVE`: violation ≤ 0 and `abs(margin) ≤ 0.05 * max(abs(limit), 1e-8)` (close to boundary)
- `VIOLATED`: violation > 0

The `max(abs(limit), 1e-8)` prevents division-by-zero when `limit = 0`.

### SurrogateQuality

| Field | Type | Description |
|-------|------|-------------|
| column | str | Surrogate column name |
| cv_rmse | float | Best cross-validation RMSE from trial history |
| conformal_coverage | float | Empirical coverage at default confidence level |
| ensemble_size | int | Number of models in ensemble |
| warning | str \| None | Warning message if quality is concerning (e.g., cv_rmse is large relative to target range) |

Computed from `SurrogateManager.get_ensemble(column)` (ensemble size) and `SurrogateManager.get_trial_history(column)` (best cv_rmse). Conformal coverage is computed from `SurrogateManager.get_surrogate_result(column).conformal`.

**Required SurrogateManager API extension:** `get_surrogate_result(column) -> SurrogateResult` to expose the `ConformalCalibration` object. The `SurrogateResult` already exists internally — this just makes it accessible.

### ExtrapolationWarning

| Field | Type | Description |
|-------|------|-------------|
| point_index | int | Index into feasible_points |
| distance | float | Normalized k-NN distance |
| threshold | float | Gate threshold that was exceeded |

Only for feasible solutions that are flagged as extrapolating (`is_extrapolating=True`).

### MonotonicityViolation

Empirical spot-check: for each MonotonicRelation, evaluate the ensemble along a single 1D grid of the decision variable (holding others fixed at the recommended solution values). Report if the predicted values violate the declared direction. This checks monotonicity along one slice through the feature space at the recommended solution — it is not an exhaustive proof of global monotonicity.

| Field | Type | Description |
|-------|------|-------------|
| decision_variable | str | Variable name |
| target | str | Objective or constraint name |
| declared_direction | MonotonicDirection | Expected direction |
| violation_fraction | float | Fraction of adjacent grid point pairs where direction is violated (0.0 = perfect, 1.0 = fully reversed) |
| max_reversal | float | Maximum magnitude of monotonicity reversal between adjacent grid points |

Only reported when `violation_fraction > 0`.

## Detail Analyses

### FeatureImportanceResult

SHAP-based feature importance. Derived from `ShapGlobalResult` via `np.abs(shap_values).mean(axis=0)`. `Analyzer.feature_importance(column)` internally calls `Analyzer.shap_global(column)` and extracts importance — no separate computation.

| Field | Type | Description |
|-------|------|-------------|
| column | str | Surrogate column |
| importances | dict[str, float] | Variable name → mean |SHAP| importance |
| decision_importances | dict[str, float] | Only decision variables (subset of above) |

### ShapGlobalResult

SHAP values over the training data for a single surrogate column.

| Field | Type | Description |
|-------|------|-------------|
| column | str | Surrogate column |
| feature_names | tuple[str, ...] | Feature names in order |
| shap_values | NDArray | Shape (n_samples, n_features) — SHAP values |
| base_value | float | Expected model output (baseline) |
| feature_values | NDArray | Shape (n_samples, n_features) — original input data |

**Computation:**
For each ensemble member, create `shap.TreeExplainer(member.model)`, compute `explainer(X)`. Aggregate SHAP values across members as weighted average (using ensemble member weights). `base_value` is the weighted average of member base values.

Background data: sample up to `shap_background_size` rows from training data.

### ShapLocalResult

SHAP values for a single solution point.

| Field | Type | Description |
|-------|------|-------------|
| column | str | Surrogate column |
| feature_names | tuple[str, ...] | Feature names in order |
| shap_values | NDArray | Shape (n_features,) — SHAP values for this point |
| base_value | float | Expected model output (baseline) |
| feature_values | dict[str, float] | Feature values of the explained point |
| predicted_value | float | `base_value + sum(shap_values)` |

`point_index` refers to `OptimizationResult.feasible_points`. The solution point is constructed as a DataFrame row from `EvaluatedPoint.variables` + scenario context (if applicable).

### PDPICEResult

Partial Dependence and Individual Conditional Expectation for a single decision variable and surrogate column.

| Field | Type | Description |
|-------|------|-------------|
| variable_name | str | Decision variable being varied |
| column | str | Surrogate column |
| grid_values | NDArray | Shape (n_grid,) — grid points for the varied variable |
| pdp_values | NDArray | Shape (n_grid,) — mean prediction at each grid point (PDP) |
| ice_values | NDArray | Shape (n_samples, n_grid) — per-sample prediction (ICE) |

**Computation:**
Use `sklearn.inspection.partial_dependence(estimator, X, features=[variable_name], kind="both", method="brute", grid_resolution=pdp_grid_resolution)`.

Called per ensemble member, aggregated as weighted average. Background data: sample from training data.

For categorical/ordinal variables: grid values are the declared categories (not interpolated). Use `categorical_features` parameter of `partial_dependence`.

### TradeOffResult

Marginal trade-offs along the Pareto front. Multi-objective only (≥ 2 objectives, ≥ 2 feasible points).

| Field | Type | Description |
|-------|------|-------------|
| objective_pairs | tuple[tuple[str, str], ...] | Pairs of objectives analyzed |
| marginal_rates | dict[tuple[str, str], NDArray] | Pair → marginal trade-off rates along Pareto front |
| pareto_objectives | NDArray | Shape (n_points, n_objectives) — Pareto front objective values |

**Computation:**
Sort Pareto front by first objective. For each consecutive pair of points, compute the marginal rate: `Δobj_b / Δobj_a`. Large rates indicate "steep" regions where small gains in one objective cost large losses in another.

Raises `AnalysisError` if single-objective or < 2 feasible points.

### WhatIfResult

Predict objectives and constraints for arbitrary decision variable inputs. Not cached — each call is a fresh surrogate evaluation.

| Field | Type | Description |
|-------|------|-------------|
| variables | dict[str, Any] | Input variable values |
| objectives | dict[str, WhatIfPrediction] | Per-objective predictions |
| constraints | dict[str, WhatIfPrediction] | Per-data-constraint predictions |
| extrapolation_distance | float | k-NN distance to training data |
| is_extrapolating | bool | Whether the point is outside training region |

### WhatIfPrediction

| Field | Type | Description |
|-------|------|-------------|
| predicted | float | Point prediction |
| lower | float | Conformal lower bound |
| upper | float | Conformal upper bound |
| recommended_value | float | Same prediction for the recommended solution (for comparison) |
| historical_mean | float | Mean of this column in training data |

### ScenarioComparisonResult

Compares optimization results across scenarios. Produced by `compare_scenarios()`, not by `Analyzer`.

| Field | Type | Description |
|-------|------|-------------|
| scenario_names | tuple[str, ...] | Scenarios compared |
| variable_robustness | dict[str, VariableRobustness] | Per-decision-variable robustness |

### VariableRobustness

| Field | Type | Description |
|-------|------|-------------|
| variable_name | str | Decision variable name |
| values_per_scenario | dict[str, Any] | Scenario name → recommended value |
| is_robust | bool | True if value is consistent across scenarios |
| spread | float | Range or standard deviation of values across scenarios |

**Robust definition:** For continuous/integer variables: coefficient of variation < 0.1 (or range < 5% of bounds). For categorical/ordinal: same value across all scenarios.

## SHAP Computation Details

### Ensemble Aggregation

`shap.TreeExplainer` accepts a single model. For an ensemble of K models:

1. Create one `TreeExplainer` per member model
2. Compute SHAP values per member: `explanation_k = explainer_k(X)`
3. Weighted average: `shap_values = Σ(weight_k × explanation_k.values)` for k=1..K
4. Base value: `base_value = Σ(weight_k × explanation_k.base_values[0])` for k=1..K

### Categorical Feature Handling

The surrogate pipeline pre-encodes categoricals before training (no `enable_categorical=True` in XGBoost). TreeExplainer works on the encoded features. The SHAP results use `feature_names` from `Ensemble.feature_names`, which are the post-encoding column names.

## PDP/ICE Computation Details

### Method

Must use `method="brute"` for XGBoost and LightGBM. The `"recursion"` method only works with sklearn's own gradient boosting.

### Ensemble Aggregation

Like SHAP: compute per member, weighted average of PDP values, concatenate ICE values.

### Categorical Variables

For categorical/ordinal decision variables: the grid consists of the declared category values (from `Variable.bounds.categories`). No interpolation.

## What This Layer Does NOT Do

- No visualizations (plots, charts, dashboards)
- No text interpretation or natural language explanations
- No interactive exploration (that is kint's responsibility)
- No model retraining or hyperparameter adjustment
- No additional optimization runs
- No data export to files

## Error Handling

- `AnalysisError` for invalid method arguments (e.g., unknown column, single-objective trade-off)
- Detail analysis failures do not affect the Summary or other detail analyses
- If a SHAP computation fails for a specific ensemble member, skip that member and reweight remaining members

## Required Changes to Other Layers

### SurrogateManager API Extension

Add `get_surrogate_result(column: str) -> SurrogateResult` to expose the full `SurrogateResult` (Ensemble + ConformalCalibration + TrialHistory). The object already exists internally in `_surrogates` — this just makes it accessible. Needed for `SurrogateQuality` computation (conformal coverage).

## File Structure

```
src/surrox/analysis/
    __init__.py              # Public API re-exports
    config.py                # AnalysisConfig
    result.py                # AnalysisResult (frozen, Summary only)
    analyzer.py              # Analyzer (stateful, lazy/cached detail analyses)
    summary.py               # Summary, SolutionSummary, BaselineComparison, etc.
    shap.py                  # ShapGlobalResult, ShapLocalResult, FeatureImportanceResult
    pdp.py                   # PDPICEResult
    trade_off.py             # TradeOffResult
    what_if.py               # WhatIfResult, WhatIfPrediction
    scenario.py              # compare_scenarios(), ScenarioComparisonResult, VariableRobustness
    monotonicity.py          # MonotonicityViolation check
    types.py                 # ConstraintStatusKind

tests/analysis/
    __init__.py
    conftest.py
    test_config.py
    test_summary.py
    test_analyzer.py
    test_shap.py
    test_pdp.py
    test_trade_off.py
    test_what_if.py
    test_scenario.py
    test_monotonicity.py
```

## Consumers

| Consumer | What it reads |
|----------|---------------|
| kint API | `AnalysisResult.summary` for dashboard, `Analyzer` methods on demand |
| kint LLM | All data structures for interpretation and natural language explanation |

## Acceptance Criteria

1. `AnalysisResult` is a frozen Pydantic model containing only the Summary
2. `Analyzer` computes detail analyses lazily on first access and caches them (except `what_if`)
3. `Analyzer.feature_importance()` delegates to `shap_global()` internally
4. `compare_scenarios()` is a separate entry point taking `dict[str, OptimizationResult]`
5. `ConstraintStatus` wraps `ConstraintEvaluation` (no field duplication)
6. `SurrogateQuality` uses `cv_rmse` from trial history and conformal coverage from `SurrogateResult`
7. SHAP values aggregate correctly across ensemble members (weighted average)
8. PDP/ICE uses `method="brute"` for XGBoost/LightGBM compatibility
9. Monotonicity validation detects violations along a 1D slice at the recommended solution
10. What-If supports arbitrary variable values within bounds and is not cached
11. Scenario comparison identifies robust vs. scenario-dependent variables
12. All return types are typed data structures, not visualizations
13. Invalid inputs raise `AnalysisError`, not silent fallbacks
14. `ConstraintStatusKind` is a StrEnum, not a magic string
15. Margin active-threshold handles `limit = 0` via `max(abs(limit), 1e-8)`
