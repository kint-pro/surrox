# Surrogate Layer — Specification

## Purpose

The Surrogate layer trains, evaluates, and manages surrogate models for all targets defined in a ProblemDefinition. It consumes the Problem layer (ProblemDefinition + BoundDataset) and produces trained ensembles with calibrated uncertainty estimates. It decides HOW to model — never WHAT to model.

## Inputs

| Input | Source | Description |
|-------|--------|-------------|
| ProblemDefinition | Problem layer | Declares surrogate_columns, variables (features), monotonic_constraints_for() |
| BoundDataset | Problem layer | Validated training data |

## Protocols

### EstimatorFamily

A Protocol defining the interface for an estimator algorithm family. Each implementation encapsulates its search space, model creation, and constraint mapping. Adding a new estimator family (e.g., CatBoost, TabPFN) means implementing this protocol — no changes to the training pipeline (ADR-012).

| Method | Signature | Description |
|--------|-----------|-------------|
| name | `-> str` | Unique identifier (e.g., `"xgboost"`, `"lightgbm"`) |
| suggest_hyperparameters | `(trial: optuna.Trial) -> dict[str, Any]` | Define the Optuna search space for this family. Called once per trial. |
| build_model | `(hyperparameters: dict[str, Any], monotonic_constraints: Any, random_seed: int, n_threads: int \| None) -> BaseEstimator` | Create a configured, unfitted sklearn-compatible model. `monotonic_constraints` is the native format returned by `map_monotonic_constraints`. `n_threads` limits internal parallelism (XGBoost `nthread`, LightGBM `num_threads`). |
| map_monotonic_constraints | `(constraints: dict[str, MonotonicDirection], feature_names: list[str], categorical_features: set[str]) -> Any` | Map from surrox's MonotonicDirection representation to this estimator's native format. Returns the native format (XGBoost: `dict[str, int]`, LightGBM: `list[int]`) that `build_model` consumes directly. |

**Built-in implementations:** `XGBoostFamily` and `LightGBMFamily`.

**Registration:** `TrainingConfig.estimator_families` holds the list of families to explore. Defaults to `(XGBoostFamily(), LightGBMFamily())`.

## Domain Entities

### TrainingConfig

Configuration for the surrogate training process. Immutable after construction.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| n_trials | int | 50 | Total Optuna trials per surrogate |
| cv_folds | int | 5 | Number of cross-validation folds |
| calibration_fraction | float | 0.2 | Fraction of data reserved for conformal calibration |
| ensemble_size | int | 5 | Maximum number of models in ensemble (K) |
| diversity_threshold | float | 0.95 | Max prediction correlation between ensemble members |
| softmax_temperature | float | 1.0 | Temperature for softmax weight computation |
| default_coverage | float | 0.9 | Default conformal prediction coverage level |
| estimator_families | tuple[EstimatorFamily, ...] | (XGBoostFamily(), LightGBMFamily()) | Estimator families to explore in the Optuna study |
| n_threads | int \| None | None | Max threads per model for XGBoost/LightGBM. None = OS default (all cores). Set explicitly when multiple trainings run concurrently to prevent thread contention. Passed to `EstimatorFamily.build_model()`. |
| study_timeout_s | int | 300 | Timeout for the entire Optuna study in seconds. Mandatory — a library that runs unbounded is a resource management bug. |
| min_r2 | float \| None | 0.7 | Minimum R² on the calibration set for the final ensemble. If the ensemble scores below this threshold, training fails — the surrogate quality is insufficient for responsible recommendations. None disables the gate (not recommended for safety-critical applications). |
| random_seed | int | 42 | Seed for reproducibility |

**Validation rules:**
- n_trials >= 1
- cv_folds >= 2
- 0 < calibration_fraction < 1
- ensemble_size >= 1
- 0 < diversity_threshold <= 1
- softmax_temperature > 0
- 0 < default_coverage < 1
- estimator_families must not be empty
- All estimator family names must be unique
- n_threads >= 1 if not None
- study_timeout_s >= 1
- 0 < min_r2 < 1 if not None

### TrialRecord

Immutable record of a single Optuna trial. One per trial per surrogate.

| Field | Type | Description |
|-------|------|-------------|
| trial_number | int | Optuna trial ID |
| estimator_family | str | Name of the EstimatorFamily (e.g., `"xgboost"`, `"lightgbm"`) |
| hyperparameters | dict[str, Any] | Full hyperparameter config |
| fold_metrics | tuple[FoldMetrics, ...] | Per-fold evaluation results |
| mean_r2 | float | Mean R² across folds |
| mean_rmse | float | Mean RMSE across folds |
| mean_mae | float | Mean MAE across folds |
| mean_training_time_s | float | Mean training time per fold in seconds |
| mean_inference_time_ms | float | Mean inference time per fold in milliseconds |
| status | str | `"completed"` or `"pruned"` |

### FoldMetrics

Per-fold evaluation results.

| Field | Type | Description |
|-------|------|-------------|
| fold | int | Fold index (0-based) |
| r2 | float | R² score |
| rmse | float | Root mean squared error |
| mae | float | Mean absolute error |
| training_time_s | float | Training time in seconds |
| inference_time_ms | float | Inference time in milliseconds |

### EnsembleMember

A single model within an ensemble.

| Field | Type | Description |
|-------|------|-------------|
| trial_number | int | Reference to the TrialRecord |
| estimator_family | str | Name of the EstimatorFamily |
| model | BaseEstimator | The fitted sklearn-compatible estimator (any family) |
| weight | float | Ensemble weight (sum of all weights = 1.0) |
| cv_rmse | float | Cross-validation RMSE used for weight computation |

### Ensemble

A weighted collection of models for a single surrogate column. Provides point predictions and ensemble disagreement. The `predict` and `predict_with_std` methods define the interface consumed by ConformalCalibration, SurrogateManager, and the Analysis layer. A future stacking implementation would provide the same interface with a different internal strategy — no consumer changes needed.

| Field | Type | Description |
|-------|------|-------------|
| column | str | The target column this ensemble predicts |
| members | tuple[EnsembleMember, ...] | The K models with weights |
| feature_names | tuple[str, ...] | Ordered feature names used during training |
| monotonic_constraints | dict[str, MonotonicDirection] | Applied monotonicity constraints |

**Methods:**

- `predict(X: DataFrame) -> NDArray` — Weighted average of member predictions. Returns shape `(n_samples,)`.
- `predict_with_std(X: DataFrame) -> tuple[NDArray, NDArray]` — Returns `(mean, std)`. std is the ensemble disagreement (standard deviation of individual predictions, not weighted).

**Prediction logic:**

```
predictions = stack([member.model.predict(X) for member in members])  # (K, n_samples)
weights = array([member.weight for member in members])                # (K,)
mean = predictions.T @ weights                                        # (n_samples,)
std = predictions.std(axis=0)                                         # (n_samples,)
```

### ConformalCalibration

Calibrated prediction intervals for a single surrogate column, using MAPIE's `SplitConformalRegressor` (ADR-010).

| Field | Type | Description |
|-------|------|-------------|
| column | str | The target column this calibration applies to |
| adapter | EnsembleAdapter | The sklearn-compatible ensemble wrapper |
| X_calib | NDArray | Calibration features, shape (n_calib, n_features) |
| y_calib | NDArray | Calibration targets, shape (n_calib,) |

**Methods:**

- `prediction_interval(X: DataFrame, coverage: float) -> tuple[NDArray, NDArray, NDArray]` — Returns `(mean, lower, upper)`. Creates a `SplitConformalRegressor(estimator=adapter, confidence_level=coverage, prefit=True)`, conformalizes with stored `(X_calib, y_calib)`, and calls `predict_interval`. The `y_intervals` output has shape `(n, 2, 1)` — extract as `lower = y_intervals[:, 0, 0]`, `upper = y_intervals[:, 1, 0]`. The `conformalize` call is O(n_calib) — negligible.

MAPIE handles internally: finite-sample correction `ceil((n+1)*(1-α))/n`, conservative quantile method (`'higher'`), tied residual handling, and minimum sample validation.

**Validation rules:**
- 0 < coverage < 1

### EnsembleAdapter

A sklearn-compatible wrapper around the custom Ensemble, required by MAPIE's `prefit=True` interface.

| Field | Type | Description |
|-------|------|-------------|
| ensemble | Ensemble | The underlying weighted ensemble |

Inherits from `sklearn.base.RegressorMixin, BaseEstimator`. Implements `predict(X)` delegating to `ensemble.predict(X)`, and `__sklearn_is_fitted__()` returning `True`.

### SurrogateResult

The complete output of surrogate training for a single column.

| Field | Type | Description |
|-------|------|-------------|
| column | str | Target column |
| ensemble | Ensemble | The trained ensemble |
| conformal | ConformalCalibration | Calibrated uncertainty |
| trial_history | tuple[TrialRecord, ...] | All trials (completed + pruned) |

### SurrogateManager

The top-level container. Holds all trained surrogates and provides the evaluation interface consumed by the Optimizer and Analysis layers.

| Field | Type | Description |
|-------|------|-------------|
| problem | ProblemDefinition | The problem this manager was trained for |
| config | TrainingConfig | Configuration used for training |
| surrogates | dict[str, SurrogateResult] | Column name → trained surrogate |

**Construction:** via a `train()` class method or factory function, not direct instantiation. Training is the only way to create a SurrogateManager.

```python
manager = SurrogateManager.train(problem, dataset, config)
```

**Public methods:**

- `evaluate(X: DataFrame) -> dict[str, NDArray]` — Point predictions for all surrogate columns. X must contain all variable columns (decision + context). Returns `{column_name: predictions}`.

- `evaluate_with_uncertainty(X: DataFrame, coverage: float = 0.9) -> dict[str, SurrogatePrediction]` — Predictions with confidence intervals. Returns `{column_name: SurrogatePrediction}`.

- `get_ensemble(column: str) -> Ensemble` — Access individual ensemble for analysis (SHAP, PDP/ICE). Raises KeyError if column not found.

- `get_trial_history(column: str) -> tuple[TrialRecord, ...]` — Access trial history for leaderboard. Raises KeyError if column not found.

### SurrogatePrediction

Prediction result for a single column, including uncertainty.

| Field | Type | Description |
|-------|------|-------------|
| mean | NDArray | Point predictions, shape (n_samples,) |
| std | NDArray | Ensemble disagreement, shape (n_samples,) |
| lower | NDArray | Lower bound of conformal interval, shape (n_samples,) |
| upper | NDArray | Upper bound of conformal interval, shape (n_samples,) |

## Training Pipeline

The training pipeline runs once per unique surrogate column. The columns are derived from `ProblemDefinition.surrogate_columns`.

### Step 1: Data Split

```
BoundDataset.dataframe
  → feature_columns: all variable names (decision + context)
  → target_column: the surrogate column being trained
  → Split: (1 - calibration_fraction) for train_pool, calibration_fraction for calibration_set
  → Split is stratified by nothing (regression) — random, seeded
```

### Step 2: Optuna Study

One study per surrogate column. The study searches across all configured estimator families in a single study (ADR-008).

**Estimator family selection per trial:**

```python
family_names = [f.name for f in config.estimator_families]
family_name = trial.suggest_categorical("estimator_family", family_names)
family = families_by_name[family_name]
```

**Hyperparameter suggestion per trial:**

```python
hyperparameters = family.suggest_hyperparameters(trial)
```

Each family defines its own search space via the `EstimatorFamily` protocol. Optuna explores all families and their respective spaces jointly.

**Monotonicity constraints:**

```python
raw_constraints = problem.monotonic_constraints_for(column)
mapped_constraints = family.map_monotonic_constraints(raw_constraints, feature_names, categorical_features)
model = family.build_model(hyperparameters, mapped_constraints, config.random_seed, config.n_threads)
```

Each family maps surrox's `MonotonicDirection` representation to its native format (e.g., XGBoost dict, LightGBM positional list). Only continuous and integer variables can have monotonicity constraints — enforced at the ProblemDefinition level (ADR-011). Constraints are fixed across all trials — they are domain knowledge, not tunable.

**Cross-validation per trial:**

k-Fold CV on `train_pool`. Per fold: train model, compute R², RMSE, MAE, training time, inference time. Report cumulative mean RMSE after each fold for pruning.

**Pruning:**

`MedianPruner(n_startup_trials=10, n_warmup_steps=2)` — don't prune during first 10 trials, skip first 2 folds per trial.

**Sampler:**

`TPESampler(seed=random_seed, multivariate=True, n_startup_trials=20)`

**Timeout:**

The study stops after `study_timeout_s` seconds regardless of how many trials have completed. At least one trial must complete — if the timeout is too short for even a single trial, training fails.

### Step 3: Ensemble Construction

After the study completes:

1. Collect all completed trials, sorted by mean RMSE (ascending)
2. **Diversity filter**: Greedy selection — iterate sorted trials, add to ensemble only if Pearson correlation of its OOF predictions with every already-selected member's OOF predictions < `diversity_threshold`.
3. Stop when `ensemble_size` members are selected or no more candidates pass the diversity filter.
4. **Weight computation**: softmax of negative RMSE values with `softmax_temperature`.
5. **Retrain**: Each selected model is rebuilt via `family.build_model(hyperparameters, mapped_constraints, random_seed, n_threads)` and trained on the full `train_pool` (not just a CV fold).

**Softmax weight formula:**

```
scores = [-rmse_1, -rmse_2, ..., -rmse_K]
exp_scores = exp(scores / temperature)
weights = exp_scores / sum(exp_scores)
```

**Known limitation:** Softmax on raw RMSE values is scale-dependent. A 1% relative difference produces different weight distributions depending on the absolute RMSE scale (e.g., RMSE ~0.1 vs. RMSE ~1000). The temperature parameter compensates partially, but may need tuning per problem. A future improvement would be to apply softmax to ranks or range-normalized RMSE values instead of raw values.

### Step 4: Conformal Calibration

The ensemble is wrapped in an `EnsembleAdapter` (sklearn-compatible). The `ConformalCalibration` stores the adapter and calibration data:

```python
adapter = EnsembleAdapter(ensemble)
conformal = ConformalCalibration(
    column=column,
    adapter=adapter,
    X_calib=X_calib,
    y_calib=y_calib,
)
```

At prediction time, `prediction_interval(X, coverage)` creates a fresh `SplitConformalRegressor(estimator=adapter, confidence_level=coverage, prefit=True)`, conformalizes with the stored calibration data, and calls `predict_interval`. This supports arbitrary coverage levels without pre-committing to one. The `conformalize` call is O(n_calib) — negligible.

### Step 5: Assembly

The `SurrogateResult` for this column is assembled from the ensemble, conformal calibration, and trial history. Repeat for all columns in `surrogate_columns`.

## Out-of-Fold Predictions for Diversity

During the Optuna study, each completed trial's k-fold CV produces out-of-fold (OOF) predictions: every sample in `train_pool` receives exactly one prediction — from the fold where it served as a validation point. The OOF vector is the concatenation of all fold validation predictions, reassembled in the original sample order. This gives one array of shape `(len(train_pool),)` per completed trial.

These OOF arrays are held in memory for the duration of the study and used during ensemble construction for pairwise correlation computation (diversity filtering). After ensemble construction they are discarded. Memory cost: one float array per completed trial × `len(train_pool)`. For 50 trials and 10,000 samples this is ~4 MB — negligible.

## Consumers

| Layer | What it reads |
|-------|---------------|
| Optimizer | `evaluate(X)` for objective function evaluation, `evaluate_with_uncertainty(X, coverage)` for confidence filtering |
| Analysis | `get_ensemble(column)` for SHAP/PDP/ICE per surrogate, `get_trial_history(column)` for leaderboard, `evaluate_with_uncertainty()` for constraint confidence |

## Minimum Data Requirements

Training fails immediately if data is insufficient:

- **Calibration set**: at least 100 points. With fewer points, the finite-sample correction pushes `q_level` toward 1.0, making intervals impractically wide (approaching the maximum residual). The threshold of 100 ensures intervals are practically useful (coverage precision ±1%).
- **Train pool per CV fold**: at least 50 validation samples per fold. With `cv_folds=5`, this means `train_pool >= 250`. Fewer samples produce unreliable fold metrics.
- **Total dataset minimum**: derived from the above. With `calibration_fraction=0.2` and `cv_folds=5`: minimum total = `max(250 / (1 - 0.2), 100 / 0.2)` = `max(313, 500)` = **500 rows**. This is validated before training begins.

The formula for the minimum is: `max(ceil(50 * cv_folds / (1 - calibration_fraction)), ceil(100 / calibration_fraction))`.

## Error Handling

- If the dataset has fewer rows than the minimum (see above), training fails immediately.
- If all trials for a surrogate column are pruned (no completed trials), training fails immediately.
- If the diversity filter eliminates all candidates after the first (correlation ≥ threshold with the best model for all others), the ensemble contains only the single best model. This is not an error — it means all models are nearly identical.
- If `min_r2` is set and the final ensemble's R² on the calibration set falls below it, training fails immediately. The error message includes the actual R² and the threshold. This is a safety gate — insufficient surrogate quality means the framework cannot make responsible recommendations.
- Optuna exceptions (e.g., NaN in objective) are caught per trial and recorded as failed trials, not propagated.

## What SurrogateManager Does NOT Do

- No feature engineering or preprocessing
- No feature selection
- No data cleaning or imputation
- No built-in estimator families beyond XGBoost and LightGBM (extensible via EstimatorFamily protocol)
- No stacking or meta-learning (Phase 2)
- No parallelization of training jobs (Phase 2)
- No drift detection or selective retraining (Phase 2)
- No persistent storage of models to disk

## Acceptance Criteria

1. One ensemble is trained per unique surrogate column (not per Objective/DataConstraint)
2. Both XGBoost and LightGBM are explored in the same Optuna study
3. Monotonicity constraints are correctly mapped per estimator family, categorical features excluded
4. Cross-validation metrics (R², RMSE, MAE) are recorded per fold per trial
5. Training and inference times are recorded per fold per trial
6. Pruning reduces total training time without degrading ensemble quality
7. Ensemble members are selected with diversity filtering (correlation threshold)
8. Ensemble weights are computed via softmax of negative RMSE
9. Selected models are retrained on the full train_pool
10. Conformal prediction intervals achieve the requested coverage on held-out data
11. `evaluate(X)` returns predictions for all surrogate columns in a single call
12. `evaluate_with_uncertainty(X, coverage)` returns point predictions + conformal intervals
13. Individual ensembles are accessible for the Analysis layer
14. Trial histories are accessible for leaderboard display
15. Training fails fast on insufficient data or zero completed trials
16. Training is reproducible given the same data, config, and random_seed
