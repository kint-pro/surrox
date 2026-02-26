# SurrogateManager — Research

## Optuna HPO

### Study API

```python
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
)
study.optimize(objective, n_trials=100, n_jobs=1)
```

- `n_jobs=1` is required for reproducibility — parallel execution breaks sampler determinism.
- `TPESampler(multivariate=True)` models parameter correlations — outperforms independent TPE for tree-based models.
- `n_startup_trials=20` recommended (random exploration before TPE kicks in).

### Trial API

```python
trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
trial.suggest_int("max_depth", 3, 12)
trial.suggest_int("num_leaves", 16, 512, log=True)
trial.suggest_categorical("booster", ["gbtree", "dart"])

trial.set_user_attr("fold_scores", [0.91, 0.89, 0.92])
trial.set_user_attr("training_time_s", elapsed)
trial.set_user_attr("inference_ms", inference_time)
```

`set_user_attr` stores arbitrary JSON-serializable metadata per trial — accessible later via `trial.user_attrs`.

### Pruning with Cross-Validation

Report cumulative mean after each fold, prune between folds:

```python
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
    model = lgb.LGBMRegressor(**params)
    model.fit(X[train_idx], y[train_idx])
    fold_rmse = root_mean_squared_error(y[val_idx], model.predict(X[val_idx]))
    fold_scores.append(fold_rmse)

    trial.report(np.mean(fold_scores), step=fold_idx)
    if trial.should_prune():
        raise optuna.TrialPruned()
```

`MedianPruner` prunes trials whose intermediate value is worse than the median of completed trials at the same step.

### Top-K Trials

```python
completed = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
top_k = sorted(completed, key=lambda t: t.value)[:K]

for t in top_k:
    print(t.number, t.value, t.params, t.user_attrs)
```

`FrozenTrial` fields: `number`, `value`, `params`, `user_attrs`, `datetime_start`, `datetime_complete`, `duration`, `state`.

### XGBoost Search Space

```python
params = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=50),
    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
    "max_depth": trial.suggest_int("max_depth", 3, 12),
    "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
    "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
    "random_state": 42,
}
```

### LightGBM Search Space

```python
params = {
    "objective": "regression",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=50),
    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
    "num_leaves": trial.suggest_int("num_leaves", 16, 512),
    "max_depth": trial.suggest_int("max_depth", 3, 12),
    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
    "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
    "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
    "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    "min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 1.0, log=True),
    "random_state": 42,
}
```

---

## Monotonicity Constraints

### XGBoost

Dict form (preferred, since v1.5+) — only constrained features need to appear:

```python
XGBRegressor(monotone_constraints={"credit_score": -1, "income": 1})
```

Values: `1` = increasing, `-1` = decreasing, `0` = unconstrained. Omitted features default to `0`. Resolves against feature names from pandas DataFrame columns.

### LightGBM

Positional list — full length required, one entry per feature in column order:

```python
LGBMRegressor(
    monotone_constraints=[1, 0, -1, 0],
    monotone_constraints_method="intermediate",
)
```

No dict/feature-name form. All features must be represented.

`monotone_constraints_method`:
- `"basic"` — fastest, may over-constrain (LightGBM default)
- `"intermediate"` — recommended, less over-constraining
- `"advanced"` — slowest, most permissive enforcement

### Categorical features

- **XGBoost**: technically allowed but semantically undefined for nominal categoricals. Avoid.
- **LightGBM**: hard restriction — constraint must be `0` for categorical features. Raises error otherwise.

### Mapping helper pattern

```python
def xgb_monotone_constraints(
    feature_names: list[str],
    relations: dict[str, str],
) -> dict[str, int]:
    direction_map = {"increasing": 1, "decreasing": -1}
    return {
        name: direction_map[direction]
        for name, direction in relations.items()
        if name in feature_names
    }

def lgbm_monotone_constraints(
    feature_names: list[str],
    categorical_features: set[str],
    relations: dict[str, str],
) -> list[int]:
    direction_map = {"increasing": 1, "decreasing": -1}
    return [
        0 if name in categorical_features
        else direction_map.get(relations.get(name, ""), 0)
        for name in feature_names
    ]
```

### Key rules

- Constraints are fixed domain knowledge, not a tunable hyperparameter — set before the Optuna study.
- `monotone_constraints_method` in LightGBM can optionally be tuned via Optuna as a categorical.
- No interaction issues between Optuna and monotonicity constraints.
- Training time impact: moderate for XGBoost (hist method required), minimal-to-moderate for LightGBM depending on method.

---

## Conformal Prediction

### Split Conformal — Algorithm

1. **Split data**: `D_train` (model training) + `D_calib` (calibration, never seen by model)
2. **Train model** on `D_train`, freeze it
3. **Compute calibration scores**: `s_i = |y_i - f(x_i)|` for each `(x_i, y_i)` in `D_calib`
4. **Compute quantile threshold**: `q_hat = quantile(scores, ceil((n+1)*(1-α))/n, method='higher')`
5. **Predict intervals**: `[f(x) - q_hat, f(x) + q_hat]`

### Implementation (from scratch, ~10 lines)

```python
class SplitConformalRegressor:
    def __init__(self, coverage: float = 0.9):
        self._coverage = coverage
        self._q_hat: float | None = None

    def calibrate(self, y_true: NDArray, y_pred: NDArray) -> None:
        n = len(y_true)
        scores = np.abs(y_true - y_pred)
        level = min(np.ceil((n + 1) * self._coverage) / n, 1.0)
        self._q_hat = float(np.quantile(scores, level, method='higher'))

    def predict_interval(self, y_pred: NDArray) -> tuple[NDArray, NDArray]:
        return y_pred - self._q_hat, y_pred + self._q_hat
```

### Coverage guarantee

- **Marginal**: `P(y_new ∈ interval) >= 1 - α` — holds on average, not conditionally per x.
- **Finite-sample**: valid for any calibration set size, no asymptotic approximation.
- **Assumption**: exchangeability (weaker than i.i.d., fails for time series or distribution shift).
- **Slightly conservative**: with n=100 and α=0.1, actual level is 0.91.

### Ensemble + Conformal

Apply conformal to the **ensemble's aggregate predictions**, not individual models. The ensemble is the deployed prediction function — conformal wraps any fixed function f.

### Calibration set sizing

| n_calib | Coverage precision | Comment |
|---------|-------------------|---------|
| 50      | ±2%               | Borderline |
| 100     | ±1%               | Acceptable minimum |
| 200     | ±0.7%             | Good |
| 500+    | ±0.45%            | Tight |

Practical rule: 10-15% of total data for calibration, minimum 100 points. The calibration split must happen **after** HPO — calibration data must never influence model selection.

### Multiple targets

Each surrogate gets its own independent `SplitConformalRegressor`. Intervals are statistically independent. Per-objective marginal coverage is the correct guarantee for surrox — joint coverage would require multivariate conformal methods.

### Library decision

No library needed. MAPIE's sklearn-estimator interface does not fit cleanly over a custom weighted ensemble. The math is 10 lines.

---

## Ensemble Strategy

### Weighting: Softmax of negative CV-RMSE

```python
scores = [-rmse_1, -rmse_2, ..., -rmse_K]
weights = softmax(scores * temperature)
```

- Numerically stable (no division by near-zero RMSE).
- Temperature = 1.0 as default, configurable.
- Avoid inverse (1 - R²) — R² is unbounded below zero, breaks normalization.

Fallback to uniform weights when score coefficient of variation < 0.1 (all models are similar quality).

### Optimal K

- **K=5 default** — captures almost all ensemble benefit for gradient boosting.
- Diminishing returns after top 3-5 due to internal boosting already averaging many weak learners.
- K=3 for latency-sensitive deployments. Never K>10 without measuring marginal improvement.

### Diversity

Greedy diversity pruning:
1. Sort candidates by RMSE (best first)
2. For each candidate, add only if correlation with every already-selected model < τ (default τ=0.95)
3. Stop at K models

Cross-family (XGBoost + LightGBM) is the strongest diversity source — structurally different algorithms (depth-wise vs. leaf-wise growth).

### Ensemble disagreement as uncertainty

```python
predictions = np.stack([model.predict(X) for model in models], axis=1)
ensemble_mean = predictions @ weights
ensemble_std = predictions.std(axis=1)
```

Ensemble variance measures epistemic uncertainty (model disagreement). Conformal prediction measures total uncertainty (including aleatoric). Use both: ensemble variance as cheap real-time signal, conformal for statistically rigorous intervals.

### Batch prediction

Vectorized in one pass:

```python
predictions = np.stack([model.predict(X) for model in models], axis=1)  # (n_samples, K)
ensemble_prediction = predictions @ weights  # (n_samples,)
ensemble_std = predictions.std(axis=1)  # (n_samples,)
```

### Memory

Single XGBoost/LightGBM model (500 trees, 64 leaves): ~10-50 MB. K=5 → 50-250 MB. Acceptable for production.

### Stacking

Deferred to Phase 2. Risks meta-learner overfitting on small datasets (<1000 samples). Weighted averaging is sufficient for Phase 1 and the `predict(X) → (mean, std)` interface is stacking-compatible for later swap.

---

## Design Implications for surrox

### Data flow

```
BoundDataset
  → 80% train_pool / 20% calibration_set (per surrogate)
  → train_pool used for Optuna study (k-fold CV internally)
  → Top-K models from study → Ensemble (weighted average)
  → Ensemble predictions on calibration_set → SplitConformalRegressor
```

### One surrogate per unique column

`ProblemDefinition.surrogate_columns` returns deduplicated columns. The SurrogateManager trains one ensemble per column. Objectives and DataConstraints that share a column share the same ensemble (ADR-007).

### Monotonicity constraint flow

```
ProblemDefinition.monotonic_constraints_for("objective_name")
  → {"temperature": MonotonicDirection.INCREASING, ...}
  → mapped to XGBoost dict or LightGBM positional list per estimator family
  → injected as fixed param into every Optuna trial
```

Categorical features are excluded from monotonicity constraints (LightGBM hard requirement, XGBoost semantically undefined).

### What the SurrogateManager exposes

- `evaluate(X) → dict[str, NDArray]` — point predictions per column
- `evaluate_with_uncertainty(X, confidence) → dict[str, (pred, lower, upper)]`
- Access to individual ensembles (for Analysis layer: SHAP, PDP/ICE)
- Access to trial histories (for Leaderboard)
- The conformal calibration data and scores

### Key defaults

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| K (ensemble size) | 5 | Captures most benefit, manageable memory |
| Temperature | 1.0 | Balanced weight differentiation |
| Diversity threshold τ | 0.95 | Prevents degenerate near-identical models |
| CV folds | 5 | Standard, balances bias-variance |
| Calibration split | 20% | ≥100 points needed for reliable intervals |
| Conformal coverage | 0.9 | Industry standard, configurable |
| Estimator families | XGBoost + LightGBM | Cross-family diversity |
