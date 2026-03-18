# Configuration

Each pipeline layer accepts an optional config object. Defaults work well for most cases, but all parameters can be tuned.

## Full Example

```python
result, analyzer = surrox.run(
    problem=problem,
    dataframe=df,
    surrogate_config=surrox.TrainingConfig(
        n_trials=100,
        ensemble_size=7,
        cv_folds=10,
        default_coverage=0.95,
    ),
    optimizer_config=surrox.OptimizerConfig(
        population_size=200,
        n_generations=500,
    ),
    analysis_config=surrox.AnalysisConfig(
        shap_background_size=200,
        pdp_grid_resolution=100,
    ),
)
```

## TrainingConfig

Controls surrogate model training: HPO budget, ensemble construction, and conformal calibration.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_trials` | 50 | Optuna HPO trials per target column |
| `cv_folds` | 5 | Cross-validation folds |
| `ensemble_size` | 5 | Maximum models in the ensemble |
| `calibration_fraction` | 0.2 | Data fraction held out for conformal calibration |
| `default_coverage` | 0.9 | Conformal prediction interval coverage |
| `study_timeout_s` | 300 | Optuna study timeout in seconds |
| `min_r2` | 0.7 | Minimum R² quality threshold (None to disable) |
| `random_seed` | 42 | Random seed |

### FeatureReductionConfig

Controls automatic feature reduction (importance screening + correlation grouping). Nested inside `TrainingConfig`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `True` | Enable automatic feature reduction |
| `importance_threshold` | 0.01 | Minimum relative importance to keep a feature (XGBoost-based screening) |
| `correlation_threshold` | 0.9 | Absolute correlation above which features are grouped via PCA |

Feature reduction is skipped when there are fewer than 10 features or fewer than 100 samples. Features involved in monotonic constraints are never dropped or grouped.

```python
surrox.TrainingConfig(
    feature_reduction=surrox.FeatureReductionConfig(
        enabled=True,
        importance_threshold=0.02,
        correlation_threshold=0.85,
    ),
)
```

See [`TrainingConfig`][surrox.TrainingConfig] for the full API.

## OptimizerConfig

Controls the optimization strategy. The optimizer auto-selects between a global surrogate strategy (pymoo) for low-dimensional problems and a trust region strategy (TuRBO) for high-dimensional problems.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `strategy` | `None` | `GLOBAL_SURROGATE`, `TRUST_REGION`, or `None` (auto-select based on `dim_threshold`) |
| `dim_threshold` | 15 | Decision variable count above which TuRBO is auto-selected |
| `population_size` | 100 | Population size for pymoo (global strategy only) |
| `n_generations` | 200 | Number of generations for pymoo (global strategy only) |
| `extrapolation_k` | 5 | k-NN neighbors for extrapolation detection |
| `extrapolation_threshold` | 2.0 | Distance threshold for extrapolation flag |
| `constraint_confidence` | 0.95 | Conformal confidence for constraint evaluation |
| `seed` | 42 | Random seed |
| `turbo` | `TuRBOConfig()` | TuRBO-specific configuration (trust region strategy only) |

### TuRBOConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_initial` | `None` | Initial Sobol points (`None` = 2 × n_decision_variables) |
| `max_evaluations` | 500 | Total evaluation budget |
| `batch_size` | 1 | Candidates per iteration |
| `length_init` | 0.8 | Initial trust region side length in [0,1]^d |
| `length_min` | 0.0078125 | Minimum TR length before restart |
| `length_max` | 1.6 | Maximum TR length |
| `success_tolerance` | 3 | Consecutive successes before TR expansion |
| `failure_tolerance` | `None` | Consecutive failures before TR shrinkage (`None` = ceil(dim / batch_size)) |
| `n_restarts` | 3 | Maximum TR restarts before termination |

See [`OptimizerConfig`][surrox.OptimizerConfig] for the full API.

## AnalysisConfig

Controls the post-optimization analysis.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `shap_background_size` | 100 | Background samples for SHAP |
| `pdp_grid_resolution` | 50 | Grid points for PDP/ICE |
| `pdp_percentiles` | (0.05, 0.95) | Grid range percentile bounds |
| `monotonicity_check_resolution` | 50 | Grid resolution for monotonicity checks |
| `random_seed` | 42 | Random seed |

See [`AnalysisConfig`][surrox.AnalysisConfig] for the full API.
