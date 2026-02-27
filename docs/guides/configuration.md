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
| `diversity_threshold` | 0.95 | Max correlation between ensemble members |
| `calibration_fraction` | 0.2 | Data fraction held out for conformal calibration |
| `default_coverage` | 0.9 | Conformal prediction interval coverage |
| `study_timeout_s` | 300 | Optuna study timeout in seconds |
| `min_r2` | 0.7 | Minimum R² quality threshold (None to disable) |
| `random_seed` | 42 | Random seed |

See [`TrainingConfig`][surrox.TrainingConfig] for the full API.

## OptimizerConfig

Controls the evolutionary optimization.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 100 | Population size for the evolutionary algorithm |
| `n_generations` | 200 | Number of generations |
| `extrapolation_k` | 5 | k-NN neighbors for extrapolation detection |
| `extrapolation_threshold` | 2.0 | Distance threshold for extrapolation flag |
| `constraint_confidence` | 0.95 | Conformal confidence for constraint evaluation |
| `seed` | 42 | Random seed |

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
