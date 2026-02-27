# Persistence

surrox provides two persistence mechanisms: lightweight result serialization and full surrogate model persistence.

## Result Serialization

Save and load optimization + analysis results as JSON:

```python
from pathlib import Path
import surrox

surrox.save_result(result, Path("result.json"))
loaded = surrox.load_result(Path("result.json"))
```

This serializes the [`SurroxResult`][surrox.SurroxResult] (optimization points, summary analysis) but not the trained models. Suitable for archiving results and loading them in a different session.

## Surrogate Model Persistence

Save the full [`SurrogateManager`][surrox.SurrogateManager] including trained models, conformal calibration, and metadata:

```python
surrogate_manager.save(Path("surrogates/"))
```

This creates:

```
surrogates/
├── metadata.json          # problem, config, versions, trial history
├── models/
│   ├── yield_0            # model files per ensemble member
│   ├── yield_1
│   ├── cost_0
│   └── ...
└── conformal/
    ├── yield.npz          # conformity scores per target
    ├── cost.npz
    └── ...
```

Load it back:

```python
loaded_manager = surrox.SurrogateManager.load(Path("surrogates/"))
```

The loaded manager can be used for predictions and analysis without retraining. Version mismatches between the saved and current environment are logged as warnings.

## Metadata

The `metadata.json` includes:

- Full problem definition (for reproducibility)
- Training configuration
- Library versions (surrox, numpy, scikit-learn, xgboost, lightgbm)
- Dataset fingerprint (SHA-256 hash)
- Per-column: ensemble members, trial history, conformal coverage
