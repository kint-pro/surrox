# ADR-022: Native Model Persistence per EstimatorFamily

## Status

Accepted

## Date

2026-02-27

## Context

Surrox needs to persist trained surrogate models for re-analysis without re-training. The models are sklearn-compatible estimators from XGBoost, LightGBM, and potentially future frameworks (TabPFN).

Enterprise multi-tenant deployments require secure persistence — no arbitrary code execution on load.

### Options

**Option A — Binary serialization (joblib etc.):** Fast, universal. But executes arbitrary code on load. Unacceptable for multi-tenant environments where model files may cross trust boundaries.

**Option B — skops.io:** Scikit-learn's recommended secure alternative. Supports sklearn, XGBoost, LightGBM via granular type trust. But: each new framework (TabPFN, etc.) may not be supported, and it adds a dependency for something each framework already solves natively.

**Option C — Native format per EstimatorFamily:** Each framework uses its own secure, non-executable format. XGBoost saves to UBJSON (`.ubj`), LightGBM saves to its native text format (`.lgbm`). Each `EstimatorFamily` implements `save_model()`/`load_model()`. New frameworks implement the same two methods using their own format.

## Decision

Option C. Persistence responsibility lives in the `EstimatorFamily` protocol (Strategy pattern). Each family knows its framework's safest native format.

```python
class EstimatorFamily(Protocol):
    def save_model(self, model: BaseEstimator, path: Path) -> None: ...
    def load_model(self, path: Path) -> BaseEstimator: ...
```

Conformal calibration data (numpy arrays) is saved as `.npz` and the `SplitConformalRegressor` is re-fit on load (cheap, ~ms).

## Rationale

- Zero arbitrary code execution risk. Native formats (UBJSON, text) contain only model parameters, no executable code.
- No additional dependencies. XGBoost and LightGBM already provide save/load APIs.
- Extensible. Adding TabPFN (`.tabpfn_fit`) or any future framework requires only implementing two methods — the persistence infrastructure remains unchanged.
- Each framework's native format is its most stable, backward-compatible format. XGBoost explicitly recommends UBJSON over binary serialization for long-term storage.

## Consequences

- The surrogates directory contains mixed file formats (`.ubj`, `.lgbm`, `.tabpfn_fit`). This is an internal detail hidden behind `SurrogateManager.save()`/`load()`.
- `metadata.json` stores the family name per ensemble member, so `load()` knows which family to delegate to.
- A `_FAMILY_REGISTRY` maps family names to classes for reconstruction on load.
- Conformal re-fitting on load adds ~ms of overhead but avoids persisting the complex `SplitConformalRegressor` object.
