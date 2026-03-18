from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from surrox.exceptions import ConfigurationError
from surrox.surrogate.families import (
    GaussianProcessFamily,
    LightGBMFamily,
    TabICLFamily,
    XGBoostFamily,
)
from surrox.surrogate.models import EnsembleMemberConfig
from surrox.surrogate.protocol import EstimatorFamily


def _default_families() -> tuple[EstimatorFamily, ...]:
    return (XGBoostFamily(), LightGBMFamily(), GaussianProcessFamily(), TabICLFamily())  # pyright: ignore[reportReturnType]


class FeatureReductionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    importance_threshold: float = 0.01
    correlation_threshold: float = 0.9

    @model_validator(mode="after")
    def _validate(self) -> FeatureReductionConfig:
        if not (0 < self.importance_threshold < 1):
            raise ConfigurationError(
                "importance_threshold must be between 0 and 1 exclusive"
            )
        if not (0 < self.correlation_threshold <= 1):
            raise ConfigurationError(
                "correlation_threshold must be between 0 and 1 inclusive of upper bound"
            )
        return self


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    n_trials: int = 50
    cv_folds: int = 5
    calibration_fraction: float = 0.2
    ensemble_size: int = 5
    default_coverage: float = 0.9
    estimator_families: tuple[EstimatorFamily, ...] = _default_families()
    n_threads: int | None = None
    study_timeout_s: int = 300
    min_r2: float | None = 0.7
    min_samples_per_fold: int = 50
    min_calibration_samples: int = 100
    random_seed: int = 42
    feature_reduction: FeatureReductionConfig = FeatureReductionConfig()
    refit_ensemble: dict[str, tuple[EnsembleMemberConfig, ...]] | None = None

    @model_validator(mode="after")
    def _validate_config(self) -> TrainingConfig:
        if self.refit_ensemble is not None:
            for column, members in self.refit_ensemble.items():
                if not members:
                    raise ConfigurationError(
                        f"refit_ensemble['{column}'] must not be empty"
                    )
            return self

        if self.n_trials < 1:
            raise ConfigurationError("n_trials must be >= 1")
        if self.cv_folds < 2:
            raise ConfigurationError("cv_folds must be >= 2")
        if not (0 < self.calibration_fraction < 1):
            raise ConfigurationError(
                "calibration_fraction must be between 0 and 1 exclusive"
            )
        if self.ensemble_size < 1:
            raise ConfigurationError("ensemble_size must be >= 1")
        if not (0 < self.default_coverage < 1):
            raise ConfigurationError(
                "default_coverage must be between 0 and 1 exclusive"
            )
        if len(self.estimator_families) == 0:
            raise ConfigurationError("estimator_families must not be empty")
        family_names = [f.name for f in self.estimator_families]
        if len(family_names) != len(set(family_names)):
            raise ConfigurationError("estimator family names must be unique")
        if self.n_threads is not None and self.n_threads < 1:
            raise ConfigurationError("n_threads must be >= 1 if set")
        if self.study_timeout_s < 1:
            raise ConfigurationError("study_timeout_s must be >= 1")
        if self.min_r2 is not None and not (0 < self.min_r2 < 1):
            raise ConfigurationError("min_r2 must be between 0 and 1 exclusive if set")
        if self.min_samples_per_fold < 1:
            raise ConfigurationError("min_samples_per_fold must be >= 1")
        if self.min_calibration_samples < 1:
            raise ConfigurationError("min_calibration_samples must be >= 1")
        return self
