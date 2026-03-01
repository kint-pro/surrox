from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from surrox.exceptions import ConfigurationError
from surrox.surrogate.families import GaussianProcessFamily, LightGBMFamily, XGBoostFamily
from surrox.surrogate.protocol import EstimatorFamily


def _default_families() -> tuple[EstimatorFamily, ...]:
    return (XGBoostFamily(), LightGBMFamily(), GaussianProcessFamily())  # pyright: ignore[reportReturnType]


class TrainingConfig(BaseModel):
    """Configuration for surrogate model training.

    Attributes:
        n_trials: Number of Optuna HPO trials per surrogate.
        cv_folds: Number of cross-validation folds.
        calibration_fraction: Fraction of data held out for conformal calibration.
        ensemble_size: Maximum number of models in the ensemble.
        diversity_threshold: Maximum correlation allowed between ensemble members.
        softmax_temperature: Temperature for softmax ensemble weight selection.
        default_coverage: Default conformal prediction interval coverage (0–1).
        estimator_families: Estimator families to search over (XGBoost, LightGBM).
        n_threads: Thread limit per model. None uses all available cores.
        study_timeout_s: Optuna study timeout in seconds.
        min_r2: Minimum R² threshold for model quality. None disables the check.
        random_seed: Random seed for reproducibility.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    n_trials: int = 50
    cv_folds: int = 5
    calibration_fraction: float = 0.2
    ensemble_size: int = 5
    diversity_threshold: float = 0.95
    softmax_temperature: float = 1.0
    default_coverage: float = 0.9
    estimator_families: tuple[EstimatorFamily, ...] = _default_families()
    n_threads: int | None = None
    study_timeout_s: int = 300
    min_r2: float | None = 0.7
    min_samples_per_fold: int = 50
    min_calibration_samples: int = 100
    random_seed: int = 42

    @model_validator(mode="after")
    def _validate_config(self) -> TrainingConfig:
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
        if not (0 < self.diversity_threshold <= 1):
            raise ConfigurationError(
                "diversity_threshold must be between 0 (exclusive) and 1 (inclusive)"
            )
        if self.softmax_temperature <= 0:
            raise ConfigurationError("softmax_temperature must be > 0")
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
