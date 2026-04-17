from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from surrox._logging import log_duration
from surrox.exceptions import SurroxError
from surrox.surrogate.conformal import ConformalCalibration
from surrox.surrogate.ensemble import Ensemble
from surrox.surrogate.families import (
    GaussianProcessFamily,
    LightGBMFamily,
    TabICLFamily,
    XGBoostFamily,
)
from surrox.surrogate.models import EnsembleMember, EnsembleMemberConfig, SurrogatePrediction, TrialRecord
from surrox.surrogate.protocol import EstimatorFamily

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

    from surrox.problem.dataset import BoundDataset
    from surrox.problem.definition import ProblemDefinition
    from surrox.problem.types import MonotonicDirection
    from surrox.surrogate.config import TrainingConfig


_logger = logging.getLogger(__name__)

_FAMILY_REGISTRY: MappingProxyType[str, type[EstimatorFamily]] = MappingProxyType({
    "xgboost": XGBoostFamily,  # type: ignore[type-abstract]
    "lightgbm": LightGBMFamily,  # type: ignore[type-abstract]
    "gaussian_process": GaussianProcessFamily,  # type: ignore[type-abstract]
    "tabicl": TabICLFamily,  # type: ignore[type-abstract]
})


class SurrogateResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    column: str
    ensemble: Ensemble
    conformal: ConformalCalibration
    trial_history: tuple[TrialRecord, ...]
    ensemble_r2: float


class SurrogateManager:
    """Manages trained surrogate ensembles for all target columns.

    Provides prediction, uncertainty quantification, and persistence.
    Created via the `train` class method or loaded from disk via `load`.
    """

    def __init__(
        self,
        problem: ProblemDefinition,
        config: TrainingConfig,
        surrogates: dict[str, SurrogateResult],
        dataset_fingerprint: str,
    ) -> None:
        self._problem = problem
        self._config = config
        self._surrogates = surrogates
        self._dataset_fingerprint = dataset_fingerprint

    @classmethod
    def fast_train(
        cls,
        problem: ProblemDefinition,
        dataset: BoundDataset,
        families: tuple[str, ...] = ("xgboost", "lightgbm", "gaussian_process"),
        calibration_fraction: float | None = None,
        random_seed: int = 42,
    ) -> "SurrogateManager":
        """Train surrogates without Optuna — ~20x faster than ``train``.

        Uses fixed default hyperparameters for each family and skips
        cross-validation entirely.  Produces the same ``SurrogateManager``
        interface including conformal calibration and uncertainty estimates.

        Suitable for a first-look quality signal, smoke tests, and benchmarks
        where training time matters more than optimal hyperparameters.

        Args:
            problem: Problem definition.
            dataset: Bound dataset.
            families: Which estimator families to include.  GP is silently
                dropped for n > 200 to avoid O(n³) slowdown.
            calibration_fraction: Fraction held out for conformal calibration.
                Defaults to 0.15 for n < 200, 0.2 otherwise.
            random_seed: Random seed for reproducibility.

        Returns:
            A trained ``SurrogateManager``.
        """
        from surrox.surrogate.pipeline import fast_train_surrogate

        columns = problem.surrogate_columns
        surrogates: dict[str, SurrogateResult] = {}
        for column in columns:
            surrogates[column] = fast_train_surrogate(
                problem=problem,
                dataset_df=dataset.dataframe,
                column=column,
                families=families,
                calibration_fraction=calibration_fraction,
                random_seed=random_seed,
            )

        from surrox.surrogate.config import TrainingConfig
        config = TrainingConfig(min_r2=None, min_samples_per_fold=5, min_calibration_samples=5)

        fingerprint = _compute_dataset_fingerprint(dataset.dataframe)
        return cls(
            problem=problem,
            config=config,
            surrogates=surrogates,
            dataset_fingerprint=fingerprint,
        )

    @classmethod
    def train(
        cls,
        problem: ProblemDefinition,
        dataset: BoundDataset,
        config: TrainingConfig,
    ) -> SurrogateManager:
        from surrox.surrogate.pipeline import refit_surrogate, train_surrogate

        columns = problem.surrogate_columns
        is_refit = config.refit_ensemble is not None
        _logger.info(
            "surrogate training started",
            extra={
                "n_columns": len(columns),
                "columns": list(columns),
                "n_trials": config.n_trials,
                "n_families": len(config.estimator_families),
                "refit": is_refit,
            },
        )

        surrogates: dict[str, SurrogateResult] = {}
        for column in columns:
            with log_duration(
                _logger, "surrogate column training",
                column=column,
                refit=is_refit,
            ):
                if config.refit_ensemble and column in config.refit_ensemble:
                    surrogates[column] = refit_surrogate(
                        problem=problem,
                        dataset_df=dataset.dataframe,
                        config=config,
                        column=column,
                        member_configs=config.refit_ensemble[column],
                    )
                else:
                    surrogates[column] = train_surrogate(
                        problem=problem,
                        dataset_df=dataset.dataframe,
                        config=config,
                        column=column,
                    )
            ensemble = surrogates[column].ensemble
            _logger.info(
                "surrogate column complete",
                extra={
                    "column": column,
                    "ensemble_size": len(ensemble.members),
                },
            )

        _logger.info(
            "surrogate training complete",
            extra={"n_columns": len(columns)},
        )
        fingerprint = _compute_dataset_fingerprint(dataset.dataframe)
        return cls(
            problem=problem,
            config=config,
            surrogates=surrogates,
            dataset_fingerprint=fingerprint,
        )

    @property
    def problem(self) -> ProblemDefinition:
        return self._problem

    @property
    def config(self) -> TrainingConfig:
        return self._config

    def evaluate(self, X: pd.DataFrame) -> dict[str, NDArray]:
        return {
            column: result.ensemble.predict(X)
            for column, result in self._surrogates.items()
        }

    def evaluate_with_uncertainty(
        self, X: pd.DataFrame, coverage: float | None = None
    ) -> dict[str, SurrogatePrediction]:
        if coverage is None:
            coverage = self._config.default_coverage

        results: dict[str, SurrogatePrediction] = {}
        for column, result in self._surrogates.items():
            mean, std = result.ensemble.predict_with_std(X)
            pred_mean, lower, upper = result.conformal.prediction_interval(X, coverage)
            results[column] = SurrogatePrediction(
                mean=mean, std=std, lower=lower, upper=upper
            )
        return results

    def get_ensemble(self, column: str) -> Ensemble:
        return self._surrogates[column].ensemble

    def get_ensemble_r2(self, column: str) -> float:
        return self._surrogates[column].ensemble_r2

    def get_trial_history(self, column: str) -> tuple[TrialRecord, ...]:
        return self._surrogates[column].trial_history

    def get_surrogate_result(self, column: str) -> SurrogateResult:
        return self._surrogates[column]

    def get_ensemble_member_configs(
        self, column: str,
    ) -> tuple[EnsembleMemberConfig, ...]:
        result = self._surrogates[column]
        if result.trial_history:
            trial_by_number = {
                t.trial_number: t for t in result.trial_history
            }
            return tuple(
                EnsembleMemberConfig(
                    estimator_family=member.estimator_family,
                    hyperparameters=trial_by_number[member.trial_number].hyperparameters,
                    weight=member.weight,
                )
                for member in result.ensemble.members
            )
        if self._config.refit_ensemble and column in self._config.refit_ensemble:
            return self._config.refit_ensemble[column]
        raise SurroxError(
            f"cannot extract ensemble config for '{column}': "
            f"no trial history and no refit config"
        )

    def save(self, path: Path) -> None:
        _logger.info(
            "saving surrogates",
            extra={"path": str(path), "n_columns": len(self._surrogates)},
        )
        path.mkdir(parents=True, exist_ok=True)
        models_dir = path / "models"
        models_dir.mkdir(exist_ok=True)
        conformal_dir = path / "conformal"
        conformal_dir.mkdir(exist_ok=True)

        config_dict = json.loads(self._config.model_dump_json(
            exclude={"estimator_families"},
        ))
        config_dict["estimator_family_names"] = [
            f.name for f in self._config.estimator_families
        ]

        metadata: dict[str, Any] = {
            "problem": json.loads(self._problem.model_dump_json()),
            "training_config": config_dict,
            "versions": _collect_versions(),
            "dataset_fingerprint": self._dataset_fingerprint,
            "columns": {},
        }

        for column, surrogate in self._surrogates.items():
            ensemble = surrogate.ensemble
            family_instances = self._resolve_families()

            members_meta: list[dict[str, Any]] = []
            for i, member in enumerate(ensemble.members):
                family = family_instances[member.estimator_family]
                model_path = models_dir / f"{column}_{i}"
                family.save_model(member.model, model_path)
                members_meta.append({
                    "trial_number": member.trial_number,
                    "estimator_family": member.estimator_family,
                    "weight": member.weight,
                    "cv_rmse": member.cv_rmse,
                })

            conformal = surrogate.conformal
            np.savez(
                conformal_dir / f"{column}.npz",
                conformity_scores=conformal.conformity_scores,
            )

            trial_history = [
                json.loads(t.model_dump_json()) for t in surrogate.trial_history
            ]

            metadata["columns"][column] = {
                "feature_names": list(ensemble.feature_names),
                "monotonic_constraints": {
                    k: v.value for k, v in ensemble.monotonic_constraints.items()
                },
                "members": members_meta,
                "trial_history": trial_history,
                "default_coverage": conformal._default_coverage,
                "ensemble_r2": surrogate.ensemble_r2,
                "y_min": ensemble.y_min,
                "y_max": ensemble.y_max,
            }

        (path / "metadata.json").write_text(json.dumps(metadata, indent=2))
        _logger.info("surrogates saved", extra={"path": str(path)})

    @classmethod
    def load(cls, path: Path) -> SurrogateManager:
        from surrox.problem.definition import ProblemDefinition
        from surrox.problem.types import MonotonicDirection
        from surrox.surrogate.config import TrainingConfig

        metadata_path = path / "metadata.json"
        if not metadata_path.exists():
            raise SurroxError(f"metadata.json not found in {path}")

        metadata = json.loads(metadata_path.read_text())
        problem = ProblemDefinition.model_validate(metadata["problem"])

        config_data = metadata.get("training_config", {})
        family_names = config_data.pop(
            "estimator_family_names", list(_FAMILY_REGISTRY.keys()),
        )
        unknown = [n for n in family_names if n not in _FAMILY_REGISTRY]
        if unknown:
            raise SurroxError(
                f"unknown estimator families in saved model: {unknown}"
            )
        families = tuple(
            _FAMILY_REGISTRY[name]() for name in family_names
        )
        config = TrainingConfig(
            estimator_families=families,
            **{k: v for k, v in config_data.items() if k != "estimator_family_names"},
        )

        family_instances: dict[str, EstimatorFamily] = {
            name: cls_() for name, cls_ in _FAMILY_REGISTRY.items()
        }

        models_dir = path / "models"
        conformal_dir = path / "conformal"
        surrogates: dict[str, SurrogateResult] = {}

        for column, col_meta in metadata["columns"].items():
            feature_names = tuple(col_meta["feature_names"])
            monotonic_constraints: dict[str, MonotonicDirection] = {
                k: MonotonicDirection(v)
                for k, v in col_meta["monotonic_constraints"].items()
            }

            members: list[EnsembleMember] = []
            for i, member_meta in enumerate(col_meta["members"]):
                family_name = member_meta["estimator_family"]
                family = family_instances[family_name]
                model_path = models_dir / f"{column}_{i}"
                model = family.load_model(model_path)
                members.append(EnsembleMember(
                    trial_number=member_meta["trial_number"],
                    estimator_family=family_name,
                    model=model,
                    weight=member_meta["weight"],
                    cv_rmse=member_meta["cv_rmse"],
                ))

            ensemble = Ensemble(
                column=column,
                members=tuple(members),
                feature_names=feature_names,
                monotonic_constraints=monotonic_constraints,
                y_min=col_meta.get("y_min", -np.inf),
                y_max=col_meta.get("y_max", np.inf),
            )

            calib_data = np.load(conformal_dir / f"{column}.npz")
            conformal = ConformalCalibration(
                column=column,
                ensemble=ensemble,
                conformity_scores=calib_data["conformity_scores"],
                default_coverage=col_meta["default_coverage"],
            )

            trial_history = tuple(
                TrialRecord.model_validate(t) for t in col_meta["trial_history"]
            )

            surrogates[column] = SurrogateResult(
                column=column,
                ensemble=ensemble,
                conformal=conformal,
                trial_history=trial_history,
                ensemble_r2=col_meta.get("ensemble_r2", 0.0),
            )

        saved_versions = metadata.get("versions", {})
        current_versions = _collect_versions()
        _warn_version_mismatches(saved_versions, current_versions)

        dataset_fingerprint = metadata.get("dataset_fingerprint")
        if dataset_fingerprint is None:
            raise SurroxError("dataset_fingerprint missing in metadata.json")

        _logger.info(
            "surrogates loaded",
            extra={"path": str(path), "n_columns": len(surrogates)},
        )
        return cls(
            problem=problem,
            config=config,
            surrogates=surrogates,
            dataset_fingerprint=dataset_fingerprint,
        )

    @property
    def dataset_fingerprint(self) -> str:
        return self._dataset_fingerprint

    def _resolve_families(self) -> dict[str, EstimatorFamily]:
        return {
            family.name: family for family in self._config.estimator_families
        }


def _compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    import pandas as pd_mod

    hash_values = pd_mod.util.hash_pandas_object(df, index=False)
    combined = hashlib.sha256(hash_values.values.tobytes()).hexdigest()
    return combined


def _collect_versions() -> dict[str, str]:
    import importlib.metadata
    from contextlib import suppress

    packages = {
        "surrox": "surrox",
        "numpy": "numpy",
        "scikit-learn": "scikit-learn",
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "tabicl": "tabicl",
    }
    versions: dict[str, str] = {}
    for key, dist_name in packages.items():
        with suppress(importlib.metadata.PackageNotFoundError):
            versions[key] = importlib.metadata.version(dist_name)
    return versions


def _warn_version_mismatches(
    saved: dict[str, str], current: dict[str, str],
) -> None:
    for pkg, saved_ver in saved.items():
        current_ver = current.get(pkg)
        if current_ver and current_ver != saved_ver:
            _logger.warning(
                "version mismatch",
                extra={
                    "package": pkg,
                    "saved_version": saved_ver,
                    "current_version": current_ver,
                },
            )
