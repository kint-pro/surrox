from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from surrox.exceptions import SurroxError
from surrox.surrogate.conformal import ConformalCalibration
from surrox.surrogate.ensemble import Ensemble, EnsembleAdapter
from surrox.surrogate.families import LightGBMFamily, XGBoostFamily
from surrox.surrogate.models import EnsembleMember, SurrogatePrediction, TrialRecord
from surrox.surrogate.protocol import EstimatorFamily

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

    from surrox.problem.dataset import BoundDataset
    from surrox.problem.definition import ProblemDefinition
    from surrox.problem.types import MonotonicDirection
    from surrox.surrogate.config import TrainingConfig


_FAMILY_REGISTRY: dict[str, type[EstimatorFamily]] = {
    "xgboost": XGBoostFamily,  # type: ignore[type-abstract]
    "lightgbm": LightGBMFamily,  # type: ignore[type-abstract]
}


class SurrogateResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    column: str
    ensemble: Ensemble
    conformal: ConformalCalibration
    trial_history: tuple[TrialRecord, ...]


class SurrogateManager:
    def __init__(
        self,
        problem: ProblemDefinition,
        config: TrainingConfig,
        surrogates: dict[str, SurrogateResult],
    ) -> None:
        self._problem = problem
        self._config = config
        self._surrogates = surrogates

    @classmethod
    def train(
        cls,
        problem: ProblemDefinition,
        dataset: BoundDataset,
        config: TrainingConfig,
    ) -> SurrogateManager:
        from surrox.surrogate.pipeline import train_surrogate

        surrogates: dict[str, SurrogateResult] = {}
        for column in problem.surrogate_columns:
            surrogates[column] = train_surrogate(
                problem=problem,
                dataset_df=dataset.dataframe,
                config=config,
                column=column,
            )

        return cls(problem=problem, config=config, surrogates=surrogates)

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

    def get_trial_history(self, column: str) -> tuple[TrialRecord, ...]:
        return self._surrogates[column].trial_history

    def get_surrogate_result(self, column: str) -> SurrogateResult:
        return self._surrogates[column]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        models_dir = path / "models"
        models_dir.mkdir(exist_ok=True)
        conformal_dir = path / "conformal"
        conformal_dir.mkdir(exist_ok=True)

        metadata: dict[str, Any] = {
            "problem": json.loads(self._problem.model_dump_json()),
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
                X_calib=conformal.X_calib,
                y_calib=conformal.y_calib,
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
            }

        (path / "metadata.json").write_text(json.dumps(metadata, indent=2))

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
        config = TrainingConfig()

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
            )

            calib_data = np.load(conformal_dir / f"{column}.npz")
            adapter = EnsembleAdapter(ensemble=ensemble)
            conformal = ConformalCalibration(
                column=column,
                adapter=adapter,
                X_calib=calib_data["X_calib"],
                y_calib=calib_data["y_calib"],
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
            )

        return cls(problem=problem, config=config, surrogates=surrogates)

    def _resolve_families(self) -> dict[str, EstimatorFamily]:
        return {
            family.name: family for family in self._config.estimator_families
        }
