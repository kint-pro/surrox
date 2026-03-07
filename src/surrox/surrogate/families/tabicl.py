from pathlib import Path
from typing import Any

import optuna
from sklearn.base import BaseEstimator
from tabicl import TabICLRegressor

from surrox.problem.types import MonotonicDirection


class TabICLFamily:
    @property
    def name(self) -> str:
        return "tabicl"

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        return {}

    def build_model(
        self,
        hyperparameters: dict[str, Any],
        monotonic_constraints: Any,
        random_seed: int,
        n_threads: int | None,
    ) -> TabICLRegressor:
        return TabICLRegressor(random_state=random_seed, n_jobs=n_threads)

    def map_monotonic_constraints(
        self,
        constraints: dict[str, MonotonicDirection],
        feature_names: list[str],
        categorical_features: set[str],
    ) -> None:
        return None

    def save_model(self, model: BaseEstimator, path: Path) -> None:
        if not isinstance(model, TabICLRegressor):
            raise TypeError(f"expected TabICLRegressor, got {type(model).__name__}")
        model.save(path.with_suffix(".tabicl"), save_training_data=True)

    def load_model(self, path: Path) -> TabICLRegressor:
        return TabICLRegressor.load(path.with_suffix(".tabicl"))
