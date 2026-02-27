from pathlib import Path
from typing import Any

import optuna
from sklearn.base import BaseEstimator
from xgboost import XGBRegressor

from surrox.problem.types import MonotonicDirection

_DIRECTION_MAP = {
    MonotonicDirection.INCREASING: 1,
    MonotonicDirection.DECREASING: -1,
}


class XGBoostFamily:
    @property
    def name(self) -> str:
        return "xgboost"

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    def build_model(
        self,
        hyperparameters: dict[str, Any],
        monotonic_constraints: Any,
        random_seed: int,
        n_threads: int | None,
    ) -> XGBRegressor:
        return XGBRegressor(
            **hyperparameters,
            monotone_constraints=monotonic_constraints,
            random_state=random_seed,
            nthread=n_threads,
            verbosity=0,
        )

    def map_monotonic_constraints(
        self,
        constraints: dict[str, MonotonicDirection],
        feature_names: list[str],
        categorical_features: set[str],
    ) -> dict[str, int]:
        return {
            name: _DIRECTION_MAP[direction]
            for name, direction in constraints.items()
            if name not in categorical_features
        }

    def save_model(self, model: BaseEstimator, path: Path) -> None:
        if not isinstance(model, XGBRegressor):
            raise TypeError(f"expected XGBRegressor, got {type(model).__name__}")
        model.save_model(path.with_suffix(".ubj"))

    def load_model(self, path: Path) -> XGBRegressor:
        model = XGBRegressor()
        model.load_model(path.with_suffix(".ubj"))
        return model
