from pathlib import Path
from typing import Any

import lightgbm as lgb
import optuna
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator

from surrox.problem.types import MonotonicDirection

_DIRECTION_MAP = {
    MonotonicDirection.INCREASING: 1,
    MonotonicDirection.DECREASING: -1,
}


class LightGBMFamily:
    @property
    def name(self) -> str:
        return "lightgbm"

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    def build_model(
        self,
        hyperparameters: dict[str, Any],
        monotonic_constraints: Any,
        random_seed: int,
        n_threads: int | None,
    ) -> LGBMRegressor:
        return LGBMRegressor(
            **hyperparameters,
            monotone_constraints=monotonic_constraints,
            random_state=random_seed,
            num_threads=n_threads if n_threads is not None else -1,
            verbosity=-1,
        )

    def map_monotonic_constraints(
        self,
        constraints: dict[str, MonotonicDirection],
        feature_names: list[str],
        categorical_features: set[str],
    ) -> list[int]:
        return [
            _DIRECTION_MAP.get(constraints[name], 0)
            if name not in categorical_features and name in constraints
            else 0
            for name in feature_names
        ]

    def save_model(self, model: BaseEstimator, path: Path) -> None:
        if not isinstance(model, LGBMRegressor):
            raise TypeError(f"expected LGBMRegressor, got {type(model).__name__}")
        model.booster_.save_model(str(path.with_suffix(".lgbm")))

    def load_model(self, path: Path) -> LGBMRegressor:
        booster = lgb.Booster(model_file=str(path.with_suffix(".lgbm")))
        model = LGBMRegressor()
        model._Booster = booster
        model.fitted_ = True
        n_features = booster.num_feature()
        model._n_features = n_features
        model.n_features_in_ = n_features
        return model
