from pathlib import Path
from typing import Any

import joblib
import optuna
from sklearn.base import BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from surrox.problem.types import MonotonicDirection


class GaussianProcessFamily:
    @property
    def name(self) -> str:
        return "gaussian_process"

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "nu": trial.suggest_categorical("nu", [0.5, 1.5, 2.5]),
            "alpha": trial.suggest_float("alpha", 1e-10, 1e-2, log=True),
        }

    def build_model(
        self,
        hyperparameters: dict[str, Any],
        monotonic_constraints: Any,
        random_seed: int,
        n_threads: int | None,
    ) -> Pipeline:
        kernel = ConstantKernel() * Matern(nu=hyperparameters["nu"])
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=hyperparameters["alpha"],
            n_restarts_optimizer=2,
            normalize_y=True,
            random_state=random_seed,
        )
        return make_pipeline(StandardScaler(), gpr)

    def map_monotonic_constraints(
        self,
        constraints: dict[str, MonotonicDirection],
        feature_names: list[str],
        categorical_features: set[str],
    ) -> None:
        return None

    def save_model(self, model: BaseEstimator, path: Path) -> None:
        joblib.dump(model, path.with_suffix(".joblib"))

    def load_model(self, path: Path) -> BaseEstimator:
        return joblib.load(path.with_suffix(".joblib"))
