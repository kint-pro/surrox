from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
from sklearn.base import BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from surrox.problem.types import MonotonicDirection

_HIGH_DIM_THRESHOLD = 10


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
        n_features = hyperparameters.get("n_features")
        length_scale = 1.0
        if n_features is not None and n_features > _HIGH_DIM_THRESHOLD:
            length_scale = float(np.sqrt(n_features))

        kernel = ConstantKernel() * Matern(
            nu=hyperparameters["nu"],
            length_scale=length_scale,
        )
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
        if not isinstance(model, Pipeline):
            raise TypeError(f"expected Pipeline, got {type(model).__name__}")
        joblib.dump(model, path.with_suffix(".joblib"))

    def load_model(self, path: Path) -> BaseEstimator:
        return joblib.load(path.with_suffix(".joblib"))
