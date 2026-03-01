from pathlib import Path
from typing import Any

import joblib
import optuna
from sklearn.base import BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from surrox.problem.types import MonotonicDirection


class GaussianProcessFamily:
    @property
    def name(self) -> str:
        return "gaussian_process"

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "length_scale": trial.suggest_float("length_scale", 0.01, 10.0, log=True),
            "constant_value": trial.suggest_float("constant_value", 0.01, 100.0, log=True),
            "noise_level": trial.suggest_float("noise_level", 1e-5, 1.0, log=True),
            "nu": trial.suggest_categorical("nu", [1.5, 2.5]),
        }

    def build_model(
        self,
        hyperparameters: dict[str, Any],
        monotonic_constraints: Any,
        random_seed: int,
        n_threads: int | None,
    ) -> GaussianProcessRegressor:
        kernel = (
            ConstantKernel(constant_value=hyperparameters["constant_value"])
            * Matern(
                length_scale=hyperparameters["length_scale"],
                nu=hyperparameters["nu"],
            )
            + WhiteKernel(noise_level=hyperparameters["noise_level"])
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=0,
            normalize_y=True,
            random_state=random_seed,
        )

    def map_monotonic_constraints(
        self,
        constraints: dict[str, MonotonicDirection],
        feature_names: list[str],
        categorical_features: set[str],
    ) -> None:
        return None

    def save_model(self, model: BaseEstimator, path: Path) -> None:
        if not isinstance(model, GaussianProcessRegressor):
            raise TypeError(
                f"expected GaussianProcessRegressor, got {type(model).__name__}"
            )
        joblib.dump(model, path.with_suffix(".joblib"))

    def load_model(self, path: Path) -> GaussianProcessRegressor:
        model = joblib.load(path.with_suffix(".joblib"))
        if not isinstance(model, GaussianProcessRegressor):
            raise TypeError(
                f"expected GaussianProcessRegressor, got {type(model).__name__}"
            )
        return model
