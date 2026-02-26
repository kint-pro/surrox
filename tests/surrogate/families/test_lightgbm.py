import optuna
import pytest

from surrox.problem.types import MonotonicDirection
from surrox.surrogate.families.lightgbm import LightGBMFamily


class TestLightGBMFamily:
    def test_name(self, lightgbm_family: LightGBMFamily) -> None:
        assert lightgbm_family.name == "lightgbm"

    def test_suggest_hyperparameters_returns_expected_keys(
        self, lightgbm_family: LightGBMFamily
    ) -> None:
        study = optuna.create_study()
        trial = study.ask()
        params = lightgbm_family.suggest_hyperparameters(trial)

        expected_keys = {
            "num_leaves", "learning_rate", "n_estimators", "subsample",
            "colsample_bytree", "min_child_samples", "reg_alpha", "reg_lambda",
        }
        assert set(params.keys()) == expected_keys

    def test_map_monotonic_constraints_positional(
        self, lightgbm_family: LightGBMFamily
    ) -> None:
        constraints = {
            "temperature": MonotonicDirection.INCREASING,
            "pressure": MonotonicDirection.DECREASING,
        }
        feature_names = ["temperature", "pressure", "duration"]
        categorical_features: set[str] = set()

        result = lightgbm_family.map_monotonic_constraints(
            constraints, feature_names, categorical_features
        )
        assert result == [1, -1, 0]

    def test_map_monotonic_constraints_zeros_for_categoricals(
        self, lightgbm_family: LightGBMFamily
    ) -> None:
        constraints = {"category": MonotonicDirection.INCREASING}
        result = lightgbm_family.map_monotonic_constraints(
            constraints, ["temperature", "category"], {"category"}
        )
        assert result == [0, 0]

    def test_build_model_returns_unfitted_regressor(
        self, lightgbm_family: LightGBMFamily
    ) -> None:
        from lightgbm import LGBMRegressor

        model = lightgbm_family.build_model(
            hyperparameters={"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 100,
                             "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20,
                             "reg_alpha": 0.1, "reg_lambda": 1.0},
            monotonic_constraints=[1, -1, 0],
            random_seed=42,
            n_threads=2,
        )
        assert isinstance(model, LGBMRegressor)

    def test_build_model_defaults_threads_to_minus_one(
        self, lightgbm_family: LightGBMFamily
    ) -> None:
        model = lightgbm_family.build_model(
            hyperparameters={"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 50,
                             "subsample": 1.0, "colsample_bytree": 1.0, "min_child_samples": 5,
                             "reg_alpha": 0.001, "reg_lambda": 0.001},
            monotonic_constraints=[],
            random_seed=42,
            n_threads=None,
        )
        assert model.get_params()["num_threads"] == -1
