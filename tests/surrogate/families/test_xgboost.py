import optuna
import pytest

from surrox.problem.types import MonotonicDirection
from surrox.surrogate.families.xgboost import XGBoostFamily


class TestXGBoostFamily:
    def test_name(self, xgboost_family: XGBoostFamily) -> None:
        assert xgboost_family.name == "xgboost"

    def test_suggest_hyperparameters_returns_expected_keys(
        self, xgboost_family: XGBoostFamily
    ) -> None:
        study = optuna.create_study()
        trial = study.ask()
        params = xgboost_family.suggest_hyperparameters(trial)

        expected_keys = {
            "max_depth", "learning_rate", "n_estimators", "subsample",
            "colsample_bytree", "min_child_weight", "gamma", "reg_alpha", "reg_lambda",
        }
        assert set(params.keys()) == expected_keys

    def test_map_monotonic_constraints(self, xgboost_family: XGBoostFamily) -> None:
        constraints = {
            "temperature": MonotonicDirection.INCREASING,
            "pressure": MonotonicDirection.DECREASING,
        }
        feature_names = ["temperature", "pressure", "category"]
        categorical_features = {"category"}

        result = xgboost_family.map_monotonic_constraints(
            constraints, feature_names, categorical_features
        )
        assert result == {"temperature": 1, "pressure": -1}

    def test_map_monotonic_constraints_skips_categoricals(
        self, xgboost_family: XGBoostFamily
    ) -> None:
        constraints = {"category": MonotonicDirection.INCREASING}
        result = xgboost_family.map_monotonic_constraints(
            constraints, ["category"], {"category"}
        )
        assert result == {}

    def test_build_model_returns_unfitted_regressor(
        self, xgboost_family: XGBoostFamily
    ) -> None:
        from xgboost import XGBRegressor

        model = xgboost_family.build_model(
            hyperparameters={"max_depth": 5, "learning_rate": 0.1, "n_estimators": 100,
                             "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
                             "gamma": 0.01, "reg_alpha": 0.1, "reg_lambda": 1.0},
            monotonic_constraints={"temperature": 1},
            random_seed=42,
            n_threads=2,
        )
        assert isinstance(model, XGBRegressor)

    def test_build_model_with_no_threads(self, xgboost_family: XGBoostFamily) -> None:
        model = xgboost_family.build_model(
            hyperparameters={"max_depth": 3, "learning_rate": 0.1, "n_estimators": 50,
                             "subsample": 1.0, "colsample_bytree": 1.0, "min_child_weight": 1,
                             "gamma": 0.001, "reg_alpha": 0.001, "reg_lambda": 0.001},
            monotonic_constraints={},
            random_seed=42,
            n_threads=None,
        )
        assert model.get_params()["nthread"] is None
