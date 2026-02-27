from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from surrox.surrogate.families import LightGBMFamily, XGBoostFamily


class TestXGBoostFamilyPersistence:
    def test_save_load_roundtrip(self, tmp_path: pytest.TempPathFactory) -> None:
        family = XGBoostFamily()
        model = family.build_model(
            hyperparameters={"max_depth": 3, "learning_rate": 0.1, "n_estimators": 10},
            monotonic_constraints=None,
            random_seed=42,
            n_threads=1,
        )

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = X @ [1, 2, 3] + rng.normal(0, 0.1, 100)
        model.fit(X, y)

        original_preds = model.predict(X[:5])

        model_path = tmp_path / "xgb_model"  # type: ignore[operator]
        family.save_model(model, model_path)
        loaded = family.load_model(model_path)

        loaded_preds = loaded.predict(X[:5])
        np.testing.assert_array_almost_equal(loaded_preds, original_preds)


class TestLightGBMFamilyPersistence:
    def test_save_load_roundtrip(self, tmp_path: pytest.TempPathFactory) -> None:
        family = LightGBMFamily()
        model = family.build_model(
            hyperparameters={
                "max_depth": 3,
                "learning_rate": 0.1,
                "n_estimators": 10,
                "num_leaves": 8,
                "min_child_samples": 5,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_alpha": 0.0,
                "reg_lambda": 0.0,
            },
            monotonic_constraints=None,
            random_seed=42,
            n_threads=1,
        )

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = X @ [1, 2, 3] + rng.normal(0, 0.1, 100)
        model.fit(X, y)

        original_preds = model.predict(X[:5])

        model_path = tmp_path / "lgbm_model"  # type: ignore[operator]
        family.save_model(model, model_path)
        loaded = family.load_model(model_path)

        loaded_preds = loaded.predict(X[:5])
        np.testing.assert_array_almost_equal(loaded_preds, original_preds, decimal=5)

    def test_loaded_model_has_required_attributes(
        self, tmp_path: pytest.TempPathFactory,
    ) -> None:
        family = LightGBMFamily()
        model = family.build_model(
            hyperparameters={
                "max_depth": 3,
                "learning_rate": 0.1,
                "n_estimators": 10,
                "num_leaves": 8,
                "min_child_samples": 5,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_alpha": 0.0,
                "reg_lambda": 0.0,
            },
            monotonic_constraints=None,
            random_seed=42,
            n_threads=1,
        )

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = X @ [1, 2, 3]
        model.fit(X, y)

        model_path = tmp_path / "lgbm_attrs"  # type: ignore[operator]
        family.save_model(model, model_path)
        loaded = family.load_model(model_path)

        assert hasattr(loaded, "fitted_")
        assert loaded.fitted_ is True
        assert isinstance(loaded, BaseEstimator)
