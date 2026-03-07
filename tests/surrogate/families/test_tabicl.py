import numpy as np
import optuna
import pytest

from surrox.problem.types import MonotonicDirection
from surrox.surrogate.families.tabicl import TabICLFamily

tabicl = pytest.importorskip("tabicl")


class TestTabICLFamily:
    def test_name(self, tabicl_family: TabICLFamily) -> None:
        assert tabicl_family.name == "tabicl"

    def test_suggest_hyperparameters_returns_empty_dict(
        self, tabicl_family: TabICLFamily
    ) -> None:
        study = optuna.create_study()
        trial = study.ask()
        params = tabicl_family.suggest_hyperparameters(trial)

        assert params == {}

    def test_build_model_returns_tabicl_regressor(
        self, tabicl_family: TabICLFamily
    ) -> None:
        model = tabicl_family.build_model(
            hyperparameters={},
            monotonic_constraints=None,
            random_seed=42,
            n_threads=1,
        )
        assert isinstance(model, tabicl.TabICLRegressor)

    def test_build_model_passes_n_jobs(
        self, tabicl_family: TabICLFamily
    ) -> None:
        model = tabicl_family.build_model(
            hyperparameters={},
            monotonic_constraints=None,
            random_seed=42,
            n_threads=2,
        )
        assert model.n_jobs == 2

    def test_map_monotonic_constraints_returns_none(
        self, tabicl_family: TabICLFamily
    ) -> None:
        constraints = {"temperature": MonotonicDirection.INCREASING}
        result = tabicl_family.map_monotonic_constraints(
            constraints, ["temperature"], set()
        )
        assert result is None

    def test_save_load_roundtrip(
        self, tabicl_family: TabICLFamily, tmp_path: pytest.TempPathFactory
    ) -> None:
        model = tabicl_family.build_model(
            hyperparameters={},
            monotonic_constraints=None,
            random_seed=42,
            n_threads=1,
        )
        rng = np.random.default_rng(42)
        X_train = rng.uniform(0, 1, (20, 2))
        y_train = np.sin(X_train[:, 0]) + 0.1 * rng.normal(size=20)
        model.fit(X_train, y_train)

        model_path = tmp_path / "tabicl_model"
        tabicl_family.save_model(model, model_path)
        loaded = tabicl_family.load_model(model_path)

        X_test = rng.uniform(0, 1, (5, 2))
        np.testing.assert_array_almost_equal(
            model.predict(X_test), loaded.predict(X_test)
        )

    def test_save_rejects_wrong_type(
        self, tabicl_family: TabICLFamily, tmp_path: pytest.TempPathFactory
    ) -> None:
        from unittest.mock import MagicMock

        with pytest.raises(TypeError, match="expected TabICLRegressor"):
            tabicl_family.save_model(MagicMock(), tmp_path / "bad")
