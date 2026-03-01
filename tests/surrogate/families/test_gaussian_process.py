import optuna
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor

from surrox.problem.types import MonotonicDirection
from surrox.surrogate.families.gaussian_process import GaussianProcessFamily


class TestGaussianProcessFamily:
    def test_name(self, gaussian_process_family: GaussianProcessFamily) -> None:
        assert gaussian_process_family.name == "gaussian_process"

    def test_suggest_hyperparameters_returns_expected_keys(
        self, gaussian_process_family: GaussianProcessFamily
    ) -> None:
        study = optuna.create_study()
        trial = study.ask()
        params = gaussian_process_family.suggest_hyperparameters(trial)

        expected_keys = {"length_scale", "constant_value", "noise_level", "nu"}
        assert set(params.keys()) == expected_keys

    def test_build_model_returns_gpr(
        self, gaussian_process_family: GaussianProcessFamily
    ) -> None:
        model = gaussian_process_family.build_model(
            hyperparameters={
                "length_scale": 1.0,
                "constant_value": 1.0,
                "noise_level": 1e-3,
                "nu": 2.5,
            },
            monotonic_constraints={},
            random_seed=42,
            n_threads=None,
        )
        assert isinstance(model, GaussianProcessRegressor)

    def test_build_model_normalizes_y(
        self, gaussian_process_family: GaussianProcessFamily
    ) -> None:
        model = gaussian_process_family.build_model(
            hyperparameters={
                "length_scale": 1.0,
                "constant_value": 1.0,
                "noise_level": 1e-3,
                "nu": 1.5,
            },
            monotonic_constraints={},
            random_seed=42,
            n_threads=None,
        )
        assert model.normalize_y is True

    def test_map_monotonic_constraints_returns_none(
        self, gaussian_process_family: GaussianProcessFamily
    ) -> None:
        constraints = {"temperature": MonotonicDirection.INCREASING}
        result = gaussian_process_family.map_monotonic_constraints(
            constraints, ["temperature"], set()
        )
        assert result is None

    def test_save_load_roundtrip(
        self, gaussian_process_family: GaussianProcessFamily, tmp_path: pytest.TempPathFactory
    ) -> None:
        import numpy as np

        model = gaussian_process_family.build_model(
            hyperparameters={
                "length_scale": 1.0,
                "constant_value": 1.0,
                "noise_level": 1e-3,
                "nu": 2.5,
            },
            monotonic_constraints={},
            random_seed=42,
            n_threads=None,
        )
        rng = np.random.default_rng(42)
        X_train = rng.uniform(0, 1, (20, 2))
        y_train = np.sin(X_train[:, 0]) + 0.1 * rng.normal(size=20)
        model.fit(X_train, y_train)

        model_path = tmp_path / "gp_model"
        gaussian_process_family.save_model(model, model_path)
        loaded = gaussian_process_family.load_model(model_path)

        X_test = rng.uniform(0, 1, (5, 2))
        np.testing.assert_array_almost_equal(
            model.predict(X_test), loaded.predict(X_test)
        )

    def test_save_rejects_wrong_type(
        self, gaussian_process_family: GaussianProcessFamily, tmp_path: pytest.TempPathFactory
    ) -> None:
        from unittest.mock import MagicMock

        with pytest.raises(TypeError, match="expected GaussianProcessRegressor"):
            gaussian_process_family.save_model(MagicMock(), tmp_path / "bad")
