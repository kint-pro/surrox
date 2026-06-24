import numpy as np
import pandas as pd
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from surrox.analysis.shapley import shapley_values
from surrox.surrogate.ensemble import Ensemble
from surrox.surrogate.models import EnsembleMember

FEATURE_NAMES = ("x0", "x1", "x2", "x3")


def _make_training_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.uniform(-2, 2, size=(120, 4)), columns=FEATURE_NAMES)
    y = (
        3 * X["x0"]
        - 2 * X["x1"] ** 2
        + np.sin(2 * X["x2"])
        + 0.5 * X["x3"]
        + rng.normal(scale=0.05, size=len(X))
    ).to_numpy()
    return X, y


def _fit_xgboost(X: pd.DataFrame, y: np.ndarray):
    import xgboost

    model = xgboost.XGBRegressor(n_estimators=40, max_depth=3, random_state=0)
    model.fit(X, y)
    return model


def _fit_lightgbm(X: pd.DataFrame, y: np.ndarray):
    import lightgbm

    model = lightgbm.LGBMRegressor(
        n_estimators=40, max_depth=3, verbose=-1, random_state=0
    )
    model.fit(X, y)
    return model


def _fit_gaussian_process(X: pd.DataFrame, y: np.ndarray):
    kernel = ConstantKernel() * Matern(nu=2.5)
    gpr = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-3, normalize_y=True, random_state=0
    )
    model = make_pipeline(StandardScaler(), gpr)
    model.fit(X, y)
    return model


def _member(family: str, model, weight: float) -> EnsembleMember:
    return EnsembleMember(
        trial_number=0,
        estimator_family=family,
        model=model,
        weight=weight,
        cv_rmse=0.1,
    )


@pytest.fixture(scope="module")
def training_data() -> tuple[pd.DataFrame, np.ndarray]:
    return _make_training_data()


@pytest.fixture(scope="module")
def fitted_models(training_data: tuple[pd.DataFrame, np.ndarray]) -> dict[str, object]:
    X, y = training_data
    return {
        "xgboost": _fit_xgboost(X, y),
        "lightgbm": _fit_lightgbm(X, y),
        "gaussian_process": _fit_gaussian_process(X, y),
    }


def _ensemble(members: tuple[EnsembleMember, ...]) -> Ensemble:
    return Ensemble(
        column="target",
        members=members,
        feature_names=FEATURE_NAMES,
        monotonic_constraints={},
    )


def _check_efficiency(ensemble: Ensemble, X: pd.DataFrame, threshold: int) -> None:
    background = ensemble._prepare_features(X.iloc[:30])
    instances = ensemble._prepare_features(X.iloc[30:34])
    result = shapley_values(
        predict=ensemble.predict,
        instances=instances,
        background=background,
        feature_names=FEATURE_NAMES,
        exact_threshold=threshold,
        sampling_permutations=200,
        rng=np.random.default_rng(0),
    )
    reconstructed = result.base_value + result.shap_values.sum(axis=1)
    np.testing.assert_allclose(reconstructed, ensemble.predict(instances), atol=1e-6)


class TestSingleArchitecture:
    @pytest.mark.parametrize("family", ["xgboost", "lightgbm", "gaussian_process"])
    def test_exact_efficiency_per_architecture(
        self,
        family: str,
        fitted_models: dict[str, object],
        training_data: tuple[pd.DataFrame, np.ndarray],
    ) -> None:
        X, _ = training_data
        ensemble = _ensemble((_member(family, fitted_models[family], 1.0),))
        _check_efficiency(ensemble, X, threshold=12)

    @pytest.mark.parametrize("family", ["xgboost", "lightgbm", "gaussian_process"])
    def test_sampling_efficiency_per_architecture(
        self,
        family: str,
        fitted_models: dict[str, object],
        training_data: tuple[pd.DataFrame, np.ndarray],
    ) -> None:
        X, _ = training_data
        ensemble = _ensemble((_member(family, fitted_models[family], 1.0),))
        _check_efficiency(ensemble, X, threshold=2)


class TestHeterogeneousEnsemble:
    def test_tree_and_gp_mixed_exact(
        self,
        fitted_models: dict[str, object],
        training_data: tuple[pd.DataFrame, np.ndarray],
    ) -> None:
        X, _ = training_data
        ensemble = _ensemble(
            (
                _member("xgboost", fitted_models["xgboost"], 0.4),
                _member("lightgbm", fitted_models["lightgbm"], 0.3),
                _member("gaussian_process", fitted_models["gaussian_process"], 0.3),
            )
        )
        _check_efficiency(ensemble, X, threshold=12)

    def test_tree_and_gp_mixed_sampling_matches_exact(
        self,
        fitted_models: dict[str, object],
        training_data: tuple[pd.DataFrame, np.ndarray],
    ) -> None:
        X, _ = training_data
        ensemble = _ensemble(
            (
                _member("xgboost", fitted_models["xgboost"], 0.5),
                _member("gaussian_process", fitted_models["gaussian_process"], 0.5),
            )
        )
        background = ensemble._prepare_features(X.iloc[:30])
        instances = ensemble._prepare_features(X.iloc[30:33])
        exact = shapley_values(
            predict=ensemble.predict,
            instances=instances,
            background=background,
            feature_names=FEATURE_NAMES,
            exact_threshold=12,
            sampling_permutations=1,
            rng=np.random.default_rng(0),
        )
        sampled = shapley_values(
            predict=ensemble.predict,
            instances=instances,
            background=background,
            feature_names=FEATURE_NAMES,
            exact_threshold=2,
            sampling_permutations=400,
            rng=np.random.default_rng(0),
        )
        np.testing.assert_allclose(sampled.shap_values, exact.shap_values, atol=0.05)
