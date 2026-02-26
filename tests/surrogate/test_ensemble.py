import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from surrox.surrogate.ensemble import Ensemble, EnsembleAdapter
from surrox.surrogate.models import EnsembleMember


class _MockModel(BaseEstimator):
    def __init__(self, constant: float) -> None:
        self.constant = constant

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.constant)


def _make_ensemble(constants: list[float], weights: list[float]) -> Ensemble:
    members = tuple(
        EnsembleMember(
            trial_number=i,
            estimator_family="mock",
            model=_MockModel(constant=c),
            weight=w,
            cv_rmse=0.1,
        )
        for i, (c, w) in enumerate(zip(constants, weights))
    )
    return Ensemble(
        column="target",
        members=members,
        feature_names=("x1", "x2"),
        monotonic_constraints={},
    )


class TestEnsemble:
    def test_predict_weighted_average(self) -> None:
        ensemble = _make_ensemble([10.0, 20.0], [0.3, 0.7])
        X = pd.DataFrame({"x1": [1.0, 2.0], "x2": [3.0, 4.0]})
        result = ensemble.predict(X)
        expected = 0.3 * 10.0 + 0.7 * 20.0
        np.testing.assert_allclose(result, [expected, expected])

    def test_predict_with_std(self) -> None:
        ensemble = _make_ensemble([10.0, 20.0], [0.5, 0.5])
        X = pd.DataFrame({"x1": [1.0], "x2": [2.0]})
        mean, std = ensemble.predict_with_std(X)
        np.testing.assert_allclose(mean, [15.0])
        np.testing.assert_allclose(std, [5.0])

    def test_single_member_std_is_zero(self) -> None:
        ensemble = _make_ensemble([10.0], [1.0])
        X = pd.DataFrame({"x1": [1.0], "x2": [2.0]})
        mean, std = ensemble.predict_with_std(X)
        np.testing.assert_allclose(mean, [10.0])
        np.testing.assert_allclose(std, [0.0])


class TestEnsembleAdapter:
    def test_predict_delegates_to_ensemble(self) -> None:
        ensemble = _make_ensemble([5.0], [1.0])
        adapter = EnsembleAdapter(ensemble)
        X = pd.DataFrame({"x1": [1.0], "x2": [2.0]})
        result = adapter.predict(X)
        np.testing.assert_allclose(result, [5.0])

    def test_predict_with_numpy_array(self) -> None:
        ensemble = _make_ensemble([5.0], [1.0])
        adapter = EnsembleAdapter(ensemble)
        X = np.array([[1.0, 2.0]])
        result = adapter.predict(X)
        np.testing.assert_allclose(result, [5.0])

    def test_fit_returns_self(self) -> None:
        ensemble = _make_ensemble([5.0], [1.0])
        adapter = EnsembleAdapter(ensemble)
        result = adapter.fit(None, None)
        assert result is adapter

    def test_is_fitted(self) -> None:
        ensemble = _make_ensemble([5.0], [1.0])
        adapter = EnsembleAdapter(ensemble)
        assert adapter.__sklearn_is_fitted__() is True
