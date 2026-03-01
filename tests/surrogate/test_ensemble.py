import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from surrox.surrogate.ensemble import Ensemble
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

    def test_predict_clips_to_y_bounds(self) -> None:
        members = tuple(
            EnsembleMember(
                trial_number=i,
                estimator_family="mock",
                model=_MockModel(constant=c),
                weight=0.5,
                cv_rmse=0.1,
            )
            for i, c in enumerate([100.0, 200.0])
        )
        ensemble = Ensemble(
            column="target",
            members=members,
            feature_names=("x1", "x2"),
            monotonic_constraints={},
            y_min=0.0,
            y_max=120.0,
        )
        X = pd.DataFrame({"x1": [1.0], "x2": [2.0]})
        result = ensemble.predict(X)
        assert result[0] == 120.0

    def test_predict_with_std_clips_mean_not_std(self) -> None:
        members = tuple(
            EnsembleMember(
                trial_number=i,
                estimator_family="mock",
                model=_MockModel(constant=c),
                weight=0.5,
                cv_rmse=0.1,
            )
            for i, c in enumerate([100.0, 200.0])
        )
        ensemble = Ensemble(
            column="target",
            members=members,
            feature_names=("x1", "x2"),
            monotonic_constraints={},
            y_min=0.0,
            y_max=120.0,
        )
        X = pd.DataFrame({"x1": [1.0], "x2": [2.0]})
        mean, std = ensemble.predict_with_std(X)
        assert mean[0] == 120.0
        np.testing.assert_allclose(std, [50.0])
