import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from surrox.exceptions import ConfigurationError
from surrox.surrogate.conformal import ConformalCalibration
from surrox.surrogate.ensemble import Ensemble
from surrox.surrogate.models import EnsembleMember


def _make_conformal() -> ConformalCalibration:
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((200, 3))
    y_train = X_train @ [1, 2, 3] + rng.normal(0, 0.5, 200)
    X_calib_np = rng.standard_normal((100, 3))
    y_calib = X_calib_np @ [1, 2, 3] + rng.normal(0, 0.5, 100)
    X_calib = pd.DataFrame(X_calib_np, columns=["f1", "f2", "f3"])

    model = LinearRegression()
    model.fit(X_train, y_train)

    ensemble = Ensemble(
        column="target",
        members=(
            EnsembleMember(
                trial_number=0, estimator_family="mock",
                model=model, weight=1.0, cv_rmse=0.1,
            ),
        ),
        feature_names=("f1", "f2", "f3"),
        monotonic_constraints={},
    )
    return ConformalCalibration.from_calibration_data(
        column="target", ensemble=ensemble,
        X_calib=X_calib, y_calib=y_calib,
        default_coverage=0.9,
    )


class TestConformalCalibration:
    def test_prediction_interval_shapes(self) -> None:
        conformal = _make_conformal()
        X = pd.DataFrame(
            np.random.default_rng(0).standard_normal((10, 3)),
            columns=["f1", "f2", "f3"],
        )
        mean, lower, upper = conformal.prediction_interval(X, coverage=0.9)
        assert mean.shape == (10,)
        assert lower.shape == (10,)
        assert upper.shape == (10,)
        assert (upper >= lower).all()

    def test_higher_coverage_wider_intervals(self) -> None:
        conformal = _make_conformal()
        X = pd.DataFrame(
            np.random.default_rng(0).standard_normal((10, 3)),
            columns=["f1", "f2", "f3"],
        )
        _, lower_80, upper_80 = conformal.prediction_interval(X, coverage=0.8)
        _, lower_95, upper_95 = conformal.prediction_interval(X, coverage=0.95)
        width_80 = (upper_80 - lower_80).mean()
        width_95 = (upper_95 - lower_95).mean()
        assert width_95 > width_80

    def test_invalid_coverage_raises(self) -> None:
        conformal = _make_conformal()
        X = pd.DataFrame({"f1": [1.0], "f2": [2.0], "f3": [3.0]})
        with pytest.raises(ConfigurationError, match="coverage"):
            conformal.prediction_interval(X, coverage=0.0)
        with pytest.raises(ConfigurationError, match="coverage"):
            conformal.prediction_interval(X, coverage=1.0)

    def test_no_raw_data_stored(self) -> None:
        conformal = _make_conformal()
        assert not hasattr(conformal, "X_calib")
        assert not hasattr(conformal, "y_calib")
        assert hasattr(conformal, "conformity_scores")
        assert conformal.conformity_scores.ndim == 1
