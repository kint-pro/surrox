import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.linear_model import LinearRegression

from surrox.surrogate.models import EnsembleMember, FoldMetrics, SurrogatePrediction, TrialRecord


class TestFoldMetrics:
    def test_construction(self) -> None:
        fm = FoldMetrics(
            fold=0, r2=0.95, rmse=0.1, mae=0.08,
            training_time_s=1.5, inference_time_ms=0.3,
        )
        assert fm.fold == 0
        assert fm.r2 == 0.95

    def test_frozen(self) -> None:
        fm = FoldMetrics(fold=0, r2=0.9, rmse=0.1, mae=0.08,
                         training_time_s=1.0, inference_time_ms=0.2)
        with pytest.raises(ValidationError):
            fm.r2 = 0.5


class TestTrialRecord:
    def test_construction(self) -> None:
        fm = FoldMetrics(fold=0, r2=0.9, rmse=0.2, mae=0.15,
                         training_time_s=1.0, inference_time_ms=0.5)
        record = TrialRecord(
            trial_number=0,
            estimator_family="xgboost",
            hyperparameters={"max_depth": 5},
            fold_metrics=(fm,),
            mean_r2=0.9, mean_rmse=0.2, mean_mae=0.15,
            mean_training_time_s=1.0, mean_inference_time_ms=0.5,
            status="completed",
        )
        assert record.status == "completed"
        assert len(record.fold_metrics) == 1


class TestEnsembleMember:
    def test_construction(self) -> None:
        model = LinearRegression()
        member = EnsembleMember(
            trial_number=0, estimator_family="xgboost",
            model=model, weight=0.5, cv_rmse=0.2,
        )
        assert member.weight == 0.5
        assert member.model is model


class TestSurrogatePrediction:
    def test_construction(self) -> None:
        pred = SurrogatePrediction(
            mean=np.array([1.0, 2.0]),
            std=np.array([0.1, 0.2]),
            lower=np.array([0.8, 1.6]),
            upper=np.array([1.2, 2.4]),
        )
        assert len(pred.mean) == 2
