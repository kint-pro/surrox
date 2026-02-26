from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from surrox.problem.definition import ProblemDefinition
from surrox.surrogate.config import TrainingConfig
from surrox.surrogate.pipeline import (
    _max_correlation,
    _validate_minimum_data,
    train_surrogate,
)


class TestValidateMinimumData:
    def test_sufficient_data_passes(self) -> None:
        config = TrainingConfig(cv_folds=5, calibration_fraction=0.2)
        _validate_minimum_data(500, config)

    def test_insufficient_data_raises(self) -> None:
        config = TrainingConfig(cv_folds=5, calibration_fraction=0.2)
        with pytest.raises(ValueError, match="minimum is 500"):
            _validate_minimum_data(499, config)

    def test_minimum_driven_by_calibration(self) -> None:
        config = TrainingConfig(cv_folds=2, calibration_fraction=0.1)
        with pytest.raises(ValueError, match="minimum is 1000"):
            _validate_minimum_data(999, config)

    def test_minimum_driven_by_cv_folds(self) -> None:
        config = TrainingConfig(cv_folds=10, calibration_fraction=0.5)
        with pytest.raises(ValueError, match="minimum is 1000"):
            _validate_minimum_data(999, config)


class TestTrainSurrogate:
    def test_produces_surrogate_result(
        self,
        problem_definition: ProblemDefinition,
        synthetic_dataframe: pd.DataFrame,
        training_config: TrainingConfig,
    ) -> None:
        result = train_surrogate(
            problem=problem_definition,
            dataset_df=synthetic_dataframe,
            config=training_config,
            column="yield_pct",
        )
        assert result.column == "yield_pct"
        assert len(result.ensemble.members) >= 1
        assert len(result.trial_history) > 0
        assert all(m.weight > 0 for m in result.ensemble.members)

    def test_ensemble_weights_sum_to_one(
        self,
        problem_definition: ProblemDefinition,
        synthetic_dataframe: pd.DataFrame,
        training_config: TrainingConfig,
    ) -> None:
        result = train_surrogate(
            problem=problem_definition,
            dataset_df=synthetic_dataframe,
            config=training_config,
            column="yield_pct",
        )
        total_weight = sum(m.weight for m in result.ensemble.members)
        np.testing.assert_allclose(total_weight, 1.0, atol=1e-10)

    def test_conformal_intervals_have_correct_shape(
        self,
        problem_definition: ProblemDefinition,
        synthetic_dataframe: pd.DataFrame,
        training_config: TrainingConfig,
    ) -> None:
        result = train_surrogate(
            problem=problem_definition,
            dataset_df=synthetic_dataframe,
            config=training_config,
            column="yield_pct",
        )
        X = synthetic_dataframe[["temperature", "pressure", "duration"]].head(5)
        mean, lower, upper = result.conformal.prediction_interval(X, coverage=0.9)
        assert mean.shape == (5,)
        assert (upper >= lower).all()

    def test_insufficient_data_raises(
        self,
        problem_definition: ProblemDefinition,
        synthetic_dataframe: pd.DataFrame,
        training_config: TrainingConfig,
    ) -> None:
        with pytest.raises(ValueError, match="minimum"):
            train_surrogate(
                problem=problem_definition,
                dataset_df=synthetic_dataframe.head(50),
                config=training_config,
                column="yield_pct",
            )

    def test_quality_gate_rejects_poor_surrogate(
        self,
        problem_definition: ProblemDefinition,
        training_config: TrainingConfig,
    ) -> None:
        rng = np.random.default_rng(42)
        n = 600
        df = pd.DataFrame({
            "temperature": rng.uniform(0, 100, n),
            "pressure": rng.uniform(0, 200, n),
            "duration": rng.integers(1, 51, n),
            "yield_pct": rng.normal(0, 1, n),
            "emissions": rng.normal(0, 1, n),
        })

        config = TrainingConfig(
            n_trials=5,
            cv_folds=3,
            study_timeout_s=60,
            ensemble_size=3,
            min_r2=0.9,
        )

        with pytest.raises(ValueError, match="R².*below minimum threshold"):
            train_surrogate(
                problem=problem_definition,
                dataset_df=df,
                config=config,
                column="yield_pct",
            )


class TestMaxCorrelation:
    def test_high_correlation(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        assert _max_correlation(a, [b]) > 0.99

    def test_low_correlation(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.standard_normal(100)
        b = rng.standard_normal(100)
        assert _max_correlation(a, [b]) < 0.5

    def test_nan_from_constant_predictions_returns_one(self) -> None:
        constant = np.full(10, 5.0)
        varying = np.arange(10, dtype=float)
        assert _max_correlation(constant, [varying]) == 1.0

    def test_both_constant_returns_one(self) -> None:
        a = np.full(10, 3.0)
        b = np.full(10, 7.0)
        assert _max_correlation(a, [b]) == 1.0

    def test_returns_max_across_multiple(self) -> None:
        candidate = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        high_corr = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        low_corr = np.array([5.0, 3.0, 1.0, 4.0, 2.0])
        result = _max_correlation(candidate, [low_corr, high_corr])
        assert result > 0.99


