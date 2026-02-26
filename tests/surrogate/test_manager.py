import numpy as np
import pandas as pd
import pytest

from surrox.problem.dataset import BoundDataset
from surrox.problem.definition import ProblemDefinition
from surrox.surrogate.config import TrainingConfig
from surrox.surrogate.ensemble import Ensemble
from surrox.surrogate.manager import SurrogateManager
from surrox.surrogate.models import SurrogatePrediction, TrialRecord


class TestSurrogateManager:
    @pytest.fixture
    def manager(
        self,
        problem_definition: ProblemDefinition,
        bound_dataset: BoundDataset,
        training_config: TrainingConfig,
    ) -> SurrogateManager:
        return SurrogateManager.train(
            problem=problem_definition,
            dataset=bound_dataset,
            config=training_config,
        )

    def test_train_produces_surrogates_for_all_columns(
        self, manager: SurrogateManager, problem_definition: ProblemDefinition
    ) -> None:
        for column in problem_definition.surrogate_columns:
            assert manager.get_ensemble(column) is not None

    def test_evaluate_returns_predictions_for_all_columns(
        self,
        manager: SurrogateManager,
        synthetic_dataframe: pd.DataFrame,
        problem_definition: ProblemDefinition,
    ) -> None:
        X = synthetic_dataframe.head(10)
        results = manager.evaluate(X)
        assert set(results.keys()) == set(problem_definition.surrogate_columns)
        for predictions in results.values():
            assert predictions.shape == (10,)

    def test_evaluate_with_uncertainty_returns_surrogate_predictions(
        self,
        manager: SurrogateManager,
        synthetic_dataframe: pd.DataFrame,
    ) -> None:
        X = synthetic_dataframe.head(10)
        results = manager.evaluate_with_uncertainty(X, coverage=0.9)
        for pred in results.values():
            assert isinstance(pred, SurrogatePrediction)
            assert pred.mean.shape == (10,)
            assert pred.lower.shape == (10,)
            assert pred.upper.shape == (10,)
            assert (pred.upper >= pred.lower).all()

    def test_get_ensemble_returns_ensemble(self, manager: SurrogateManager) -> None:
        ensemble = manager.get_ensemble("yield_pct")
        assert isinstance(ensemble, Ensemble)

    def test_get_ensemble_unknown_column_raises(self, manager: SurrogateManager) -> None:
        with pytest.raises(KeyError):
            manager.get_ensemble("nonexistent")

    def test_get_trial_history_returns_records(self, manager: SurrogateManager) -> None:
        history = manager.get_trial_history("yield_pct")
        assert isinstance(history, tuple)
        assert len(history) > 0
        assert all(isinstance(r, TrialRecord) for r in history)

    def test_get_trial_history_unknown_column_raises(
        self, manager: SurrogateManager
    ) -> None:
        with pytest.raises(KeyError):
            manager.get_trial_history("nonexistent")

    def test_evaluate_with_default_coverage(
        self, manager: SurrogateManager, synthetic_dataframe: pd.DataFrame
    ) -> None:
        X = synthetic_dataframe.head(5)
        results = manager.evaluate_with_uncertainty(X)
        assert len(results) > 0
