from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import surrox
from surrox import (
    ObjectivePrediction,
    OptimizerConfig,
    Suggestion,
    SuggestionResult,
    TrainingConfig,
)
from surrox.exceptions import ConfigurationError
from surrox.optimizer.result import EvaluatedPoint
from surrox.problem.definition import ProblemDefinition
from surrox.problem.objectives import Objective
from surrox.problem.scenarios import Scenario
from surrox.problem.types import Direction, DType, Role
from surrox.problem.variables import ContinuousBounds, Variable
from surrox.surrogate.models import SurrogatePrediction


@pytest.fixture
def problem() -> ProblemDefinition:
    return ProblemDefinition(
        variables=(
            Variable(
                name="x1", dtype=DType.CONTINUOUS, role=Role.DECISION,
                bounds=ContinuousBounds(lower=0.0, upper=10.0),
            ),
            Variable(
                name="x2", dtype=DType.CONTINUOUS, role=Role.DECISION,
                bounds=ContinuousBounds(lower=0.0, upper=10.0),
            ),
        ),
        objectives=(
            Objective(name="cost", direction=Direction.MINIMIZE, column="cost"),
        ),
    )


@pytest.fixture
def problem_with_context() -> ProblemDefinition:
    return ProblemDefinition(
        variables=(
            Variable(
                name="x1", dtype=DType.CONTINUOUS, role=Role.DECISION,
                bounds=ContinuousBounds(lower=0.0, upper=10.0),
            ),
            Variable(
                name="location", dtype=DType.CONTINUOUS, role=Role.CONTEXT,
                bounds=ContinuousBounds(lower=0.0, upper=5.0),
            ),
        ),
        objectives=(
            Objective(name="cost", direction=Direction.MINIMIZE, column="cost"),
        ),
    )


@pytest.fixture
def dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame({
        "x1": rng.uniform(0, 10, n),
        "x2": rng.uniform(0, 10, n),
        "cost": rng.uniform(20, 80, n),
    })


def _mock_candidates(n: int) -> tuple[EvaluatedPoint, ...]:
    return tuple(
        EvaluatedPoint(
            variables={"x1": float(i), "x2": float(i + 1)},
            objectives={"cost": 50.0 - i * 5.0},
            constraints=(),
            feasible=True,
            extrapolation_distance=0.1,
            is_extrapolating=False,
        )
        for i in range(n)
    )


def _mock_surrogate_manager(columns: list[str]) -> MagicMock:
    manager = MagicMock()
    manager.evaluate_with_uncertainty.side_effect = lambda df, coverage=None: {
        col: SurrogatePrediction(
            mean=np.full(len(df), 50.0),
            std=np.full(len(df), 5.0),
            lower=np.full(len(df), 40.0),
            upper=np.full(len(df), 60.0),
        )
        for col in columns
    }
    manager.get_ensemble_r2.side_effect = lambda c: 0.95
    return manager


class TestSuggest:
    @patch("surrox.suggest.suggest_candidates")
    @patch("surrox.suggest.SurrogateManager.train")
    def test_returns_suggestion_result(
        self,
        mock_train: MagicMock,
        mock_suggest: MagicMock,
        problem: ProblemDefinition,
        dataframe: pd.DataFrame,
    ) -> None:
        mock_manager = _mock_surrogate_manager(["cost"])
        mock_train.return_value = mock_manager
        mock_suggest.return_value = _mock_candidates(3)

        result = surrox.suggest(problem, dataframe, n_suggestions=3)

        assert isinstance(result, SuggestionResult)
        assert len(result.suggestions) == 3
        assert all(isinstance(s, Suggestion) for s in result.suggestions)

    @patch("surrox.suggest.suggest_candidates")
    @patch("surrox.suggest.SurrogateManager.train")
    def test_suggestions_have_uncertainty(
        self,
        mock_train: MagicMock,
        mock_suggest: MagicMock,
        problem: ProblemDefinition,
        dataframe: pd.DataFrame,
    ) -> None:
        mock_manager = _mock_surrogate_manager(["cost"])
        mock_train.return_value = mock_manager
        mock_suggest.return_value = _mock_candidates(2)

        result = surrox.suggest(problem, dataframe, n_suggestions=2)

        suggestion = result.suggestions[0]
        assert "cost" in suggestion.objectives
        pred = suggestion.objectives["cost"]
        assert isinstance(pred, ObjectivePrediction)
        assert pred.std > 0
        assert pred.lower < pred.mean < pred.upper

    @patch("surrox.suggest.suggest_candidates")
    @patch("surrox.suggest.SurrogateManager.train")
    def test_surrogate_quality_populated(
        self,
        mock_train: MagicMock,
        mock_suggest: MagicMock,
        problem: ProblemDefinition,
        dataframe: pd.DataFrame,
    ) -> None:
        mock_manager = _mock_surrogate_manager(["cost"])
        mock_train.return_value = mock_manager
        mock_suggest.return_value = _mock_candidates(1)

        result = surrox.suggest(problem, dataframe, n_suggestions=1)

        assert "cost" in result.surrogate_quality
        assert result.surrogate_quality["cost"] == 0.95

    @patch("surrox.suggest.suggest_candidates")
    @patch("surrox.suggest.SurrogateManager.train")
    def test_passes_configs(
        self,
        mock_train: MagicMock,
        mock_suggest: MagicMock,
        problem: ProblemDefinition,
        dataframe: pd.DataFrame,
    ) -> None:
        mock_manager = _mock_surrogate_manager(["cost"])
        mock_train.return_value = mock_manager
        mock_suggest.return_value = _mock_candidates(1)

        surrogate_config = TrainingConfig(n_trials=10)
        optimizer_config = OptimizerConfig(n_generations=50)

        surrox.suggest(
            problem, dataframe, n_suggestions=1,
            surrogate_config=surrogate_config,
            optimizer_config=optimizer_config,
        )

        mock_train.assert_called_once()
        assert mock_train.call_args.kwargs["config"] == surrogate_config
        mock_suggest.assert_called_once()
        assert mock_suggest.call_args.kwargs["config"] == optimizer_config
        assert mock_suggest.call_args.kwargs["n_candidates"] == 1

    @patch("surrox.suggest.suggest_candidates")
    @patch("surrox.suggest.SurrogateManager.train")
    def test_empty_candidates_returns_empty_suggestions(
        self,
        mock_train: MagicMock,
        mock_suggest: MagicMock,
        problem: ProblemDefinition,
        dataframe: pd.DataFrame,
    ) -> None:
        mock_manager = _mock_surrogate_manager(["cost"])
        mock_train.return_value = mock_manager
        mock_suggest.return_value = ()

        result = surrox.suggest(problem, dataframe, n_suggestions=5)

        assert len(result.suggestions) == 0

    @patch("surrox.suggest.suggest_candidates")
    @patch("surrox.suggest.SurrogateManager.train")
    def test_variables_preserved(
        self,
        mock_train: MagicMock,
        mock_suggest: MagicMock,
        problem: ProblemDefinition,
        dataframe: pd.DataFrame,
    ) -> None:
        mock_manager = _mock_surrogate_manager(["cost"])
        mock_train.return_value = mock_manager
        mock_suggest.return_value = _mock_candidates(1)

        result = surrox.suggest(problem, dataframe, n_suggestions=1)

        s = result.suggestions[0]
        assert "x1" in s.variables
        assert "x2" in s.variables
        assert s.variables["x1"] == 0.0
        assert s.variables["x2"] == 1.0

    @patch("surrox.suggest.suggest_candidates")
    @patch("surrox.suggest.SurrogateManager.train")
    def test_result_is_frozen(
        self,
        mock_train: MagicMock,
        mock_suggest: MagicMock,
        problem: ProblemDefinition,
        dataframe: pd.DataFrame,
    ) -> None:
        mock_manager = _mock_surrogate_manager(["cost"])
        mock_train.return_value = mock_manager
        mock_suggest.return_value = _mock_candidates(1)

        result = surrox.suggest(problem, dataframe, n_suggestions=1)

        with pytest.raises(ValidationError):
            result.suggestions = ()  # type: ignore[misc]


class TestSuggestValidation:
    def test_n_suggestions_zero_raises(
        self,
        problem: ProblemDefinition,
        dataframe: pd.DataFrame,
    ) -> None:
        with pytest.raises(ConfigurationError, match="n_suggestions must be >= 1"):
            surrox.suggest(problem, dataframe, n_suggestions=0)

    def test_n_suggestions_negative_raises(
        self,
        problem: ProblemDefinition,
        dataframe: pd.DataFrame,
    ) -> None:
        with pytest.raises(ConfigurationError, match="n_suggestions must be >= 1"):
            surrox.suggest(problem, dataframe, n_suggestions=-1)

    def test_coverage_zero_raises(
        self,
        problem: ProblemDefinition,
        dataframe: pd.DataFrame,
    ) -> None:
        with pytest.raises(ConfigurationError, match="coverage must be between"):
            surrox.suggest(problem, dataframe, coverage=0.0)

    def test_coverage_one_raises(
        self,
        problem: ProblemDefinition,
        dataframe: pd.DataFrame,
    ) -> None:
        with pytest.raises(ConfigurationError, match="coverage must be between"):
            surrox.suggest(problem, dataframe, coverage=1.0)

    def test_coverage_above_one_raises(
        self,
        problem: ProblemDefinition,
        dataframe: pd.DataFrame,
    ) -> None:
        with pytest.raises(ConfigurationError, match="coverage must be between"):
            surrox.suggest(problem, dataframe, coverage=1.5)

    def test_coverage_negative_raises(
        self,
        problem: ProblemDefinition,
        dataframe: pd.DataFrame,
    ) -> None:
        with pytest.raises(ConfigurationError, match="coverage must be between"):
            surrox.suggest(problem, dataframe, coverage=-0.1)


class TestSuggestScenario:
    @patch("surrox.suggest.suggest_candidates")
    @patch("surrox.suggest.SurrogateManager.train")
    def test_scenario_passed_to_suggest_candidates(
        self,
        mock_train: MagicMock,
        mock_suggest: MagicMock,
        problem_with_context: ProblemDefinition,
    ) -> None:
        rng = np.random.default_rng(42)
        n = 50
        df = pd.DataFrame({
            "x1": rng.uniform(0, 10, n),
            "location": rng.uniform(0, 5, n),
            "cost": rng.uniform(20, 80, n),
        })

        mock_manager = _mock_surrogate_manager(["cost"])
        mock_train.return_value = mock_manager
        mock_suggest.return_value = _mock_candidates(1)

        scenario = Scenario(name="north", context_values={"location": 3.0})

        surrox.suggest(
            problem_with_context, df, n_suggestions=1, scenario=scenario,
        )

        mock_suggest.assert_called_once()
        assert mock_suggest.call_args.kwargs["scenario"] == scenario

    @patch("surrox.suggest.suggest_candidates")
    @patch("surrox.suggest.SurrogateManager.train")
    def test_no_scenario_passes_none(
        self,
        mock_train: MagicMock,
        mock_suggest: MagicMock,
        problem: ProblemDefinition,
        dataframe: pd.DataFrame,
    ) -> None:
        mock_manager = _mock_surrogate_manager(["cost"])
        mock_train.return_value = mock_manager
        mock_suggest.return_value = _mock_candidates(1)

        surrox.suggest(problem, dataframe, n_suggestions=1)

        mock_suggest.assert_called_once()
        assert mock_suggest.call_args.kwargs["scenario"] is None
