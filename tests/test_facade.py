from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import surrox
from surrox import (
    AnalysisConfig,
    AnalysisResult,
    Analyzer,
    OptimizerConfig,
    ProblemDefinition,
    ScenariosResult,
    SurroxResult,
    TrainingConfig,
)
from surrox.analysis.scenario import ScenarioComparisonResult
from surrox.exceptions import SurroxError
from surrox.optimizer.result import (
    ConstraintEvaluation,
    EvaluatedPoint,
    OptimizationResult,
)
from surrox.problem.constraints import DataConstraint
from surrox.problem.objectives import Objective
from surrox.problem.scenarios import Scenario
from surrox.problem.types import (
    ConstraintOperator,
    ConstraintSeverity,
    Direction,
    DType,
    Role,
)
from surrox.problem.variables import ContinuousBounds, Variable
from surrox.surrogate.models import FoldMetrics, SurrogatePrediction, TrialRecord


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
        data_constraints=(
            DataConstraint(
                name="emission_cap", column="emissions",
                operator=ConstraintOperator.LE, limit=100.0,
            ),
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
        "emissions": rng.uniform(50, 150, n),
    })


def _make_mock_optimization_result(
    problem: ProblemDefinition,
    scenario: Scenario | None = None,
) -> OptimizationResult:
    ce = ConstraintEvaluation(
        name="emission_cap", violation=0.0, prediction=85.0,
        severity=ConstraintSeverity.HARD,
    )
    feasible = tuple(
        EvaluatedPoint(
            variables={"x1": float(i), "x2": float(i + 1)},
            objectives={"cost": 50.0 - i * 5.0},
            constraints=(ce,),
            feasible=True,
            extrapolation_distance=0.1,
            is_extrapolating=False,
        )
        for i in range(5)
    )
    return OptimizationResult(
        feasible_points=feasible,
        infeasible_points=(),
        has_feasible_solutions=True,
        compromise_index=None,
        hypervolume=None,
        problem=problem,
        n_generations=10,
        n_evaluations=100,
    )


def _make_mock_surrogate_manager(columns: list[str]) -> MagicMock:
    manager = MagicMock()
    predict_values = {c: 50.0 for c in columns}

    manager.evaluate.side_effect = lambda df: {
        col: np.full(len(df), predict_values[col]) for col in columns
    }
    manager.evaluate_with_uncertainty.side_effect = lambda df, coverage=None: {
        col: SurrogatePrediction(
            mean=np.full(len(df), predict_values[col]),
            std=np.full(len(df), 5.0),
            lower=np.full(len(df), predict_values[col] - 10.0),
            upper=np.full(len(df), predict_values[col] + 10.0),
        )
        for col in columns
    }

    for col in columns:
        ensemble_mock = MagicMock()
        ensemble_mock.members = ()
        ensemble_mock.feature_names = ("x1", "x2")
        ensemble_mock.predict.side_effect = lambda df, _c=col: np.full(
            len(df), predict_values[_c]
        )

        sr_mock = MagicMock()
        sr_mock.ensemble = ensemble_mock
        sr_mock.trial_history = (
            TrialRecord(
                trial_number=0, estimator_family="xgboost",
                hyperparameters={"n_estimators": 100},
                fold_metrics=(
                    FoldMetrics(fold=0, r2=0.9, rmse=5.0, mae=3.0, training_time_s=1.0, inference_time_ms=0.5),
                ),
                mean_r2=0.9, mean_rmse=5.0, mean_mae=3.0,
                mean_training_time_s=1.0, mean_inference_time_ms=0.5, status="COMPLETE",
            ),
        )
        conformal_mock = MagicMock()
        conformal_mock.X_calib = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0]})
        conformal_mock.y_calib = np.array([40.0, 50.0, 60.0])
        conformal_mock._default_coverage = 0.9
        conformal_mock.prediction_interval.return_value = (
            np.array([40.0, 50.0, 60.0]),
            np.array([35.0, 45.0, 55.0]),
            np.array([45.0, 55.0, 65.0]),
        )
        sr_mock.conformal = conformal_mock

        manager.get_ensemble.side_effect = (
            lambda c, _cols={col: ensemble_mock for col in columns}: _cols[c]
        )
        manager.get_surrogate_result.side_effect = (
            lambda c, _cols={col: sr_mock for col in columns}: _cols[c]
        )

    return manager


class TestRun:
    @patch("surrox.optimize")
    @patch("surrox.SurrogateManager.train")
    def test_returns_correct_types(
        self,
        mock_train: MagicMock,
        mock_optimize: MagicMock,
        problem: ProblemDefinition,
        dataframe: pd.DataFrame,
    ) -> None:
        mock_manager = _make_mock_surrogate_manager(["cost", "emissions"])
        mock_train.return_value = mock_manager
        mock_optimize.return_value = _make_mock_optimization_result(problem)

        result, analyzer = surrox.run(problem, dataframe)

        assert isinstance(result, SurroxResult)
        assert isinstance(result.optimization, OptimizationResult)
        assert isinstance(result.analysis, AnalysisResult)
        assert isinstance(analyzer, Analyzer)

    @patch("surrox.optimize")
    @patch("surrox.SurrogateManager.train")
    def test_passes_configs(
        self,
        mock_train: MagicMock,
        mock_optimize: MagicMock,
        problem: ProblemDefinition,
        dataframe: pd.DataFrame,
    ) -> None:
        mock_manager = _make_mock_surrogate_manager(["cost", "emissions"])
        mock_train.return_value = mock_manager
        mock_optimize.return_value = _make_mock_optimization_result(problem)

        surrogate_config = TrainingConfig(n_trials=10)
        optimizer_config = OptimizerConfig(n_generations=50)
        analysis_config = AnalysisConfig(shap_background_size=20)

        surrox.run(
            problem, dataframe,
            surrogate_config=surrogate_config,
            optimizer_config=optimizer_config,
            analysis_config=analysis_config,
        )

        mock_train.assert_called_once()
        assert mock_train.call_args.kwargs["config"] == surrogate_config
        mock_optimize.assert_called_once()
        assert mock_optimize.call_args.kwargs["config"] == optimizer_config

    @patch("surrox.optimize")
    @patch("surrox.SurrogateManager.train")
    def test_passes_scenario(
        self,
        mock_train: MagicMock,
        mock_optimize: MagicMock,
        problem: ProblemDefinition,
        dataframe: pd.DataFrame,
    ) -> None:
        mock_manager = _make_mock_surrogate_manager(["cost", "emissions"])
        mock_train.return_value = mock_manager
        mock_optimize.return_value = _make_mock_optimization_result(problem)

        scenario = Scenario(name="summer", context_values={"season": "summer"})

        surrox.run(problem, dataframe, scenario=scenario)

        mock_optimize.assert_called_once()
        assert mock_optimize.call_args.kwargs["scenario"] == scenario

    @patch("surrox.optimize")
    @patch("surrox.SurrogateManager.train")
    def test_result_is_frozen(
        self,
        mock_train: MagicMock,
        mock_optimize: MagicMock,
        problem: ProblemDefinition,
        dataframe: pd.DataFrame,
    ) -> None:
        mock_manager = _make_mock_surrogate_manager(["cost", "emissions"])
        mock_train.return_value = mock_manager
        mock_optimize.return_value = _make_mock_optimization_result(problem)

        result, _ = surrox.run(problem, dataframe)

        with pytest.raises(Exception):
            result.optimization = None  # type: ignore[assignment]


class TestRunScenarios:
    @patch("surrox.optimize")
    @patch("surrox.SurrogateManager.train")
    def test_returns_correct_types(
        self,
        mock_train: MagicMock,
        mock_optimize: MagicMock,
        problem: ProblemDefinition,
        dataframe: pd.DataFrame,
    ) -> None:
        mock_manager = _make_mock_surrogate_manager(["cost", "emissions"])
        mock_train.return_value = mock_manager
        mock_optimize.return_value = _make_mock_optimization_result(problem)

        scenarios = {
            "summer": Scenario(name="summer", context_values={"season": "summer"}),
            "winter": Scenario(name="winter", context_values={"season": "winter"}),
        }

        result, analyzers = surrox.run_scenarios(problem, dataframe, scenarios)

        assert isinstance(result, ScenariosResult)
        assert isinstance(result.comparison, ScenarioComparisonResult)
        assert set(result.per_scenario.keys()) == {"summer", "winter"}
        assert all(isinstance(r, SurroxResult) for r in result.per_scenario.values())
        assert set(analyzers.keys()) == {"summer", "winter"}
        assert all(isinstance(a, Analyzer) for a in analyzers.values())

    @patch("surrox.optimize")
    @patch("surrox.SurrogateManager.train")
    def test_trains_surrogates_once(
        self,
        mock_train: MagicMock,
        mock_optimize: MagicMock,
        problem: ProblemDefinition,
        dataframe: pd.DataFrame,
    ) -> None:
        mock_manager = _make_mock_surrogate_manager(["cost", "emissions"])
        mock_train.return_value = mock_manager
        mock_optimize.return_value = _make_mock_optimization_result(problem)

        scenarios = {
            "summer": Scenario(name="summer", context_values={"season": "summer"}),
            "winter": Scenario(name="winter", context_values={"season": "winter"}),
        }

        surrox.run_scenarios(problem, dataframe, scenarios)

        mock_train.assert_called_once()
        assert mock_optimize.call_count == 2

    def test_less_than_two_scenarios_raises(
        self,
        problem: ProblemDefinition,
        dataframe: pd.DataFrame,
    ) -> None:
        with pytest.raises(SurroxError, match="at least 2 scenarios"):
            surrox.run_scenarios(
                problem, dataframe,
                scenarios={"only_one": Scenario(name="only_one", context_values={"season": "x"})},
            )
