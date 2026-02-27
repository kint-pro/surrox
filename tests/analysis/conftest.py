from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from surrox.analysis.config import AnalysisConfig
from surrox.optimizer.result import (
    ConstraintEvaluation,
    EvaluatedPoint,
    OptimizationResult,
)
from surrox.problem.constraints import DataConstraint
from surrox.problem.definition import ProblemDefinition
from surrox.problem.dataset import BoundDataset
from surrox.problem.objectives import Objective
from surrox.problem.types import (
    ConstraintOperator,
    ConstraintSeverity,
    Direction,
    DType,
    MonotonicDirection,
    Role,
)
from surrox.problem.domain_knowledge import MonotonicRelation
from surrox.problem.variables import ContinuousBounds, Variable
from surrox.surrogate.models import SurrogatePrediction, TrialRecord, FoldMetrics


def _make_fold_metrics(rmse: float = 5.0) -> tuple[FoldMetrics, ...]:
    return (
        FoldMetrics(
            fold=0, r2=0.9, rmse=rmse, mae=3.0,
            training_time_s=1.0, inference_time_ms=0.5,
        ),
        FoldMetrics(
            fold=1, r2=0.88, rmse=rmse + 0.5, mae=3.5,
            training_time_s=1.1, inference_time_ms=0.6,
        ),
    )


def _make_trial_record(
    trial_number: int = 0,
    mean_rmse: float = 5.0,
) -> TrialRecord:
    folds = _make_fold_metrics(mean_rmse)
    return TrialRecord(
        trial_number=trial_number,
        estimator_family="xgboost",
        hyperparameters={"n_estimators": 100},
        fold_metrics=folds,
        mean_r2=0.89,
        mean_rmse=mean_rmse,
        mean_mae=3.25,
        mean_training_time_s=1.05,
        mean_inference_time_ms=0.55,
        status="COMPLETE",
    )


def _make_analysis_surrogate_manager(
    columns: list[str],
    predict_fn: dict[str, float] | None = None,
) -> MagicMock:
    predict_values = predict_fn or {c: 50.0 for c in columns}

    manager = MagicMock()

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
        sr_mock.trial_history = (_make_trial_record(),)

        conformal_mock = MagicMock()
        conformal_mock.X_calib = pd.DataFrame(
            {"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0]}
        )
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
        manager.get_trial_history.side_effect = (
            lambda c, _cols={
                col: (_make_trial_record(),) for col in columns
            }: _cols[c]
        )
        manager.get_surrogate_result.side_effect = (
            lambda c, _cols={col: sr_mock for col in columns}: _cols[c]
        )

    return manager


@pytest.fixture
def single_objective_problem() -> ProblemDefinition:
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
def multi_objective_problem() -> ProblemDefinition:
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
            Objective(
                name="quality", direction=Direction.MAXIMIZE, column="quality"
            ),
        ),
    )


@pytest.fixture
def monotonic_problem() -> ProblemDefinition:
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
        monotonic_relations=(
            MonotonicRelation(
                decision_variable="x1",
                objective_or_constraint="cost",
                direction=MonotonicDirection.INCREASING,
            ),
        ),
    )


def _make_evaluated_point(
    variables: dict[str, float],
    objectives: dict[str, float],
    constraints: tuple[ConstraintEvaluation, ...] = (),
    feasible: bool = True,
    extrapolation_distance: float = 0.1,
    is_extrapolating: bool = False,
) -> EvaluatedPoint:
    return EvaluatedPoint(
        variables=variables,
        objectives=objectives,
        constraints=constraints,
        feasible=feasible,
        extrapolation_distance=extrapolation_distance,
        is_extrapolating=is_extrapolating,
    )


@pytest.fixture
def single_objective_result(
    single_objective_problem: ProblemDefinition,
) -> OptimizationResult:
    ce = ConstraintEvaluation(
        name="emission_cap", violation=-10.0, prediction=90.0,
        severity=ConstraintSeverity.HARD,
    )
    feasible = tuple(
        _make_evaluated_point(
            variables={"x1": float(i), "x2": float(i + 1)},
            objectives={"cost": 50.0 - i * 5.0},
            constraints=(ce,),
        )
        for i in range(5)
    )
    infeasible = (
        _make_evaluated_point(
            variables={"x1": 8.0, "x2": 9.0},
            objectives={"cost": 20.0},
            constraints=(
                ConstraintEvaluation(
                    name="emission_cap", violation=10.0, prediction=110.0,
                    severity=ConstraintSeverity.HARD,
                ),
            ),
            feasible=False,
        ),
    )
    return OptimizationResult(
        feasible_points=feasible,
        infeasible_points=infeasible,
        has_feasible_solutions=True,
        compromise_index=None,
        hypervolume=None,
        problem=single_objective_problem,
        n_generations=10,
        n_evaluations=100,
    )


@pytest.fixture
def multi_objective_result(
    multi_objective_problem: ProblemDefinition,
) -> OptimizationResult:
    feasible = (
        _make_evaluated_point(
            variables={"x1": 2.0, "x2": 3.0},
            objectives={"cost": 30.0, "quality": 80.0},
        ),
        _make_evaluated_point(
            variables={"x1": 5.0, "x2": 5.0},
            objectives={"cost": 50.0, "quality": 90.0},
        ),
        _make_evaluated_point(
            variables={"x1": 8.0, "x2": 7.0},
            objectives={"cost": 70.0, "quality": 95.0},
        ),
    )
    return OptimizationResult(
        feasible_points=feasible,
        infeasible_points=(),
        has_feasible_solutions=True,
        compromise_index=1,
        hypervolume=1500.0,
        problem=multi_objective_problem,
        n_generations=20,
        n_evaluations=200,
    )


@pytest.fixture
def no_feasible_result(
    single_objective_problem: ProblemDefinition,
) -> OptimizationResult:
    infeasible = (
        _make_evaluated_point(
            variables={"x1": 8.0, "x2": 9.0},
            objectives={"cost": 20.0},
            feasible=False,
        ),
    )
    return OptimizationResult(
        feasible_points=(),
        infeasible_points=infeasible,
        has_feasible_solutions=False,
        compromise_index=None,
        hypervolume=None,
        problem=single_objective_problem,
        n_generations=10,
        n_evaluations=100,
    )


@pytest.fixture
def training_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame({
        "x1": rng.uniform(0, 10, n),
        "x2": rng.uniform(0, 10, n),
        "cost": rng.uniform(20, 80, n),
        "quality": rng.uniform(60, 100, n),
        "emissions": rng.uniform(50, 150, n),
    })


@pytest.fixture
def bound_dataset_single(
    single_objective_problem: ProblemDefinition,
    training_df: pd.DataFrame,
) -> BoundDataset:
    return BoundDataset(
        problem=single_objective_problem,
        dataframe=training_df[["x1", "x2", "cost", "emissions"]],
    )


@pytest.fixture
def bound_dataset_multi(
    multi_objective_problem: ProblemDefinition,
    training_df: pd.DataFrame,
) -> BoundDataset:
    return BoundDataset(
        problem=multi_objective_problem,
        dataframe=training_df[["x1", "x2", "cost", "quality"]],
    )


@pytest.fixture
def mock_surrogate_single() -> MagicMock:
    return _make_analysis_surrogate_manager(
        ["cost", "emissions"],
        predict_fn={"cost": 45.0, "emissions": 85.0},
    )


@pytest.fixture
def mock_surrogate_multi() -> MagicMock:
    return _make_analysis_surrogate_manager(
        ["cost", "quality"],
        predict_fn={"cost": 50.0, "quality": 85.0},
    )


@pytest.fixture
def analysis_config() -> AnalysisConfig:
    return AnalysisConfig()
