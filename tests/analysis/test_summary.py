from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from surrox.analysis.config import AnalysisConfig
from surrox.analysis.summary import (
    BaselineComparison,
    ConstraintStatus,
    ExtrapolationWarning,
    MonotonicityViolation,
    SolutionSummary,
    Summary,
    SurrogateQuality,
    compute_summary,
    _compute_baseline_comparison,
    _compute_constraint_status,
    _compute_extrapolation_warnings,
    _compute_solution_summary,
    _classify_constraint_status,
    _compute_margin,
    _get_recommended_solution,
)
from surrox.analysis.types import ConstraintStatusKind
from surrox.optimizer.result import (
    ConstraintEvaluation,
    EvaluatedPoint,
    OptimizationResult,
)
from surrox.problem.types import (
    ConstraintOperator,
    ConstraintSeverity,
    Direction,
    MonotonicDirection,
)


class TestSolutionSummary:
    def test_single_objective_counts(
        self, single_objective_result: OptimizationResult
    ) -> None:
        summary = _compute_solution_summary(single_objective_result)
        assert summary.n_feasible == 5
        assert summary.n_infeasible == 1

    def test_single_objective_best(
        self, single_objective_result: OptimizationResult
    ) -> None:
        summary = _compute_solution_summary(single_objective_result)
        assert summary.best_objectives["cost"] == 30.0

    def test_multi_objective_compromise(
        self, multi_objective_result: OptimizationResult
    ) -> None:
        summary = _compute_solution_summary(multi_objective_result)
        assert summary.compromise_objectives is not None
        assert summary.compromise_objectives["cost"] == 50.0
        assert summary.compromise_objectives["quality"] == 90.0

    def test_multi_objective_hypervolume(
        self, multi_objective_result: OptimizationResult
    ) -> None:
        summary = _compute_solution_summary(multi_objective_result)
        assert summary.hypervolume == 1500.0

    def test_no_feasible(
        self, no_feasible_result: OptimizationResult
    ) -> None:
        summary = _compute_solution_summary(no_feasible_result)
        assert summary.n_feasible == 0
        assert summary.best_objectives == {}
        assert summary.compromise_objectives is None


class TestBaselineComparison:
    def test_minimize_improvement(
        self, single_objective_result: OptimizationResult,
        bound_dataset_single,
    ) -> None:
        recommended = _get_recommended_solution(single_objective_result)
        comparison = _compute_baseline_comparison(
            single_objective_result, bound_dataset_single, recommended
        )
        assert comparison is not None
        assert "cost" in comparison.recommended_objectives
        assert "cost" in comparison.historical_best_per_objective
        assert "cost" in comparison.improvement

    def test_no_feasible_returns_none(
        self, no_feasible_result: OptimizationResult,
        bound_dataset_single,
    ) -> None:
        comparison = _compute_baseline_comparison(
            no_feasible_result, bound_dataset_single, None
        )
        assert comparison is None


class TestConstraintStatus:
    def test_satisfied(self) -> None:
        status = _classify_constraint_status(
            violation=-10.0, margin=10.0, limit=100.0
        )
        assert status == ConstraintStatusKind.SATISFIED

    def test_active(self) -> None:
        status = _classify_constraint_status(
            violation=-0.1, margin=0.1, limit=100.0
        )
        assert status == ConstraintStatusKind.ACTIVE

    def test_violated(self) -> None:
        status = _classify_constraint_status(
            violation=5.0, margin=-5.0, limit=100.0
        )
        assert status == ConstraintStatusKind.VIOLATED

    def test_limit_zero_no_division_error(self) -> None:
        status = _classify_constraint_status(
            violation=-0.001, margin=0.001, limit=0.0
        )
        assert status in (
            ConstraintStatusKind.SATISFIED,
            ConstraintStatusKind.ACTIVE,
        )

    def test_margin_le(self) -> None:
        margin = _compute_margin(90.0, 100.0, ConstraintOperator.LE)
        assert margin == 10.0

    def test_margin_ge(self) -> None:
        margin = _compute_margin(90.0, 50.0, ConstraintOperator.GE)
        assert margin == 40.0

    def test_margin_eq(self) -> None:
        margin = _compute_margin(25.5, 25.0, ConstraintOperator.EQ)
        assert margin == -0.5

    def test_constraint_status_wraps_evaluation(
        self, single_objective_result: OptimizationResult,
    ) -> None:
        problem = single_objective_result.problem
        recommended = _get_recommended_solution(single_objective_result)
        statuses = _compute_constraint_status(problem, recommended)
        assert len(statuses) == 1
        assert statuses[0].evaluation.name == "emission_cap"
        assert statuses[0].status == ConstraintStatusKind.SATISFIED


class TestExtrapolationWarnings:
    def test_no_extrapolating_points(
        self, single_objective_result: OptimizationResult,
    ) -> None:
        warnings = _compute_extrapolation_warnings(single_objective_result)
        assert len(warnings) == 0

    def test_extrapolating_point_detected(
        self, single_objective_problem,
    ) -> None:
        from tests.analysis.conftest import _make_evaluated_point

        feasible = (
            _make_evaluated_point(
                variables={"x1": 1.0, "x2": 2.0},
                objectives={"cost": 50.0},
                is_extrapolating=True,
                extrapolation_distance=2.5,
            ),
        )
        result = OptimizationResult(
            feasible_points=feasible,
            infeasible_points=(),
            has_feasible_solutions=True,
            compromise_index=None,
            hypervolume=None,
            problem=single_objective_problem,
            n_generations=1,
            n_evaluations=1,
        )
        warnings = _compute_extrapolation_warnings(result)
        assert len(warnings) == 1
        assert warnings[0].point_index == 0
        assert warnings[0].distance == 2.5


class TestSurrogateQuality:
    def test_quality_computed(
        self, single_objective_problem, mock_surrogate_single,
    ) -> None:
        from surrox.analysis.summary import _compute_surrogate_quality

        qualities = _compute_surrogate_quality(
            mock_surrogate_single, single_objective_problem
        )
        assert len(qualities) == 2
        assert qualities[0].column in ("cost", "emissions")
        assert qualities[0].ensemble_size >= 0
        assert 0.0 <= qualities[0].conformal_coverage <= 1.0


class TestComputeSummary:
    def test_full_summary_single_objective(
        self,
        single_objective_result,
        mock_surrogate_single,
        bound_dataset_single,
        analysis_config,
    ) -> None:
        summary = compute_summary(
            single_objective_result,
            mock_surrogate_single,
            bound_dataset_single,
            analysis_config,
        )
        assert isinstance(summary, Summary)
        assert summary.solution_summary.n_feasible == 5
        assert summary.baseline_comparison is not None

    def test_summary_no_feasible(
        self,
        no_feasible_result,
        mock_surrogate_single,
        bound_dataset_single,
        analysis_config,
    ) -> None:
        summary = compute_summary(
            no_feasible_result,
            mock_surrogate_single,
            bound_dataset_single,
            analysis_config,
        )
        assert summary.solution_summary.n_feasible == 0
        assert summary.baseline_comparison is None
        assert summary.constraint_status == ()
        assert summary.monotonicity_violations == ()
