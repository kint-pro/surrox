from unittest.mock import MagicMock

import numpy as np
import pytest

from surrox.analysis.config import AnalysisConfig
from surrox.analysis.summary import _compute_monotonicity_violations
from surrox.optimizer.result import OptimizationResult
from surrox.problem.types import MonotonicDirection

from tests.analysis.conftest import (
    _make_evaluated_point,
    _make_analysis_surrogate_manager,
)


class TestMonotonicityViolation:
    def test_no_violations_when_monotonic(
        self, monotonic_problem,
    ) -> None:
        feasible = (
            _make_evaluated_point(
                variables={"x1": 5.0, "x2": 5.0},
                objectives={"cost": 50.0},
            ),
        )
        result = OptimizationResult(
            feasible_points=feasible,
            infeasible_points=(),
            has_feasible_solutions=True,
            compromise_index=None,
            hypervolume=None,
            problem=monotonic_problem,
            n_generations=1,
            n_evaluations=1,
        )

        manager = MagicMock()
        ensemble = MagicMock()
        ensemble.predict.side_effect = lambda df: np.linspace(
            10, 100, len(df)
        )
        manager.get_ensemble.return_value = ensemble

        violations = _compute_monotonicity_violations(
            result, manager, AnalysisConfig(), feasible[0]
        )
        assert len(violations) == 0

    def test_violations_detected_when_non_monotonic(
        self, monotonic_problem,
    ) -> None:
        feasible = (
            _make_evaluated_point(
                variables={"x1": 5.0, "x2": 5.0},
                objectives={"cost": 50.0},
            ),
        )
        result = OptimizationResult(
            feasible_points=feasible,
            infeasible_points=(),
            has_feasible_solutions=True,
            compromise_index=None,
            hypervolume=None,
            problem=monotonic_problem,
            n_generations=1,
            n_evaluations=1,
        )

        manager = MagicMock()
        ensemble = MagicMock()
        rng = np.random.default_rng(42)
        ensemble.predict.side_effect = lambda df: rng.uniform(0, 100, len(df))
        manager.get_ensemble.return_value = ensemble

        violations = _compute_monotonicity_violations(
            result, manager, AnalysisConfig(), feasible[0]
        )
        assert len(violations) == 1
        assert violations[0].decision_variable == "x1"
        assert violations[0].declared_direction == MonotonicDirection.INCREASING
        assert violations[0].violation_fraction > 0
