from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from surrox.optimizer.config import OptimizerConfig
from surrox.optimizer.runner import optimize
from surrox.problem.dataset import BoundDataset
from surrox.problem.definition import ProblemDefinition
from surrox.problem.objectives import Objective
from surrox.problem.types import Direction, DType, Role
from surrox.problem.variables import ContinuousBounds, Variable
from surrox.surrogate.models import SurrogatePrediction


def _make_simple_setup() -> tuple[BoundDataset, MagicMock, OptimizerConfig]:
    problem = ProblemDefinition(
        variables=(
            Variable(
                name="x1", dtype=DType.CONTINUOUS, role=Role.DECISION,
                bounds=ContinuousBounds(lower=0.0, upper=10.0),
            ),
        ),
        objectives=(
            Objective(name="obj", direction=Direction.MINIMIZE, column="y"),
        ),
    )

    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({"x1": rng.uniform(0, 10, n), "y": rng.uniform(0, 100, n)})
    dataset = BoundDataset(problem=problem, dataframe=df)

    surrogate = MagicMock()
    surrogate.evaluate.side_effect = lambda df: {
        "y": df["x1"].to_numpy() ** 2
    }
    surrogate.get_ensemble_r2.return_value = 0.0

    config = OptimizerConfig(population_size=10, n_generations=3, seed=42, acquisition="direct")

    return dataset, surrogate, config


class TestOptimizeRunner:
    def test_returns_optimization_result(self) -> None:
        dataset, surrogate, config = _make_simple_setup()
        result = optimize(dataset, surrogate, config)
        assert result.n_generations == 3
        assert result.n_evaluations > 0
        assert result.problem == dataset.problem

    def test_has_feasible_solutions(self) -> None:
        dataset, surrogate, config = _make_simple_setup()
        result = optimize(dataset, surrogate, config)
        assert result.has_feasible_solutions
        assert len(result.feasible_points) > 0

    def test_feasible_points_have_objectives(self) -> None:
        dataset, surrogate, config = _make_simple_setup()
        result = optimize(dataset, surrogate, config)
        for point in result.feasible_points:
            assert "obj" in point.objectives

    def test_feasible_points_have_variables(self) -> None:
        dataset, surrogate, config = _make_simple_setup()
        result = optimize(dataset, surrogate, config)
        for point in result.feasible_points:
            assert "x1" in point.variables

    def test_single_objective_no_compromise(self) -> None:
        dataset, surrogate, config = _make_simple_setup()
        result = optimize(dataset, surrogate, config)
        assert result.compromise_index is None

    def test_single_objective_no_hypervolume(self) -> None:
        dataset, surrogate, config = _make_simple_setup()
        result = optimize(dataset, surrogate, config)
        assert result.hypervolume is None
