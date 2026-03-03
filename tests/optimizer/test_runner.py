from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from surrox.optimizer.config import OptimizerConfig
from surrox.optimizer.runner import _select_diverse, optimize, suggest_candidates
from surrox.problem.dataset import BoundDataset
from surrox.problem.definition import ProblemDefinition
from surrox.problem.objectives import Objective
from surrox.problem.scenarios import Scenario
from surrox.problem.types import Direction, DType, Role
from surrox.problem.variables import ContinuousBounds, Variable


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

    config = OptimizerConfig(
        population_size=10, n_generations=3, seed=42, acquisition="direct",
    )

    return dataset, surrogate, config


def _make_context_setup() -> tuple[BoundDataset, MagicMock, OptimizerConfig]:
    problem = ProblemDefinition(
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
            Objective(name="obj", direction=Direction.MINIMIZE, column="y"),
        ),
    )

    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "x1": rng.uniform(0, 10, n),
        "location": rng.uniform(0, 5, n),
        "y": rng.uniform(0, 100, n),
    })
    dataset = BoundDataset(problem=problem, dataframe=df)

    surrogate = MagicMock()
    surrogate.evaluate.side_effect = lambda df: {
        "y": df["x1"].to_numpy() ** 2
    }
    surrogate.get_ensemble_r2.return_value = 0.0

    config = OptimizerConfig(
        population_size=10, n_generations=3, seed=42, acquisition="direct",
    )

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


class TestSuggestCandidates:
    def test_returns_evaluated_points(self) -> None:
        dataset, surrogate, config = _make_simple_setup()
        candidates = suggest_candidates(
            dataset, surrogate, n_candidates=3, config=config,
        )
        assert len(candidates) > 0
        assert len(candidates) <= 3

    def test_candidates_have_variables_and_objectives(self) -> None:
        dataset, surrogate, config = _make_simple_setup()
        candidates = suggest_candidates(
            dataset, surrogate, n_candidates=3, config=config,
        )
        for c in candidates:
            assert "x1" in c.variables
            assert "obj" in c.objectives

    def test_candidates_have_extrapolation_info(self) -> None:
        dataset, surrogate, config = _make_simple_setup()
        candidates = suggest_candidates(
            dataset, surrogate, n_candidates=3, config=config,
        )
        for c in candidates:
            assert isinstance(c.extrapolation_distance, float)
            assert isinstance(c.is_extrapolating, bool)

    def test_candidates_are_feasible(self) -> None:
        dataset, surrogate, config = _make_simple_setup()
        candidates = suggest_candidates(
            dataset, surrogate, n_candidates=3, config=config,
        )
        for c in candidates:
            assert c.feasible is True

    def test_scenario_context_values_included(self) -> None:
        dataset, surrogate, config = _make_context_setup()
        scenario = Scenario(
            name="test",
            context_values={"location": 2.5},
        )
        candidates = suggest_candidates(
            dataset, surrogate, n_candidates=3, config=config, scenario=scenario,
        )
        for c in candidates:
            assert c.variables["location"] == 2.5

    def test_multiple_candidates_are_diverse(self) -> None:
        dataset, surrogate, config = _make_simple_setup()
        candidates = suggest_candidates(
            dataset, surrogate, n_candidates=5, config=config,
        )
        if len(candidates) >= 2:
            x1_values = [c.variables["x1"] for c in candidates]
            assert len(set(x1_values)) > 1


class TestSelectDiverse:
    def test_returns_all_when_fewer_than_n(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        sorted_indices = np.array([0, 1])
        result = _select_diverse(X, sorted_indices, n=5)
        assert result == [0, 1]

    def test_selects_n_points(self) -> None:
        X = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ])
        sorted_indices = np.arange(5)
        result = _select_diverse(X, sorted_indices, n=3)
        assert len(result) == 3

    def test_first_selected_is_best_scored(self) -> None:
        X = np.array([
            [0.0, 0.0],
            [5.0, 5.0],
            [10.0, 10.0],
        ])
        sorted_indices = np.array([2, 0, 1])
        result = _select_diverse(X, sorted_indices, n=2)
        assert result[0] == 2

    def test_diverse_points_are_spread(self) -> None:
        X = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [0.2, 0.2],
        ])
        sorted_indices = np.arange(4)
        result = _select_diverse(X, sorted_indices, n=2)
        assert 0 in result
        assert 2 in result

    def test_single_candidate_returns_first(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        sorted_indices = np.array([0, 1, 2])
        result = _select_diverse(X, sorted_indices, n=1)
        assert result == [0]

    def test_empty_sorted_indices(self) -> None:
        X = np.array([[1.0, 2.0]])
        sorted_indices = np.array([], dtype=int)
        result = _select_diverse(X, sorted_indices, n=3)
        assert result == []
