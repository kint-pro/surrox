from __future__ import annotations

import numpy as np
import pytest

import surrox
from surrox import AnalysisResult, Analyzer, SurroxResult
from surrox.analysis.summary import Summary
from surrox.optimizer.config import OptimizerConfig
from surrox.optimizer.result import OptimizationResult
from surrox.surrogate.config import TrainingConfig
from tests.integration.benchmarks import (
    BRANIN_KNOWN_MINIMUM,
    ROSENBROCK_KNOWN_MINIMUM,
    ROSENBROCK_KNOWN_OPTIMUM,
    generate_branin,
    generate_rosenbrock,
    generate_zdt1,
)

RunResult = tuple[SurroxResult, Analyzer]

SURROGATE_CONFIG = TrainingConfig(
    n_trials=10,
    cv_folds=3,
    ensemble_size=3,
    min_r2=None,
    random_seed=42,
)

OPTIMIZER_CONFIG = OptimizerConfig(
    population_size=50,
    n_generations=50,
    seed=42,
)


@pytest.mark.slow
class TestBraninEndToEnd:
    @pytest.fixture(scope="class")
    def branin_result(self) -> RunResult:
        problem, df = generate_branin(n_samples=600, seed=42)
        return surrox.run(
            problem, df,
            surrogate_config=SURROGATE_CONFIG,
            optimizer_config=OPTIMIZER_CONFIG,
        )

    def test_returns_correct_types(self, branin_result: RunResult) -> None:
        result, analyzer = branin_result
        assert isinstance(result, SurroxResult)
        assert isinstance(result.optimization, OptimizationResult)
        assert isinstance(result.analysis, AnalysisResult)
        assert isinstance(result.analysis.summary, Summary)
        assert isinstance(analyzer, Analyzer)

    def test_has_feasible_solutions(self, branin_result: RunResult) -> None:
        result, _ = branin_result
        assert result.optimization.has_feasible_solutions
        assert len(result.optimization.feasible_points) > 0

    def test_best_objective_near_known_minimum(
        self, branin_result: RunResult,
    ) -> None:
        result, _ = branin_result
        best = result.analysis.summary.solution_summary.best_objectives["branin"]
        assert best < BRANIN_KNOWN_MINIMUM * 10, (
            f"best={best}, expected near {BRANIN_KNOWN_MINIMUM}"
        )

    def test_summary_has_surrogate_quality(
        self, branin_result: RunResult,
    ) -> None:
        result, _ = branin_result
        qualities = result.analysis.summary.surrogate_quality
        assert len(qualities) > 0
        for q in qualities:
            assert q.cv_rmse >= 0
            assert q.ensemble_size >= 1

    def test_constraint_status_populated(
        self, branin_result: RunResult,
    ) -> None:
        result, _ = branin_result
        statuses = result.analysis.summary.constraint_status
        assert len(statuses) == 1
        assert statuses[0].evaluation.name == "sum_limit"

    def test_baseline_comparison_exists(
        self, branin_result: RunResult,
    ) -> None:
        result, _ = branin_result
        comparison = result.analysis.summary.baseline_comparison
        assert comparison is not None
        assert "branin" in comparison.recommended_objectives


@pytest.mark.slow
class TestRosenbrockEndToEnd:
    @pytest.fixture(scope="class")
    def rosenbrock_result(self) -> RunResult:
        problem, df = generate_rosenbrock(n_samples=600, seed=42)
        return surrox.run(
            problem, df,
            surrogate_config=SURROGATE_CONFIG,
            optimizer_config=OPTIMIZER_CONFIG,
        )

    def test_has_feasible_solutions(
        self, rosenbrock_result: RunResult,
    ) -> None:
        result, _ = rosenbrock_result
        assert result.optimization.has_feasible_solutions

    def test_best_objective_near_zero(
        self, rosenbrock_result: RunResult,
    ) -> None:
        result, _ = rosenbrock_result
        best = result.analysis.summary.solution_summary.best_objectives["rosenbrock"]
        assert best < 50.0, f"best={best}, expected near {ROSENBROCK_KNOWN_MINIMUM}"

    def test_best_point_near_known_optimum(
        self, rosenbrock_result: RunResult,
    ) -> None:
        result, _ = rosenbrock_result
        feasible = result.optimization.feasible_points
        best_point = min(feasible, key=lambda p: p.objectives["rosenbrock"])
        x1 = best_point.variables["x1"]
        x2 = best_point.variables["x2"]
        distance = np.sqrt(
            (x1 - ROSENBROCK_KNOWN_OPTIMUM[0]) ** 2
            + (x2 - ROSENBROCK_KNOWN_OPTIMUM[1]) ** 2
        )
        assert distance < 2.0, (
            f"best point ({x1:.2f}, {x2:.2f}) too far from known optimum "
            f"{ROSENBROCK_KNOWN_OPTIMUM}, distance={distance:.2f}"
        )

    def test_no_constraints(
        self, rosenbrock_result: RunResult,
    ) -> None:
        result, _ = rosenbrock_result
        assert len(result.analysis.summary.constraint_status) == 0


@pytest.mark.slow
class TestZDT1EndToEnd:
    @pytest.fixture(scope="class")
    def zdt1_result(self) -> RunResult:
        problem, df = generate_zdt1(n_variables=5, n_samples=600, seed=42)
        return surrox.run(
            problem, df,
            surrogate_config=SURROGATE_CONFIG,
            optimizer_config=OPTIMIZER_CONFIG,
        )

    def test_has_feasible_solutions(
        self, zdt1_result: RunResult,
    ) -> None:
        result, _ = zdt1_result
        assert result.optimization.has_feasible_solutions

    def test_pareto_front_has_multiple_points(
        self, zdt1_result: RunResult,
    ) -> None:
        result, _ = zdt1_result
        assert len(result.optimization.feasible_points) >= 5

    def test_compromise_index_set(
        self, zdt1_result: RunResult,
    ) -> None:
        result, _ = zdt1_result
        assert result.optimization.compromise_index is not None

    def test_hypervolume_computed(
        self, zdt1_result: RunResult,
    ) -> None:
        result, _ = zdt1_result
        assert result.optimization.hypervolume is not None
        assert result.optimization.hypervolume > 0

    def test_both_objectives_in_summary(
        self, zdt1_result: RunResult,
    ) -> None:
        result, _ = zdt1_result
        best_objs = result.analysis.summary.solution_summary.best_objectives
        assert "f1" in best_objs
        assert "f2" in best_objs

    def test_compromise_objectives_exist(
        self, zdt1_result: RunResult,
    ) -> None:
        result, _ = zdt1_result
        compromise = result.analysis.summary.solution_summary.compromise_objectives
        assert compromise is not None
        assert "f1" in compromise
        assert "f2" in compromise

    def test_pareto_front_dominance(
        self, zdt1_result: RunResult,
    ) -> None:
        result, _ = zdt1_result
        feasible = result.optimization.feasible_points
        f1_vals = [p.objectives["f1"] for p in feasible]
        f2_vals = [p.objectives["f2"] for p in feasible]
        assert min(f1_vals) < 0.5
        assert min(f2_vals) < 1.0

    def test_zdt1_known_pareto_front_shape(
        self, zdt1_result: RunResult,
    ) -> None:
        result, _ = zdt1_result
        feasible = result.optimization.feasible_points
        for p in feasible:
            f1 = p.objectives["f1"]
            f2 = p.objectives["f2"]
            assert f1 >= -0.1, f"f1={f1} should be >= 0"
            assert f2 >= -0.5, f"f2={f2} should be >= 0"
