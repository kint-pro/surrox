from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

import surrox
from surrox import Analyzer, SurroxResult
from surrox.optimizer.config import OptimizerConfig
from surrox.surrogate.config import TrainingConfig
from tests.integration.benchmarks import (
    generate_branin,
    generate_rosenbrock,
)

SURROGATE_CONFIG = TrainingConfig(
    n_trials=5,
    cv_folds=3,
    ensemble_size=2,
    min_r2=None,
    random_seed=42,
)

OPTIMIZER_CONFIG = OptimizerConfig(
    population_size=30,
    n_generations=20,
    seed=42,
)

RunResult = tuple[SurroxResult, Analyzer]


def _run_branin() -> RunResult:
    problem, df = generate_branin(n_samples=600, seed=42)
    return surrox.run(
        problem, df,
        surrogate_config=SURROGATE_CONFIG,
        optimizer_config=OPTIMIZER_CONFIG,
    )


def _run_rosenbrock() -> RunResult:
    problem, df = generate_rosenbrock(n_samples=600, seed=99)
    return surrox.run(
        problem, df,
        surrogate_config=SURROGATE_CONFIG,
        optimizer_config=OPTIMIZER_CONFIG,
    )


@pytest.mark.slow
class TestTenantIsolation:
    def test_parallel_runs_produce_correct_results(
        self,
    ) -> None:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_branin = executor.submit(_run_branin)
            future_rosenbrock = executor.submit(_run_rosenbrock)

            branin_result, _ = future_branin.result()
            rosenbrock_result, _ = future_rosenbrock.result()

        branin_objs = branin_result.optimization.feasible_points[0].objectives
        assert "branin" in branin_objs
        assert "rosenbrock" not in branin_objs

        rosenbrock_objs = rosenbrock_result.optimization.feasible_points[0].objectives
        assert "rosenbrock" in rosenbrock_objs
        assert "branin" not in rosenbrock_objs

    def test_parallel_runs_no_shared_state(
        self,
    ) -> None:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(_run_branin)
            future_b = executor.submit(_run_branin)

            result_a, _ = future_a.result()
            result_b, _ = future_b.result()

        summary_a = result_a.analysis.summary.solution_summary
        summary_b = result_b.analysis.summary.solution_summary

        assert summary_a.best_objectives == summary_b.best_objectives
        assert summary_a.n_feasible == summary_b.n_feasible

    def test_sequential_runs_independent(self) -> None:
        result_1, analyzer_1 = _run_branin()
        result_2, analyzer_2 = _run_rosenbrock()

        assert result_1.optimization.problem != result_2.optimization.problem
        assert analyzer_1 is not analyzer_2
        assert analyzer_1._cache is not analyzer_2._cache

    def test_result_immutability_across_tenants(self) -> None:
        result_a, _ = _run_branin()
        result_b, _ = _run_branin()

        assert result_a is not result_b
        assert (
            result_a.optimization.feasible_points
            is not result_b.optimization.feasible_points
        )

    def test_analyzer_cache_isolated(self) -> None:
        _, analyzer_a = _run_branin()
        _, analyzer_b = _run_branin()

        analyzer_a._cache[("test_key",)] = "tenant_a_data"

        assert ("test_key",) not in analyzer_b._cache
