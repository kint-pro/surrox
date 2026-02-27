import pytest

from surrox.analysis.scenario import compare_scenarios
from surrox.exceptions import AnalysisError
from surrox.optimizer.result import OptimizationResult
from surrox.problem.definition import ProblemDefinition

from tests.analysis.conftest import _make_evaluated_point


class TestCompareScenarios:
    def test_robust_variables(
        self, single_objective_problem: ProblemDefinition,
    ) -> None:
        result_a = OptimizationResult(
            feasible_points=(
                _make_evaluated_point(
                    variables={"x1": 5.0, "x2": 5.0},
                    objectives={"cost": 40.0},
                ),
            ),
            infeasible_points=(),
            has_feasible_solutions=True,
            compromise_index=None,
            hypervolume=None,
            problem=single_objective_problem,
            n_generations=10,
            n_evaluations=100,
        )
        result_b = OptimizationResult(
            feasible_points=(
                _make_evaluated_point(
                    variables={"x1": 5.1, "x2": 5.0},
                    objectives={"cost": 42.0},
                ),
            ),
            infeasible_points=(),
            has_feasible_solutions=True,
            compromise_index=None,
            hypervolume=None,
            problem=single_objective_problem,
            n_generations=10,
            n_evaluations=100,
        )
        result = compare_scenarios(
            {"scenario_a": result_a, "scenario_b": result_b},
            single_objective_problem,
        )
        assert "x1" in result.variable_robustness
        assert "x2" in result.variable_robustness
        assert result.variable_robustness["x2"].is_robust is True
        assert result.variable_robustness["x2"].spread == 0.0

    def test_non_robust_variable(
        self, single_objective_problem: ProblemDefinition,
    ) -> None:
        result_a = OptimizationResult(
            feasible_points=(
                _make_evaluated_point(
                    variables={"x1": 1.0, "x2": 5.0},
                    objectives={"cost": 40.0},
                ),
            ),
            infeasible_points=(),
            has_feasible_solutions=True,
            compromise_index=None,
            hypervolume=None,
            problem=single_objective_problem,
            n_generations=10,
            n_evaluations=100,
        )
        result_b = OptimizationResult(
            feasible_points=(
                _make_evaluated_point(
                    variables={"x1": 9.0, "x2": 5.0},
                    objectives={"cost": 60.0},
                ),
            ),
            infeasible_points=(),
            has_feasible_solutions=True,
            compromise_index=None,
            hypervolume=None,
            problem=single_objective_problem,
            n_generations=10,
            n_evaluations=100,
        )
        result = compare_scenarios(
            {"a": result_a, "b": result_b},
            single_objective_problem,
        )
        assert result.variable_robustness["x1"].is_robust is False
        assert result.variable_robustness["x1"].spread == 8.0

    def test_less_than_two_scenarios_raises(
        self, single_objective_problem: ProblemDefinition,
        single_objective_result: OptimizationResult,
    ) -> None:
        with pytest.raises(AnalysisError, match="at least 2"):
            compare_scenarios(
                {"only_one": single_objective_result},
                single_objective_problem,
            )

    def test_no_feasible_scenario_raises(
        self,
        single_objective_problem: ProblemDefinition,
        single_objective_result: OptimizationResult,
        no_feasible_result: OptimizationResult,
    ) -> None:
        with pytest.raises(AnalysisError, match="no feasible"):
            compare_scenarios(
                {"a": single_objective_result, "b": no_feasible_result},
                single_objective_problem,
            )
