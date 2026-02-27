from surrox.optimizer.result import (
    ConstraintEvaluation,
    EvaluatedPoint,
    OptimizationResult,
    _compute_compromise_index,
)
from surrox.problem.definition import ProblemDefinition
from surrox.problem.objectives import Objective
from surrox.problem.types import Direction, DType, Role
from surrox.problem.variables import ContinuousBounds, Variable


def _make_point(
    objectives: dict[str, float],
    feasible: bool = True,
    violation: float = -1.0,
) -> EvaluatedPoint:
    return EvaluatedPoint(
        variables={"x1": 1.0},
        objectives=objectives,
        constraints=(
            ConstraintEvaluation(name="c1", violation=violation, prediction=0.0),
        ),
        feasible=feasible,
        extrapolation_distance=1.0,
        is_extrapolating=False,
    )


class TestConstraintEvaluation:
    def test_create_with_bounds(self) -> None:
        ce = ConstraintEvaluation(
            name="pressure", violation=2.3, prediction=102.3,
            lower_bound=95.0, upper_bound=110.0,
        )
        assert ce.name == "pressure"
        assert ce.violation == 2.3
        assert ce.lower_bound == 95.0
        assert ce.upper_bound == 110.0

    def test_create_without_bounds(self) -> None:
        ce = ConstraintEvaluation(name="linear_c", violation=-0.5, prediction=9.5)
        assert ce.lower_bound is None
        assert ce.upper_bound is None


class TestEvaluatedPoint:
    def test_create(self) -> None:
        point = _make_point({"cost": 10.0, "quality": 80.0})
        assert point.feasible
        assert point.objectives["cost"] == 10.0

    def test_extrapolation_info(self) -> None:
        point = EvaluatedPoint(
            variables={"x1": 100.0},
            objectives={"cost": 10.0},
            constraints=(),
            feasible=True,
            extrapolation_distance=3.5,
            is_extrapolating=True,
        )
        assert point.is_extrapolating
        assert point.extrapolation_distance == 3.5


class TestCompromiseIndex:
    def test_single_objective_returns_none(self) -> None:
        points = (_make_point({"cost": 10.0}),)
        assert _compute_compromise_index(points, 1) is None

    def test_single_point_returns_none(self) -> None:
        points = (_make_point({"cost": 10.0, "quality": 80.0}),)
        assert _compute_compromise_index(points, 2) is None

    def test_selects_closest_to_utopia(self) -> None:
        points = (
            _make_point({"cost": 0.0, "quality": 100.0}),
            _make_point({"cost": 50.0, "quality": 50.0}),
            _make_point({"cost": 100.0, "quality": 0.0}),
        )
        idx = _compute_compromise_index(points, 2)
        assert idx == 1

    def test_balanced_point_preferred(self) -> None:
        points = (
            _make_point({"cost": 0.0, "quality": 10.0}),
            _make_point({"cost": 5.0, "quality": 5.0}),
            _make_point({"cost": 10.0, "quality": 0.0}),
        )
        idx = _compute_compromise_index(points, 2)
        assert idx == 1


class TestOptimizationResult:
    def test_has_feasible_solutions(self) -> None:
        problem = ProblemDefinition(
            variables=(
                Variable(name="x1", dtype=DType.CONTINUOUS, role=Role.DECISION,
                         bounds=ContinuousBounds(lower=0.0, upper=10.0)),
            ),
            objectives=(Objective(name="obj", direction=Direction.MINIMIZE, column="y"),),
        )
        result = OptimizationResult(
            feasible_points=(_make_point({"obj": 5.0}),),
            infeasible_points=(),
            has_feasible_solutions=True,
            compromise_index=None,
            hypervolume=None,
            problem=problem,
            n_generations=10,
            n_evaluations=100,
        )
        assert result.has_feasible_solutions
        assert len(result.feasible_points) == 1

    def test_no_feasible_solutions(self) -> None:
        problem = ProblemDefinition(
            variables=(
                Variable(name="x1", dtype=DType.CONTINUOUS, role=Role.DECISION,
                         bounds=ContinuousBounds(lower=0.0, upper=10.0)),
            ),
            objectives=(Objective(name="obj", direction=Direction.MINIMIZE, column="y"),),
        )
        result = OptimizationResult(
            feasible_points=(),
            infeasible_points=(_make_point({"obj": 5.0}, feasible=False, violation=1.0),),
            has_feasible_solutions=False,
            compromise_index=None,
            hypervolume=None,
            problem=problem,
            n_generations=10,
            n_evaluations=100,
        )
        assert not result.has_feasible_solutions
        assert len(result.infeasible_points) == 1
