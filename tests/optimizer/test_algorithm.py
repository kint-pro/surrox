from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.mixed import MixedVariableGA

from surrox.optimizer.algorithm import select_algorithm
from surrox.optimizer.config import OptimizerConfig
from surrox.problem.definition import ProblemDefinition
from surrox.problem.objectives import Objective
from surrox.problem.types import Direction, DType, Role
from surrox.problem.variables import (
    CategoricalBounds,
    ContinuousBounds,
    IntegerBounds,
    OrdinalBounds,
    Variable,
)


def _make_problem(
    dtypes: list[DType],
    n_objectives: int = 1,
) -> ProblemDefinition:
    bounds_map = {
        DType.CONTINUOUS: ContinuousBounds(lower=0.0, upper=10.0),
        DType.INTEGER: IntegerBounds(lower=1, upper=10),
        DType.CATEGORICAL: CategoricalBounds(categories=("a", "b")),
        DType.ORDINAL: OrdinalBounds(categories=("low", "high")),
    }
    variables = tuple(
        Variable(name=f"x{i}", dtype=dt, role=Role.DECISION, bounds=bounds_map[dt])
        for i, dt in enumerate(dtypes)
    )
    objectives = tuple(
        Objective(name=f"obj{i}", direction=Direction.MINIMIZE, column=f"y{i}")
        for i in range(n_objectives)
    )
    return ProblemDefinition(variables=variables, objectives=objectives)


class TestSingleObjective:
    def test_all_continuous_selects_de(self) -> None:
        problem = _make_problem([DType.CONTINUOUS, DType.CONTINUOUS])
        algo = select_algorithm(problem, OptimizerConfig())
        assert isinstance(algo, DE)

    def test_with_integer_selects_ga(self) -> None:
        problem = _make_problem([DType.CONTINUOUS, DType.INTEGER])
        algo = select_algorithm(problem, OptimizerConfig())
        assert isinstance(algo, GA)

    def test_with_categorical_selects_mixed_ga(self) -> None:
        problem = _make_problem([DType.CONTINUOUS, DType.CATEGORICAL])
        algo = select_algorithm(problem, OptimizerConfig())
        assert isinstance(algo, MixedVariableGA)

    def test_with_ordinal_selects_mixed_ga(self) -> None:
        problem = _make_problem([DType.CONTINUOUS, DType.ORDINAL])
        algo = select_algorithm(problem, OptimizerConfig())
        assert isinstance(algo, MixedVariableGA)


class TestMultiObjective:
    def test_two_obj_continuous_selects_nsga2(self) -> None:
        problem = _make_problem([DType.CONTINUOUS, DType.CONTINUOUS], n_objectives=2)
        algo = select_algorithm(problem, OptimizerConfig())
        assert isinstance(algo, NSGA2)

    def test_two_obj_integer_selects_nsga2(self) -> None:
        problem = _make_problem([DType.CONTINUOUS, DType.INTEGER], n_objectives=2)
        algo = select_algorithm(problem, OptimizerConfig())
        assert isinstance(algo, NSGA2)

    def test_two_obj_categorical_selects_mixed_ga(self) -> None:
        problem = _make_problem([DType.CONTINUOUS, DType.CATEGORICAL], n_objectives=2)
        algo = select_algorithm(problem, OptimizerConfig())
        assert isinstance(algo, MixedVariableGA)

    def test_three_obj_continuous_selects_nsga2(self) -> None:
        problem = _make_problem([DType.CONTINUOUS, DType.CONTINUOUS], n_objectives=3)
        algo = select_algorithm(problem, OptimizerConfig())
        assert isinstance(algo, NSGA2)


class TestManyObjective:
    def test_four_obj_continuous_selects_nsga3(self) -> None:
        problem = _make_problem([DType.CONTINUOUS, DType.CONTINUOUS], n_objectives=4)
        algo = select_algorithm(problem, OptimizerConfig())
        assert isinstance(algo, NSGA3)

    def test_four_obj_integer_selects_nsga3(self) -> None:
        problem = _make_problem([DType.CONTINUOUS, DType.INTEGER], n_objectives=4)
        algo = select_algorithm(problem, OptimizerConfig())
        assert isinstance(algo, NSGA3)

    def test_four_obj_categorical_selects_mixed_ga(self) -> None:
        problem = _make_problem([DType.CONTINUOUS, DType.CATEGORICAL], n_objectives=4)
        algo = select_algorithm(problem, OptimizerConfig())
        assert isinstance(algo, MixedVariableGA)
