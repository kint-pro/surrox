import pandas as pd
import pytest

from surrox.problem.constraints import DataConstraint, LinearConstraint
from surrox.problem.definition import ProblemDefinition
from surrox.problem.domain_knowledge import MonotonicRelation
from surrox.problem.objectives import Objective
from surrox.problem.scenarios import Scenario
from surrox.problem.types import (
    ConstraintOperator,
    Direction,
    DType,
    MonotonicDirection,
    Role,
)
from surrox.problem.variables import (
    CategoricalBounds,
    ContinuousBounds,
    IntegerBounds,
    OrdinalBounds,
    Variable,
)


@pytest.fixture
def continuous_bounds() -> ContinuousBounds:
    return ContinuousBounds(lower=0.0, upper=10.0)


@pytest.fixture
def integer_bounds() -> IntegerBounds:
    return IntegerBounds(lower=0, upper=100)


@pytest.fixture
def categorical_bounds() -> CategoricalBounds:
    return CategoricalBounds(categories=("low", "medium", "high"))


@pytest.fixture
def ordinal_bounds() -> OrdinalBounds:
    return OrdinalBounds(categories=("small", "medium", "large"))


@pytest.fixture
def decision_variable(continuous_bounds: ContinuousBounds) -> Variable:
    return Variable(
        name="temperature",
        dtype=DType.CONTINUOUS,
        role=Role.DECISION,
        bounds=continuous_bounds,
    )


@pytest.fixture
def context_variable(categorical_bounds: CategoricalBounds) -> Variable:
    return Variable(
        name="mode",
        dtype=DType.CATEGORICAL,
        role=Role.CONTEXT,
        bounds=categorical_bounds,
    )


@pytest.fixture
def minimize_objective() -> Objective:
    return Objective(name="cost", direction=Direction.MINIMIZE, column="cost_col")


@pytest.fixture
def maximize_objective() -> Objective:
    return Objective(name="efficiency", direction=Direction.MAXIMIZE, column="eff_col")


@pytest.fixture
def linear_constraint() -> LinearConstraint:
    return LinearConstraint(
        name="temp_limit",
        coefficients={"temperature": 1.0},
        operator=ConstraintOperator.LE,
        rhs=8.0,
    )


@pytest.fixture
def data_constraint() -> DataConstraint:
    return DataConstraint(
        name="pressure_limit",
        column="pressure_col",
        operator=ConstraintOperator.LE,
        limit=100.0,
    )


@pytest.fixture
def monotonic_relation() -> MonotonicRelation:
    return MonotonicRelation(
        decision_variable="temperature",
        objective_or_constraint="cost",
        direction=MonotonicDirection.INCREASING,
    )


@pytest.fixture
def scenario() -> Scenario:
    return Scenario(name="baseline", context_values={"mode": "low"})


@pytest.fixture
def minimal_problem(
    decision_variable: Variable, minimize_objective: Objective
) -> ProblemDefinition:
    return ProblemDefinition(
        variables=(decision_variable,),
        objectives=(minimize_objective,),
    )


@pytest.fixture
def full_problem(
    decision_variable: Variable,
    context_variable: Variable,
    minimize_objective: Objective,
    maximize_objective: Objective,
    linear_constraint: LinearConstraint,
    data_constraint: DataConstraint,
    monotonic_relation: MonotonicRelation,
    scenario: Scenario,
) -> ProblemDefinition:
    return ProblemDefinition(
        variables=(decision_variable, context_variable),
        objectives=(minimize_objective, maximize_objective),
        linear_constraints=(linear_constraint,),
        data_constraints=(data_constraint,),
        monotonic_relations=(monotonic_relation,),
        scenarios=(scenario,),
    )


@pytest.fixture
def valid_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temperature": [1.0, 3.0, 5.0, 7.0],
            "mode": ["low", "medium", "high", "low"],
            "cost_col": [100.0, 200.0, 150.0, 180.0],
            "eff_col": [0.8, 0.9, 0.85, 0.88],
            "pressure_col": [50.0, 60.0, 55.0, 58.0],
        }
    )
