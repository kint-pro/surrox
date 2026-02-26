import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from surrox.problem.constraints import DataConstraint
from surrox.problem.dataset import BoundDataset
from surrox.problem.definition import ProblemDefinition
from surrox.problem.objectives import Objective
from surrox.problem.types import (
    ConstraintOperator,
    Direction,
    DType,
    Role,
)
from surrox.problem.variables import (
    CategoricalBounds,
    ContinuousBounds,
    IntegerBounds,
    Variable,
)


def make_simple_problem(
    variables: tuple[Variable, ...] | None = None,
    objectives: tuple[Objective, ...] | None = None,
    data_constraints: tuple[DataConstraint, ...] = (),
) -> ProblemDefinition:
    if variables is None:
        variables = (
            Variable(
                name="x",
                dtype=DType.CONTINUOUS,
                role=Role.DECISION,
                bounds=ContinuousBounds(lower=0.0, upper=10.0),
            ),
        )
    if objectives is None:
        objectives = (
            Objective(name="y", direction=Direction.MINIMIZE, column="y_col"),
        )
    return ProblemDefinition(
        variables=variables,
        objectives=objectives,
        data_constraints=data_constraints,
    )


class TestBoundDatasetHappyPath:
    def test_create_with_valid_data(self) -> None:
        problem = make_simple_problem()
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y_col": [10.0, 20.0, 30.0]})
        bound = BoundDataset(problem=problem, dataframe=df)
        assert bound.problem is problem
        assert len(bound.dataframe) == 3

    def test_create_with_full_problem(
        self, full_problem: ProblemDefinition, valid_dataframe: pd.DataFrame
    ) -> None:
        bound = BoundDataset(problem=full_problem, dataframe=valid_dataframe)
        assert len(bound.dataframe) == 4

    def test_multiple_datasets_per_problem(self) -> None:
        problem = make_simple_problem()
        df1 = pd.DataFrame({"x": [1.0, 2.0], "y_col": [10.0, 20.0]})
        df2 = pd.DataFrame({"x": [3.0, 4.0], "y_col": [30.0, 40.0]})
        bound1 = BoundDataset(problem=problem, dataframe=df1)
        bound2 = BoundDataset(problem=problem, dataframe=df2)
        assert len(bound1.dataframe) == 2
        assert len(bound2.dataframe) == 2


class TestBoundDatasetColumnExistence:
    def test_missing_variable_column_raises(self) -> None:
        problem = make_simple_problem()
        df = pd.DataFrame({"wrong_name": [1.0], "y_col": [10.0]})
        with pytest.raises(ValidationError, match="column 'x' not found"):
            BoundDataset(problem=problem, dataframe=df)

    def test_missing_objective_column_raises(self) -> None:
        problem = make_simple_problem()
        df = pd.DataFrame({"x": [1.0], "wrong_col": [10.0]})
        with pytest.raises(ValidationError, match="column 'y_col' not found"):
            BoundDataset(problem=problem, dataframe=df)

    def test_missing_data_constraint_column_raises(self) -> None:
        dc = DataConstraint(
            name="dc", column="dc_col", operator=ConstraintOperator.LE, limit=100.0
        )
        problem = make_simple_problem(data_constraints=(dc,))
        df = pd.DataFrame({"x": [1.0], "y_col": [10.0]})
        with pytest.raises(ValidationError, match="column 'dc_col' not found"):
            BoundDataset(problem=problem, dataframe=df)


class TestBoundDatasetMissingValues:
    def test_missing_values_in_variable_column_raises(self) -> None:
        problem = make_simple_problem()
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y_col": [10.0, 20.0, 30.0]})
        with pytest.raises(
            ValidationError, match="variable 'x'.*contains missing values"
        ):
            BoundDataset(problem=problem, dataframe=df)

    def test_missing_values_in_objective_column_raises(self) -> None:
        problem = make_simple_problem()
        df = pd.DataFrame({"x": [1.0, 2.0], "y_col": [10.0, np.nan]})
        with pytest.raises(
            ValidationError, match="objective 'y'.*contains missing values"
        ):
            BoundDataset(problem=problem, dataframe=df)

    def test_missing_values_in_data_constraint_column_raises(self) -> None:
        dc = DataConstraint(
            name="dc", column="dc_col", operator=ConstraintOperator.LE, limit=100.0
        )
        problem = make_simple_problem(data_constraints=(dc,))
        df = pd.DataFrame({"x": [1.0], "y_col": [10.0], "dc_col": [np.nan]})
        with pytest.raises(
            ValidationError, match="data constraint 'dc'.*contains missing values"
        ):
            BoundDataset(problem=problem, dataframe=df)

    def test_missing_values_in_categorical_variable_raises(self) -> None:
        var = Variable(
            name="cat",
            dtype=DType.CATEGORICAL,
            role=Role.DECISION,
            bounds=CategoricalBounds(categories=("a", "b")),
        )
        problem = make_simple_problem(variables=(var,))
        df = pd.DataFrame({"cat": ["a", None], "y_col": [1.0, 2.0]})
        with pytest.raises(
            ValidationError, match="variable 'cat'.*contains missing values"
        ):
            BoundDataset(problem=problem, dataframe=df)


class TestBoundDatasetNumericValidation:
    def test_values_below_lower_bound_raises(self) -> None:
        problem = make_simple_problem()
        df = pd.DataFrame({"x": [-1.0, 5.0], "y_col": [10.0, 20.0]})
        with pytest.raises(ValidationError, match="values outside bounds"):
            BoundDataset(problem=problem, dataframe=df)

    def test_values_above_upper_bound_raises(self) -> None:
        problem = make_simple_problem()
        df = pd.DataFrame({"x": [5.0, 11.0], "y_col": [10.0, 20.0]})
        with pytest.raises(ValidationError, match="values outside bounds"):
            BoundDataset(problem=problem, dataframe=df)

    def test_values_at_bounds_accepted(self) -> None:
        problem = make_simple_problem()
        df = pd.DataFrame({"x": [0.0, 10.0], "y_col": [10.0, 20.0]})
        bound = BoundDataset(problem=problem, dataframe=df)
        assert len(bound.dataframe) == 2

    def test_non_numeric_dtype_for_continuous_variable_raises(self) -> None:
        problem = make_simple_problem()
        df = pd.DataFrame({"x": ["a", "b"], "y_col": [10.0, 20.0]})
        with pytest.raises(ValidationError, match="expected numeric dtype"):
            BoundDataset(problem=problem, dataframe=df)

    def test_integer_variable_with_float_values_raises(self) -> None:
        var = Variable(
            name="count",
            dtype=DType.INTEGER,
            role=Role.DECISION,
            bounds=IntegerBounds(lower=0, upper=10),
        )
        problem = make_simple_problem(variables=(var,))
        df = pd.DataFrame({"count": [1.5, 2.7], "y_col": [10.0, 20.0]})
        with pytest.raises(ValidationError, match="contains non-integer values"):
            BoundDataset(problem=problem, dataframe=df)

    def test_integer_variable_with_integer_values_accepted(self) -> None:
        var = Variable(
            name="count",
            dtype=DType.INTEGER,
            role=Role.DECISION,
            bounds=IntegerBounds(lower=0, upper=10),
        )
        problem = make_simple_problem(variables=(var,))
        df = pd.DataFrame({"count": [1, 3, 5], "y_col": [10.0, 20.0, 30.0]})
        bound = BoundDataset(problem=problem, dataframe=df)
        assert len(bound.dataframe) == 3

    def test_integer_variable_with_float64_whole_numbers_accepted(self) -> None:
        var = Variable(
            name="count",
            dtype=DType.INTEGER,
            role=Role.DECISION,
            bounds=IntegerBounds(lower=0, upper=10),
        )
        problem = make_simple_problem(variables=(var,))
        df = pd.DataFrame({"count": [1.0, 2.0, 3.0], "y_col": [10.0, 20.0, 30.0]})
        bound = BoundDataset(problem=problem, dataframe=df)
        assert len(bound.dataframe) == 3

    def test_integer_variable_outside_bounds_raises(self) -> None:
        var = Variable(
            name="count",
            dtype=DType.INTEGER,
            role=Role.DECISION,
            bounds=IntegerBounds(lower=0, upper=10),
        )
        problem = make_simple_problem(variables=(var,))
        df = pd.DataFrame({"count": [0, 11], "y_col": [10.0, 20.0]})
        with pytest.raises(ValidationError, match="values outside bounds"):
            BoundDataset(problem=problem, dataframe=df)


class TestBoundDatasetCategoricalValidation:
    def test_invalid_category_raises(self) -> None:
        var = Variable(
            name="cat",
            dtype=DType.CATEGORICAL,
            role=Role.DECISION,
            bounds=CategoricalBounds(categories=("a", "b")),
        )
        problem = make_simple_problem(variables=(var,))
        df = pd.DataFrame({"cat": ["a", "c"], "y_col": [10.0, 20.0]})
        with pytest.raises(ValidationError, match="invalid categories"):
            BoundDataset(problem=problem, dataframe=df)

    def test_valid_categories_accepted(self) -> None:
        var = Variable(
            name="cat",
            dtype=DType.CATEGORICAL,
            role=Role.DECISION,
            bounds=CategoricalBounds(categories=("a", "b", "c")),
        )
        problem = make_simple_problem(variables=(var,))
        df = pd.DataFrame({"cat": ["a", "b", "c"], "y_col": [10.0, 20.0, 30.0]})
        bound = BoundDataset(problem=problem, dataframe=df)
        assert len(bound.dataframe) == 3


class TestBoundDatasetTargetDtypeValidation:
    def test_non_numeric_objective_column_raises(self) -> None:
        problem = make_simple_problem()
        df = pd.DataFrame({"x": [1.0, 2.0], "y_col": ["high", "low"]})
        with pytest.raises(
            ValidationError, match="objective 'y'.*expected numeric dtype"
        ):
            BoundDataset(problem=problem, dataframe=df)

    def test_non_numeric_data_constraint_column_raises(self) -> None:
        dc = DataConstraint(
            name="dc", column="dc_col", operator=ConstraintOperator.LE, limit=100.0
        )
        problem = make_simple_problem(data_constraints=(dc,))
        df = pd.DataFrame(
            {"x": [1.0, 2.0], "y_col": [10.0, 20.0], "dc_col": ["ok", "bad"]}
        )
        with pytest.raises(
            ValidationError, match="data constraint 'dc'.*expected numeric dtype"
        ):
            BoundDataset(problem=problem, dataframe=df)


class TestBoundDatasetImmutability:
    def test_is_frozen(self) -> None:
        problem = make_simple_problem()
        df = pd.DataFrame({"x": [1.0], "y_col": [10.0]})
        bound = BoundDataset(problem=problem, dataframe=df)
        with pytest.raises(ValidationError):
            bound.problem = problem  # type: ignore[misc]
