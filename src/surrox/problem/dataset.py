import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from surrox.exceptions import ProblemDefinitionError
from surrox.problem.definition import ProblemDefinition
from surrox.problem.types import DType
from surrox.problem.variables import Variable


def _validate_column_exists(df: pd.DataFrame, column: str, context: str) -> None:
    if column not in df.columns:
        raise ProblemDefinitionError(
            f"{context}: column '{column}' not found in dataset"
        )


def _validate_no_missing_values(df: pd.DataFrame, column: str, context: str) -> None:
    if df[column].isna().any():  # type: ignore[truthy-bool]
        raise ProblemDefinitionError(
            f"{context}: column '{column}' contains missing values"
        )


def _validate_numeric_bounds(df: pd.DataFrame, variable: Variable) -> None:
    col = df[variable.name]
    lower = variable.bounds.lower  # type: ignore[union-attr]
    upper = variable.bounds.upper  # type: ignore[union-attr]
    if (col < lower).any() or (col > upper).any():
        raise ProblemDefinitionError(
            f"variable '{variable.name}': values outside bounds [{lower}, {upper}]"
        )


def _validate_categorical_values(df: pd.DataFrame, variable: Variable) -> None:
    col = df[variable.name]
    valid_categories = set(variable.bounds.categories)  # type: ignore[union-attr]
    invalid = set(col.dropna().unique()) - valid_categories
    if invalid:
        raise ProblemDefinitionError(
            f"variable '{variable.name}': invalid categories {invalid}"
        )


def _validate_integer_dtype(df: pd.DataFrame, variable: Variable) -> None:
    col = df[variable.name]
    if np.issubdtype(col.dtype, np.integer):  # pyright: ignore[reportArgumentType]
        return
    if not (col == col.astype(int)).all():
        raise ProblemDefinitionError(
            f"variable '{variable.name}': contains non-integer values"
        )


def _validate_numeric_target_dtype(df: pd.DataFrame, column: str, context: str) -> None:
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ProblemDefinitionError(
            f"{context}: column '{column}' expected numeric dtype, "
            f"got {df[column].dtype}"
        )


def _validate_numeric_dtype(df: pd.DataFrame, variable: Variable) -> None:
    col = df[variable.name]
    if not pd.api.types.is_numeric_dtype(col):
        raise ProblemDefinitionError(
            f"variable '{variable.name}': expected numeric dtype, got {col.dtype}"
        )


class BoundDataset(BaseModel):
    """A DataFrame validated against a ProblemDefinition.

    Validates at construction that all required columns exist, contain no
    missing values, and satisfy dtype and bounds constraints.

    Attributes:
        problem: The problem definition to validate against.
        dataframe: Historical data. Columns must match variable names and target columns.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    problem: ProblemDefinition
    dataframe: pd.DataFrame

    @model_validator(mode="after")
    def _validate_binding(self) -> "BoundDataset":
        self._validate_variable_columns()
        self._validate_target_columns()
        return self

    def _validate_variable_columns(self) -> None:
        for variable in self.problem.variables:
            _validate_column_exists(
                self.dataframe, variable.name, f"variable '{variable.name}'"
            )
            _validate_no_missing_values(
                self.dataframe, variable.name, f"variable '{variable.name}'"
            )

            if variable.dtype in (DType.CONTINUOUS, DType.INTEGER):
                _validate_numeric_dtype(self.dataframe, variable)
                _validate_numeric_bounds(self.dataframe, variable)

            if variable.dtype == DType.INTEGER:
                _validate_integer_dtype(self.dataframe, variable)

            if variable.dtype in (DType.CATEGORICAL, DType.ORDINAL):
                _validate_categorical_values(self.dataframe, variable)

    def _validate_target_columns(self) -> None:
        for objective in self.problem.objectives:
            context = f"objective '{objective.name}'"
            _validate_column_exists(self.dataframe, objective.column, context)
            _validate_no_missing_values(self.dataframe, objective.column, context)
            _validate_numeric_target_dtype(self.dataframe, objective.column, context)

        for constraint in self.problem.data_constraints:
            context = f"data constraint '{constraint.name}'"
            _validate_column_exists(self.dataframe, constraint.column, context)
            _validate_no_missing_values(self.dataframe, constraint.column, context)
            _validate_numeric_target_dtype(self.dataframe, constraint.column, context)
