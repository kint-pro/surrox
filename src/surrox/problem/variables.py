from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from surrox.exceptions import ProblemDefinitionError
from surrox.problem.types import DType, Role


class ContinuousBounds(BaseModel):
    """Bounds for a continuous variable, defined by a lower and upper limit.

    Attributes:
        lower: Minimum allowed value (exclusive of upper).
        upper: Maximum allowed value.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["continuous"] = "continuous"
    lower: float
    upper: float

    @model_validator(mode="after")
    def _validate_bounds(self) -> "ContinuousBounds":
        if self.lower >= self.upper:
            raise ProblemDefinitionError(
                f"lower ({self.lower}) must be less than upper ({self.upper})"
            )
        return self


class IntegerBounds(BaseModel):
    """Bounds for an integer variable, defined by a lower and upper limit.

    Attributes:
        lower: Minimum allowed value (exclusive of upper).
        upper: Maximum allowed value.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["integer"] = "integer"
    lower: int
    upper: int

    @model_validator(mode="after")
    def _validate_bounds(self) -> "IntegerBounds":
        if self.lower >= self.upper:
            raise ProblemDefinitionError(
                f"lower ({self.lower}) must be less than upper ({self.upper})"
            )
        return self


class CategoricalBounds(BaseModel):
    """Bounds for a categorical variable, defined by a set of unordered categories.

    Attributes:
        categories: Allowed category values. Must contain at least 2 unique entries.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["categorical"] = "categorical"
    categories: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_categories(self) -> "CategoricalBounds":
        if len(self.categories) < 2:
            raise ProblemDefinitionError(
                "categorical bounds require at least 2 categories"
            )
        if len(self.categories) != len(set(self.categories)):
            raise ProblemDefinitionError("categories must be unique")
        return self


class OrdinalBounds(BaseModel):
    """Bounds for an ordinal variable, defined by an ordered sequence of categories.

    Attributes:
        categories: Allowed category values in order. Must contain at least 2 unique entries.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["ordinal"] = "ordinal"
    categories: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_categories(self) -> "OrdinalBounds":
        if len(self.categories) < 2:
            raise ProblemDefinitionError("ordinal bounds require at least 2 categories")
        if len(self.categories) != len(set(self.categories)):
            raise ProblemDefinitionError("categories must be unique")
        return self


Bounds = Annotated[
    ContinuousBounds | IntegerBounds | CategoricalBounds | OrdinalBounds,
    Field(discriminator="type"),
]

_DTYPE_TO_BOUNDS_TYPE: dict[DType, str] = {
    DType.CONTINUOUS: "continuous",
    DType.INTEGER: "integer",
    DType.CATEGORICAL: "categorical",
    DType.ORDINAL: "ordinal",
}


class Variable(BaseModel):
    """A variable in the optimization problem.

    Attributes:
        name: Column name in the dataset.
        dtype: Data type (continuous, integer, categorical, ordinal).
        role: Role in optimization (decision or context).
        bounds: Domain bounds matching the dtype.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    dtype: DType
    role: Role
    bounds: Bounds

    @model_validator(mode="after")
    def _validate_dtype_bounds_consistency(self) -> "Variable":
        expected_type = _DTYPE_TO_BOUNDS_TYPE[self.dtype]
        if self.bounds.type != expected_type:
            raise ProblemDefinitionError(
                f"dtype {self.dtype} requires {expected_type} bounds, "
                f"got {self.bounds.type} bounds"
            )
        return self
