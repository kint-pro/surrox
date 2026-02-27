from pydantic import BaseModel, ConfigDict, model_validator

from surrox.exceptions import ProblemDefinitionError
from surrox.problem.types import ConstraintOperator


class LinearConstraint(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    coefficients: dict[str, float]
    operator: ConstraintOperator
    rhs: float

    @model_validator(mode="after")
    def _validate_coefficients(self) -> "LinearConstraint":
        if not self.coefficients:
            raise ProblemDefinitionError("coefficients must not be empty")
        zero_coefficients = [
            name for name, value in self.coefficients.items() if value == 0.0
        ]
        if zero_coefficients:
            raise ProblemDefinitionError(
                f"coefficients must not be zero: {zero_coefficients}"
            )
        return self


class DataConstraint(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    column: str
    operator: ConstraintOperator
    limit: float
    tolerance: float | None = None

    @model_validator(mode="after")
    def _validate_tolerance(self) -> "DataConstraint":
        if self.operator == ConstraintOperator.EQ and self.tolerance is None:
            raise ProblemDefinitionError(
                "tolerance is required when operator is EQ"
            )
        if self.operator != ConstraintOperator.EQ and self.tolerance is not None:
            raise ProblemDefinitionError(
                "tolerance must be None when operator is not EQ"
            )
        if self.tolerance is not None and self.tolerance <= 0:
            raise ProblemDefinitionError(
                "tolerance must be positive"
            )
        return self
