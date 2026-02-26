from pydantic import BaseModel, ConfigDict, model_validator

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
            raise ValueError("coefficients must not be empty")
        zero_coefficients = [
            name for name, value in self.coefficients.items() if value == 0.0
        ]
        if zero_coefficients:
            raise ValueError(f"coefficients must not be zero: {zero_coefficients}")
        return self


class DataConstraint(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    column: str
    operator: ConstraintOperator
    limit: float
