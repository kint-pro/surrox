from pydantic import BaseModel, ConfigDict, model_validator

from surrox.exceptions import ProblemDefinitionError
from surrox.problem.types import ConstraintOperator, ConstraintSeverity


class LinearConstraint(BaseModel):
    """An analytical linear constraint on decision variables.

    Enforces `sum(coeff_i * x_i) operator rhs` where coefficients map
    decision variable names to their weights.

    Attributes:
        name: Unique name for this constraint.
        coefficients: Mapping of decision variable names to their coefficients. Must be non-empty with no zero values.
        operator: Comparison operator (le, ge, eq).
        rhs: Right-hand side value.
        severity: Hard constraints must be satisfied; soft constraints are penalized.
        penalty_weight: Required (and positive) for soft constraints, must be None for hard.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    coefficients: dict[str, float]
    operator: ConstraintOperator
    rhs: float
    tolerance: float | None = None
    severity: ConstraintSeverity = ConstraintSeverity.HARD
    penalty_weight: float | None = None

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

    @model_validator(mode="after")
    def _validate_tolerance(self) -> "LinearConstraint":
        if self.operator == ConstraintOperator.EQ and self.tolerance is None:
            raise ProblemDefinitionError("tolerance is required when operator is EQ")
        if self.operator != ConstraintOperator.EQ and self.tolerance is not None:
            raise ProblemDefinitionError(
                "tolerance must be None when operator is not EQ"
            )
        if self.tolerance is not None and self.tolerance <= 0:
            raise ProblemDefinitionError("tolerance must be positive")
        return self

    @model_validator(mode="after")
    def _validate_severity(self) -> "LinearConstraint":
        if self.severity == ConstraintSeverity.HARD and self.penalty_weight is not None:
            raise ProblemDefinitionError(
                "penalty_weight must be None when severity is HARD"
            )
        if self.severity == ConstraintSeverity.SOFT and self.penalty_weight is None:
            raise ProblemDefinitionError(
                "penalty_weight is required when severity is SOFT"
            )
        if self.penalty_weight is not None and self.penalty_weight <= 0:
            raise ProblemDefinitionError("penalty_weight must be positive")
        return self


class DataConstraint(BaseModel):
    """A surrogate-based constraint evaluated on a predicted dataset column.

    Enforces `prediction(column) operator limit`. A surrogate model is trained
    for the target column, and the constraint is evaluated on its predictions.

    Attributes:
        name: Unique name for this constraint.
        column: Dataset column that the surrogate predicts.
        operator: Comparison operator (le, ge, eq).
        limit: Threshold value.
        tolerance: Required (and positive) for eq operator, must be None otherwise.
        severity: Hard constraints must be satisfied; soft constraints are penalized.
        penalty_weight: Required (and positive) for soft constraints, must be None for hard.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    column: str
    operator: ConstraintOperator
    limit: float
    tolerance: float | None = None
    severity: ConstraintSeverity = ConstraintSeverity.HARD
    penalty_weight: float | None = None

    @model_validator(mode="after")
    def _validate_tolerance(self) -> "DataConstraint":
        if self.operator == ConstraintOperator.EQ and self.tolerance is None:
            raise ProblemDefinitionError("tolerance is required when operator is EQ")
        if self.operator != ConstraintOperator.EQ and self.tolerance is not None:
            raise ProblemDefinitionError(
                "tolerance must be None when operator is not EQ"
            )
        if self.tolerance is not None and self.tolerance <= 0:
            raise ProblemDefinitionError("tolerance must be positive")
        return self

    @model_validator(mode="after")
    def _validate_severity(self) -> "DataConstraint":
        if self.severity == ConstraintSeverity.HARD and self.penalty_weight is not None:
            raise ProblemDefinitionError(
                "penalty_weight must be None when severity is HARD"
            )
        if self.severity == ConstraintSeverity.SOFT and self.penalty_weight is None:
            raise ProblemDefinitionError(
                "penalty_weight is required when severity is SOFT"
            )
        if self.penalty_weight is not None and self.penalty_weight <= 0:
            raise ProblemDefinitionError("penalty_weight must be positive")
        return self
