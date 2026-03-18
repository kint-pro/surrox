from enum import StrEnum


class DType(StrEnum):
    """Data type of a variable."""

    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"


class Role(StrEnum):
    """Role of a variable in the optimization problem."""

    DECISION = "decision"
    CONTEXT = "context"


class Direction(StrEnum):
    """Optimization direction for an objective."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class MonotonicDirection(StrEnum):
    """Direction of a monotonic relationship between a variable and a target."""

    INCREASING = "increasing"
    DECREASING = "decreasing"


class ConstraintOperator(StrEnum):
    """Comparison operator for constraints."""

    LE = "le"
    GE = "ge"
    EQ = "eq"


class ConstraintSeverity(StrEnum):
    """Severity level of a constraint. Hard constraints must be satisfied, soft constraints are penalized."""

    HARD = "hard"
    SOFT = "soft"
