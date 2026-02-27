from enum import StrEnum


class DType(StrEnum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"


class Role(StrEnum):
    DECISION = "decision"
    CONTEXT = "context"


class Direction(StrEnum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class MonotonicDirection(StrEnum):
    INCREASING = "increasing"
    DECREASING = "decreasing"


class ConstraintOperator(StrEnum):
    LE = "le"
    GE = "ge"
    EQ = "eq"


class ConstraintSeverity(StrEnum):
    HARD = "hard"
    SOFT = "soft"
