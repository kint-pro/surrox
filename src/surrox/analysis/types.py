from enum import StrEnum


class ConstraintStatusKind(StrEnum):
    SATISFIED = "satisfied"
    ACTIVE = "active"
    VIOLATED = "violated"
