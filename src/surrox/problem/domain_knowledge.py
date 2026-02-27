from pydantic import BaseModel, ConfigDict

from surrox.problem.types import MonotonicDirection


class MonotonicRelation(BaseModel):
    """A known monotonic relationship between a decision variable and a target.

    Encodes domain knowledge that increasing a decision variable always increases
    (or decreases) a specific objective or data constraint prediction. Used to
    constrain surrogate model training.

    Attributes:
        decision_variable: Name of the numeric decision variable.
        objective_or_constraint: Name of the objective or data constraint.
        direction: Whether the relationship is increasing or decreasing.
    """

    model_config = ConfigDict(frozen=True)

    decision_variable: str
    objective_or_constraint: str
    direction: MonotonicDirection
