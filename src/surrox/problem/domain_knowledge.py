from pydantic import BaseModel, ConfigDict

from surrox.problem.types import MonotonicDirection


class MonotonicRelation(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision_variable: str
    objective_or_constraint: str
    direction: MonotonicDirection
