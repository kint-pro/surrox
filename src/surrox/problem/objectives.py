from pydantic import BaseModel, ConfigDict

from surrox.problem.types import Direction


class Objective(BaseModel):
    """An optimization objective targeting a dataset column.

    Attributes:
        name: Unique name for this objective.
        direction: Whether to minimize or maximize.
        column: Dataset column that the surrogate predicts.
        reference_value: Optional baseline value for comparison.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    direction: Direction
    column: str
    reference_value: float | None = None
