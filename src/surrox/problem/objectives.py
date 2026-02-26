from pydantic import BaseModel, ConfigDict

from surrox.problem.types import Direction


class Objective(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    direction: Direction
    column: str
    reference_value: float | None = None
