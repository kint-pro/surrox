from pydantic import BaseModel, ConfigDict, model_validator

from surrox.exceptions import ProblemDefinitionError
from surrox.problem.types import Direction


class Objective(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    direction: Direction
    column: str
    reference_value: float | None = None
    prediction_lower: float | None = None
    prediction_upper: float | None = None

    @model_validator(mode="after")
    def _validate_prediction_bounds(self) -> "Objective":
        if (
            self.prediction_lower is not None
            and self.prediction_upper is not None
            and self.prediction_lower >= self.prediction_upper
        ):
            raise ProblemDefinitionError(
                f"objective '{self.name}': prediction_lower ({self.prediction_lower}) "
                f"must be less than prediction_upper ({self.prediction_upper})"
            )
        return self
