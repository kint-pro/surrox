from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from surrox.exceptions import ProblemDefinitionError


class Scenario(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    context_values: dict[str, Any]

    @model_validator(mode="after")
    def _validate_non_empty(self) -> "Scenario":
        if not self.context_values:
            raise ProblemDefinitionError(
                "scenario must define at least one context variable value"
            )
        return self
