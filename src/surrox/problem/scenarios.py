from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class Scenario(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    context_values: dict[str, Any]

    @model_validator(mode="after")
    def _validate_non_empty(self) -> "Scenario":
        if not self.context_values:
            raise ValueError("scenario must define at least one context variable value")
        return self
