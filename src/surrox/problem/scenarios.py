from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from surrox.exceptions import ProblemDefinitionError


class Scenario(BaseModel):
    """A named set of fixed context variable values for scenario-based optimization.

    Scenarios fix context variables to specific values, enabling what-if comparisons
    across different operating conditions.

    Attributes:
        name: Unique name for this scenario.
        context_values: Mapping of context variable names to their fixed values.
            Must contain at least one entry.
    """

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
