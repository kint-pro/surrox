from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class WhatIfPrediction(BaseModel):
    model_config = ConfigDict(frozen=True)

    predicted: float
    lower: float
    upper: float
    recommended_value: float
    historical_mean: float


class WhatIfResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    variables: dict[str, Any]
    objectives: dict[str, WhatIfPrediction]
    constraints: dict[str, WhatIfPrediction]
    extrapolation_distance: float
    is_extrapolating: bool
