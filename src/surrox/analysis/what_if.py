from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class WhatIfPrediction(BaseModel):
    """Prediction for a single objective or constraint in a what-if scenario.

    Attributes:
        predicted: Point prediction from the surrogate ensemble.
        lower: Lower conformal prediction interval bound.
        upper: Upper conformal prediction interval bound.
        recommended_value: Prediction at the recommended (compromise) point for comparison.
        historical_mean: Mean of the historical data for this column.
    """

    model_config = ConfigDict(frozen=True)

    predicted: float
    lower: float
    upper: float
    recommended_value: float
    historical_mean: float


class WhatIfResult(BaseModel):
    """Result of a what-if prediction for a hypothetical variable setting.

    Attributes:
        variables: The hypothetical variable values that were evaluated.
        objectives: Predictions per objective.
        constraints: Predictions per data constraint.
        extrapolation_distance: Distance to the training data manifold.
        is_extrapolating: Whether this point is outside the training domain.
    """

    model_config = ConfigDict(frozen=True)

    variables: dict[str, Any]
    objectives: dict[str, WhatIfPrediction]
    constraints: dict[str, WhatIfPrediction]
    extrapolation_distance: float
    is_extrapolating: bool
