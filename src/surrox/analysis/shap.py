from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class FeatureImportanceResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    column: str
    importances: dict[str, float]
    decision_importances: dict[str, float]


class ShapGlobalResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    column: str
    feature_names: tuple[str, ...]
    shap_values: NDArray
    base_value: float
    feature_values: NDArray


class ShapLocalResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    column: str
    feature_names: tuple[str, ...]
    shap_values: NDArray
    base_value: float
    feature_values: dict[str, float]
    predicted_value: float
