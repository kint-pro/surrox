from pydantic import BaseModel, ConfigDict

from surrox.types import NumpyArray


class FeatureImportanceResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    column: str
    importances: dict[str, float]
    decision_importances: dict[str, float]


class ShapGlobalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    column: str
    feature_names: tuple[str, ...]
    shap_values: NumpyArray
    base_value: float
    feature_values: NumpyArray


class ShapLocalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    column: str
    feature_names: tuple[str, ...]
    shap_values: NumpyArray
    base_value: float
    feature_values: dict[str, float]
    predicted_value: float
