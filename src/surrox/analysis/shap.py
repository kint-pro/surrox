from pydantic import BaseModel, ConfigDict

from surrox.types import NumpyArray


class FeatureImportanceResult(BaseModel):
    """Mean absolute SHAP-based feature importance for a target column.

    Attributes:
        column: Target column name.
        importances: Mean |SHAP| per feature (all variables).
        decision_importances: Mean |SHAP| per feature (decision variables only).
    """

    model_config = ConfigDict(frozen=True)

    column: str
    importances: dict[str, float]
    decision_importances: dict[str, float]


class ShapGlobalResult(BaseModel):
    """Global SHAP explanation over a background dataset.

    Attributes:
        column: Target column name.
        feature_names: Feature names in column order.
        shap_values: SHAP value matrix, shape (n_samples, n_features).
        base_value: Expected model output (ensemble-weighted).
        feature_values: Feature value matrix, shape (n_samples, n_features).
    """

    model_config = ConfigDict(frozen=True)

    column: str
    feature_names: tuple[str, ...]
    shap_values: NumpyArray
    base_value: float
    feature_values: NumpyArray


class ShapLocalResult(BaseModel):
    """Local SHAP explanation for a single Pareto-optimal point.

    Attributes:
        column: Target column name.
        feature_names: Feature names in column order.
        shap_values: SHAP values for this point, shape (n_features,).
        base_value: Expected model output (ensemble-weighted).
        feature_values: Feature values at this point.
        predicted_value: Model prediction (base_value + sum of SHAP values).
    """

    model_config = ConfigDict(frozen=True)

    column: str
    feature_names: tuple[str, ...]
    shap_values: NumpyArray
    base_value: float
    feature_values: dict[str, float | int | str]
    predicted_value: float
