from pydantic import BaseModel, ConfigDict, model_validator

from surrox.exceptions import AnalysisError


class AnalysisConfig(BaseModel):
    """Configuration for post-optimization analysis.

    Attributes:
        shap_background_size: Number of background samples for SHAP explanations.
        pdp_grid_resolution: Number of grid points for PDP/ICE plots.
        pdp_percentiles: Lower and upper percentile bounds for PDP grid range.
        monotonicity_check_resolution: Grid resolution for monotonicity verification.
        random_seed: Random seed for reproducibility.
    """

    model_config = ConfigDict(frozen=True)

    shap_background_size: int = 100
    pdp_grid_resolution: int = 50
    pdp_percentiles: tuple[float, float] = (0.05, 0.95)
    monotonicity_check_resolution: int = 50
    random_seed: int = 42

    @model_validator(mode="after")
    def _validate_config(self) -> "AnalysisConfig":
        if self.shap_background_size < 10:
            raise AnalysisError("shap_background_size must be >= 10")
        if self.pdp_grid_resolution < 10:
            raise AnalysisError("pdp_grid_resolution must be >= 10")
        if self.monotonicity_check_resolution < 10:
            raise AnalysisError("monotonicity_check_resolution must be >= 10")
        lo, hi = self.pdp_percentiles
        if not (0 < lo < hi < 1):
            raise AnalysisError("pdp_percentiles must satisfy 0 < lower < upper < 1")
        return self
