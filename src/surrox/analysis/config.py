from pydantic import BaseModel, ConfigDict, model_validator

from surrox.exceptions import AnalysisError


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    shap_background_size: int = 100
    pdp_grid_resolution: int = 50
    pdp_percentiles: tuple[float, float] = (0.05, 0.95)
    monotonicity_check_resolution: int = 50

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
