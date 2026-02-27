import pytest

from surrox.analysis.config import AnalysisConfig
from surrox.exceptions import AnalysisError


class TestAnalysisConfig:
    def test_defaults(self) -> None:
        config = AnalysisConfig()
        assert config.shap_background_size == 100
        assert config.pdp_grid_resolution == 50
        assert config.pdp_percentiles == (0.05, 0.95)
        assert config.monotonicity_check_resolution == 50

    def test_custom_values(self) -> None:
        config = AnalysisConfig(
            shap_background_size=200,
            pdp_grid_resolution=100,
            pdp_percentiles=(0.1, 0.9),
            monotonicity_check_resolution=25,
        )
        assert config.shap_background_size == 200

    def test_shap_background_size_too_small(self) -> None:
        with pytest.raises(AnalysisError, match="shap_background_size"):
            AnalysisConfig(shap_background_size=5)

    def test_pdp_grid_resolution_too_small(self) -> None:
        with pytest.raises(AnalysisError, match="pdp_grid_resolution"):
            AnalysisConfig(pdp_grid_resolution=5)

    def test_monotonicity_check_resolution_too_small(self) -> None:
        with pytest.raises(AnalysisError, match="monotonicity_check_resolution"):
            AnalysisConfig(monotonicity_check_resolution=5)

    def test_pdp_percentiles_invalid_order(self) -> None:
        with pytest.raises(AnalysisError, match="pdp_percentiles"):
            AnalysisConfig(pdp_percentiles=(0.9, 0.1))

    def test_pdp_percentiles_zero_lower(self) -> None:
        with pytest.raises(AnalysisError, match="pdp_percentiles"):
            AnalysisConfig(pdp_percentiles=(0.0, 0.9))

    def test_pdp_percentiles_one_upper(self) -> None:
        with pytest.raises(AnalysisError, match="pdp_percentiles"):
            AnalysisConfig(pdp_percentiles=(0.1, 1.0))

    def test_is_frozen(self) -> None:
        config = AnalysisConfig()
        with pytest.raises(Exception):
            config.shap_background_size = 200  # type: ignore[misc]
