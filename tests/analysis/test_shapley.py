import numpy as np
import pandas as pd
import pytest

from surrox.analysis.shapley import shapley_values

FEATURE_NAMES = ("a", "b", "c", "d")


def _heterogeneous_predict(df: pd.DataFrame) -> np.ndarray:
    x = df[list(FEATURE_NAMES)].to_numpy(dtype=float)
    return 2 * x[:, 0] - 3 * x[:, 1] + np.sin(x[:, 2]) + x[:, 3] ** 2


@pytest.fixture
def data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    background = pd.DataFrame(rng.normal(size=(40, 4)), columns=FEATURE_NAMES)
    instances = pd.DataFrame(rng.normal(size=(3, 4)), columns=FEATURE_NAMES)
    return instances, background


class TestExactShapley:
    def test_efficiency_property(self, data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        instances, background = data
        result = shapley_values(
            predict=_heterogeneous_predict,
            instances=instances,
            background=background,
            feature_names=FEATURE_NAMES,
            exact_threshold=12,
            sampling_permutations=1,
            rng=np.random.default_rng(0),
        )
        reconstructed = result.base_value + result.shap_values.sum(axis=1)
        np.testing.assert_allclose(
            reconstructed, _heterogeneous_predict(instances), atol=1e-9
        )

    def test_no_standard_error_when_exact(
        self, data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        instances, background = data
        result = shapley_values(
            predict=_heterogeneous_predict,
            instances=instances,
            background=background,
            feature_names=FEATURE_NAMES,
            exact_threshold=12,
            sampling_permutations=1,
            rng=np.random.default_rng(0),
        )
        assert result.standard_error is None

    def test_base_value_is_background_mean(
        self, data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        instances, background = data
        result = shapley_values(
            predict=_heterogeneous_predict,
            instances=instances,
            background=background,
            feature_names=FEATURE_NAMES,
            exact_threshold=12,
            sampling_permutations=1,
            rng=np.random.default_rng(0),
        )
        assert result.base_value == pytest.approx(
            float(_heterogeneous_predict(background).mean())
        )


class TestSampledShapley:
    def test_efficiency_property(self, data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        instances, background = data
        result = shapley_values(
            predict=_heterogeneous_predict,
            instances=instances,
            background=background,
            feature_names=FEATURE_NAMES,
            exact_threshold=2,
            sampling_permutations=200,
            rng=np.random.default_rng(0),
        )
        reconstructed = result.base_value + result.shap_values.sum(axis=1)
        np.testing.assert_allclose(
            reconstructed, _heterogeneous_predict(instances), atol=1e-9
        )

    def test_converges_to_exact(self, data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        instances, background = data
        exact = shapley_values(
            predict=_heterogeneous_predict,
            instances=instances,
            background=background,
            feature_names=FEATURE_NAMES,
            exact_threshold=12,
            sampling_permutations=1,
            rng=np.random.default_rng(0),
        )
        sampled = shapley_values(
            predict=_heterogeneous_predict,
            instances=instances,
            background=background,
            feature_names=FEATURE_NAMES,
            exact_threshold=2,
            sampling_permutations=400,
            rng=np.random.default_rng(0),
        )
        np.testing.assert_allclose(sampled.shap_values, exact.shap_values, atol=1e-2)

    def test_reports_standard_error(
        self, data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        instances, background = data
        result = shapley_values(
            predict=_heterogeneous_predict,
            instances=instances,
            background=background,
            feature_names=FEATURE_NAMES,
            exact_threshold=2,
            sampling_permutations=50,
            rng=np.random.default_rng(0),
        )
        assert result.standard_error is not None
        assert result.standard_error.shape == result.shap_values.shape
        assert np.all(result.standard_error >= 0)
