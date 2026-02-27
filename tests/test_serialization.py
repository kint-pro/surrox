from __future__ import annotations

import numpy as np

from surrox.analysis.pdp import PDPICEResult
from surrox.analysis.shap import (
    FeatureImportanceResult,
    ShapGlobalResult,
    ShapLocalResult,
)
from surrox.analysis.trade_off import TradeOffResult
from surrox.surrogate.models import SurrogatePrediction


class TestNumpyArraySerialization:
    def test_shap_global_roundtrip(self) -> None:
        original = ShapGlobalResult(
            column="cost",
            feature_names=("x1", "x2"),
            shap_values=np.array([[0.1, 0.2], [0.3, 0.4]]),
            base_value=1.5,
            feature_values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        )
        json_str = original.model_dump_json()
        loaded = ShapGlobalResult.model_validate_json(json_str)

        np.testing.assert_array_almost_equal(loaded.shap_values, original.shap_values)
        np.testing.assert_array_almost_equal(
            loaded.feature_values, original.feature_values,
        )
        assert loaded.column == original.column
        assert loaded.base_value == original.base_value

    def test_shap_local_roundtrip(self) -> None:
        original = ShapLocalResult(
            column="cost",
            feature_names=("x1", "x2"),
            shap_values=np.array([0.5, -0.3]),
            base_value=2.0,
            feature_values={"x1": 1.0, "x2": 2.0},
            predicted_value=2.2,
        )
        json_str = original.model_dump_json()
        loaded = ShapLocalResult.model_validate_json(json_str)

        np.testing.assert_array_almost_equal(loaded.shap_values, original.shap_values)
        assert loaded.feature_values == original.feature_values

    def test_pdp_ice_roundtrip(self) -> None:
        original = PDPICEResult(
            variable_name="x1",
            column="cost",
            grid_values=np.linspace(0, 10, 5),
            pdp_values=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            ice_values=np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [1.5, 2.5, 3.5, 4.5, 5.5]]),
        )
        json_str = original.model_dump_json()
        loaded = PDPICEResult.model_validate_json(json_str)

        np.testing.assert_array_almost_equal(loaded.grid_values, original.grid_values)
        np.testing.assert_array_almost_equal(loaded.pdp_values, original.pdp_values)
        np.testing.assert_array_almost_equal(loaded.ice_values, original.ice_values)

    def test_trade_off_roundtrip(self) -> None:
        original = TradeOffResult(
            objective_pairs=(("f1", "f2"),),
            marginal_rates={("f1", "f2"): np.array([0.5, 1.0, 1.5])},
            pareto_objectives=np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]]),
        )
        json_str = original.model_dump_json()
        loaded = TradeOffResult.model_validate_json(json_str)

        assert loaded.objective_pairs == original.objective_pairs
        np.testing.assert_array_almost_equal(
            loaded.marginal_rates[("f1", "f2")],
            original.marginal_rates[("f1", "f2")],
        )
        np.testing.assert_array_almost_equal(
            loaded.pareto_objectives, original.pareto_objectives
        )

    def test_surrogate_prediction_roundtrip(self) -> None:
        original = SurrogatePrediction(
            mean=np.array([1.0, 2.0, 3.0]),
            std=np.array([0.1, 0.2, 0.3]),
            lower=np.array([0.8, 1.6, 2.4]),
            upper=np.array([1.2, 2.4, 3.6]),
        )
        json_str = original.model_dump_json()
        loaded = SurrogatePrediction.model_validate_json(json_str)

        np.testing.assert_array_almost_equal(loaded.mean, original.mean)
        np.testing.assert_array_almost_equal(loaded.std, original.std)
        np.testing.assert_array_almost_equal(loaded.lower, original.lower)
        np.testing.assert_array_almost_equal(loaded.upper, original.upper)

    def test_feature_importance_roundtrip(self) -> None:
        original = FeatureImportanceResult(
            column="cost",
            importances={"x1": 0.7, "x2": 0.3},
            decision_importances={"x1": 0.8, "x2": 0.2},
        )
        json_str = original.model_dump_json()
        loaded = FeatureImportanceResult.model_validate_json(json_str)

        assert loaded.importances == original.importances
        assert loaded.decision_importances == original.decision_importances

    def test_trade_off_multiple_pairs(self) -> None:
        original = TradeOffResult(
            objective_pairs=(("f1", "f2"), ("f1", "f3")),
            marginal_rates={
                ("f1", "f2"): np.array([0.5, 1.0]),
                ("f1", "f3"): np.array([2.0, 3.0]),
            },
            pareto_objectives=np.array([[0.1, 0.5, 0.9], [0.9, 0.5, 0.1]]),
        )
        json_str = original.model_dump_json()
        loaded = TradeOffResult.model_validate_json(json_str)

        assert set(loaded.marginal_rates.keys()) == {("f1", "f2"), ("f1", "f3")}
        np.testing.assert_array_almost_equal(
            loaded.marginal_rates[("f1", "f3")],
            original.marginal_rates[("f1", "f3")],
        )
