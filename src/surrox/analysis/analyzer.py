from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from surrox.analysis.pdp import PDPICEResult
from surrox.analysis.shap import (
    FeatureImportanceResult,
    ShapGlobalResult,
    ShapLocalResult,
)
from surrox.analysis.trade_off import TradeOffResult
from surrox.analysis.what_if import WhatIfPrediction, WhatIfResult
from surrox.exceptions import AnalysisError
from surrox.optimizer.extrapolation import ExtrapolationGate

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd

    from surrox.analysis.config import AnalysisConfig
    from surrox.optimizer.result import EvaluatedPoint, OptimizationResult
    from surrox.problem.dataset import BoundDataset
    from surrox.problem.definition import ProblemDefinition
    from surrox.surrogate.manager import SurrogateManager


@dataclass(frozen=True)
class _ShapExplanation:
    shap_values: NDArray[np.floating[Any]]
    base_value: float


def _encode_feature_values(X: pd.DataFrame) -> NDArray[np.floating]:
    import pandas as pd

    result = X.copy()
    for col in result.columns:
        if isinstance(result[col].dtype, pd.CategoricalDtype):
            result[col] = result[col].cat.codes.astype(np.float64)
    return result.to_numpy(dtype=np.float64)


def _explain_tree(model: Any, X: pd.DataFrame) -> _ShapExplanation:
    import shap

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)
    shap_values = np.asarray(explanation.values)  # pyright: ignore[reportUnknownMemberType]
    raw_base: Any = explanation.base_values  # pyright: ignore[reportUnknownMemberType]
    base_value = float(raw_base[0]) if hasattr(raw_base, "__len__") else float(raw_base)
    return _ShapExplanation(shap_values=shap_values, base_value=base_value)


class Analyzer:
    """On-demand detail analysis engine with lazy computation and caching.

    Provides SHAP explanations (global and local), feature importance,
    PDP/ICE curves, trade-off analysis, and what-if predictions.
    Results are cached after first computation.
    """

    def __init__(
        self,
        optimization_result: OptimizationResult,
        surrogate_manager: SurrogateManager,
        bound_dataset: BoundDataset,
        config: AnalysisConfig,
    ) -> None:
        self._opt_result = optimization_result
        self._surrogate_manager = surrogate_manager
        self._bound_dataset = bound_dataset
        self._config = config
        self._cache: dict[tuple[str, ...], Any] = {}

    @property
    def _problem(self) -> ProblemDefinition:
        return self._opt_result.problem

    def _validate_column(self, column: str) -> None:
        if column not in self._problem.surrogate_columns:
            raise AnalysisError(
                f"column '{column}' not found in surrogate columns: "
                f"{self._problem.surrogate_columns}"
            )

    def _get_recommended(self) -> EvaluatedPoint:
        if not self._opt_result.has_feasible_solutions:
            raise AnalysisError("no feasible solutions available")
        if self._opt_result.compromise_index is not None:
            return self._opt_result.feasible_points[self._opt_result.compromise_index]
        return self._opt_result.feasible_points[0]

    def _get_background_data(self) -> pd.DataFrame:
        df = self._bound_dataset.dataframe
        n = min(len(df), self._config.shap_background_size)
        if n < len(df):
            return df.sample(n=n, random_state=self._config.random_seed)
        return df

    def shap_global(self, column: str) -> ShapGlobalResult:
        self._validate_column(column)
        cache_key = ("shap_global", column)
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._compute_shap_global(column)
        self._cache[cache_key] = result
        return result

    def _compute_shap_global(self, column: str) -> ShapGlobalResult:
        _logger.debug("computing shap_global", extra={"column": column})
        ensemble = self._surrogate_manager.get_ensemble(column)
        background = self._get_background_data()
        feature_names = ensemble.feature_names
        X = ensemble._prepare_features(background)

        all_shap_values = np.zeros((len(X), len(feature_names)))
        base_value = 0.0

        for member in ensemble.members:
            expl = _explain_tree(member.model, X)
            all_shap_values += member.weight * expl.shap_values
            base_value += member.weight * expl.base_value

        feature_values = _encode_feature_values(X)

        return ShapGlobalResult(
            column=column,
            feature_names=feature_names,
            shap_values=all_shap_values,
            base_value=base_value,
            feature_values=feature_values,
        )

    def shap_local(self, column: str, point_index: int) -> ShapLocalResult:
        self._validate_column(column)
        if not self._opt_result.has_feasible_solutions:
            raise AnalysisError("no feasible solutions available")
        if point_index < 0 or point_index >= len(self._opt_result.feasible_points):
            raise AnalysisError(
                f"point_index {point_index} out of range "
                f"[0, {len(self._opt_result.feasible_points)})"
            )

        cache_key = ("shap_local", column, str(point_index))
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._compute_shap_local(column, point_index)
        self._cache[cache_key] = result
        return result

    def _compute_shap_local(self, column: str, point_index: int) -> ShapLocalResult:
        import pandas as pd

        _logger.debug(
            "computing shap_local",
            extra={"column": column, "point_index": point_index},
        )
        ensemble = self._surrogate_manager.get_ensemble(column)
        feature_names = ensemble.feature_names
        point = self._opt_result.feasible_points[point_index]

        row_data = dict(point.variables)
        df = pd.DataFrame([row_data])
        X = ensemble._prepare_features(df)

        shap_values = np.zeros(len(feature_names))
        base_value = 0.0

        for member in ensemble.members:
            expl = _explain_tree(member.model, X)
            shap_values += member.weight * expl.shap_values[0]
            base_value += member.weight * expl.base_value

        predicted_value = base_value + float(np.sum(shap_values))
        feature_values = {name: row_data[name] for name in feature_names}

        return ShapLocalResult(
            column=column,
            feature_names=feature_names,
            shap_values=shap_values,
            base_value=base_value,
            feature_values=feature_values,
            predicted_value=predicted_value,
        )

    def feature_importance(self, column: str) -> FeatureImportanceResult:
        global_result = self.shap_global(column)
        mean_abs = np.abs(global_result.shap_values).mean(axis=0)

        importances = {
            name: float(mean_abs[i])
            for i, name in enumerate(global_result.feature_names)
        }

        decision_names = {v.name for v in self._problem.decision_variables}
        decision_importances = {
            name: val for name, val in importances.items() if name in decision_names
        }

        return FeatureImportanceResult(
            column=column,
            importances=importances,
            decision_importances=decision_importances,
        )

    def pdp_ice(self, variable_name: str, column: str) -> PDPICEResult:
        self._validate_column(column)
        decision_names = {v.name for v in self._problem.decision_variables}
        if variable_name not in decision_names:
            raise AnalysisError(f"'{variable_name}' is not a decision variable")

        cache_key = ("pdp_ice", variable_name, column)
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._compute_pdp_ice(variable_name, column)
        self._cache[cache_key] = result
        return result

    def _compute_pdp_ice(self, variable_name: str, column: str) -> PDPICEResult:
        _logger.debug(
            "computing pdp_ice",
            extra={"variable": variable_name, "column": column},
        )
        from sklearn.inspection import partial_dependence

        ensemble = self._surrogate_manager.get_ensemble(column)
        background = self._get_background_data()
        feature_names = list(ensemble.feature_names)
        X = ensemble._prepare_features(background)

        feature_idx = feature_names.index(variable_name)

        all_pdp: list[np.ndarray] = []
        all_ice: list[np.ndarray] = []

        for member in ensemble.members:
            result = partial_dependence(
                member.model,
                X,
                features=[feature_idx],
                kind="both",
                method="brute",
                grid_resolution=self._config.pdp_grid_resolution,
            )
            grid_values = result["grid_values"][0]
            pdp_avg = result["average"][0]
            ice_lines = result["individual"][0]

            all_pdp.append(member.weight * pdp_avg)
            all_ice.append(member.weight * ice_lines)

        pdp_values = np.sum(all_pdp, axis=0)
        ice_values = np.sum(all_ice, axis=0)

        return PDPICEResult(
            variable_name=variable_name,
            column=column,
            grid_values=grid_values,
            pdp_values=pdp_values,
            ice_values=ice_values,
        )

    def trade_off(self) -> TradeOffResult:
        cache_key = ("trade_off",)
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._compute_trade_off()
        self._cache[cache_key] = result
        return result

    def _compute_trade_off(self) -> TradeOffResult:
        _logger.debug("computing trade_off")
        objectives = self._problem.objectives
        if len(objectives) < 2:
            raise AnalysisError("trade-off analysis requires at least 2 objectives")

        feasible = self._opt_result.feasible_points
        if len(feasible) < 2:
            raise AnalysisError(
                "trade-off analysis requires at least 2 feasible points"
            )

        obj_names = [o.name for o in objectives]
        pareto_values = np.array(
            [[p.objectives[name] for name in obj_names] for p in feasible]
        )

        sort_idx = np.argsort(pareto_values[:, 0])
        pareto_values = pareto_values[sort_idx]

        pairs: list[tuple[str, str]] = []
        rates: dict[tuple[str, str], np.ndarray] = {}

        for i in range(len(obj_names)):
            for j in range(i + 1, len(obj_names)):
                pair = (obj_names[i], obj_names[j])
                pairs.append(pair)
                delta_a = np.diff(pareto_values[:, i])
                delta_b = np.diff(pareto_values[:, j])
                with np.errstate(divide="ignore", invalid="ignore"):
                    marginal = np.where(
                        np.abs(delta_a) > 1e-12,
                        delta_b / delta_a,
                        np.inf * np.sign(delta_b),
                    )
                rates[pair] = marginal

        return TradeOffResult(
            objective_pairs=tuple(pairs),
            marginal_rates=rates,
            pareto_objectives=pareto_values,
        )

    def what_if(self, variable_values: dict[str, Any]) -> WhatIfResult:
        return self._compute_what_if(variable_values)

    def _compute_what_if(self, variable_values: dict[str, Any]) -> WhatIfResult:
        _logger.debug("computing what_if", extra={"variables": variable_values})
        import pandas as pd

        recommended = self._get_recommended()
        df = pd.DataFrame([variable_values])

        predictions = self._surrogate_manager.evaluate_with_uncertainty(df)

        recommended_df = pd.DataFrame([recommended.variables])
        recommended_predictions = self._surrogate_manager.evaluate_with_uncertainty(
            recommended_df
        )

        historical_df = self._bound_dataset.dataframe

        objectives: dict[str, WhatIfPrediction] = {}
        for obj in self._problem.objectives:
            col = obj.column
            pred = predictions[col]
            rec_pred = recommended_predictions[col]
            objectives[obj.name] = WhatIfPrediction(
                predicted=float(pred.mean[0]),
                lower=float(pred.lower[0]),
                upper=float(pred.upper[0]),
                recommended_value=float(rec_pred.mean[0]),
                historical_mean=float(historical_df[col].mean().item()),
            )

        constraints: dict[str, WhatIfPrediction] = {}
        for dc in self._problem.data_constraints:
            col = dc.column
            pred = predictions[col]
            rec_pred = recommended_predictions[col]
            constraints[dc.name] = WhatIfPrediction(
                predicted=float(pred.mean[0]),
                lower=float(pred.lower[0]),
                upper=float(pred.upper[0]),
                recommended_value=float(rec_pred.mean[0]),
                historical_mean=float(historical_df[col].mean().item()),
            )

        gate = ExtrapolationGate(
            training_data=historical_df,
            decision_variables=self._problem.decision_variables,
            k=5,
            threshold=2.0,
        )
        is_extrapolating_arr, distances = gate.evaluate(df)
        extrapolation_distance = float(distances[0])
        is_extrapolating = bool(is_extrapolating_arr[0])

        return WhatIfResult(
            variables=variable_values,
            objectives=objectives,
            constraints=constraints,
            extrapolation_distance=extrapolation_distance,
            is_extrapolating=is_extrapolating,
        )
