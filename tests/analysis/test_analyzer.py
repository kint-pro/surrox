from unittest.mock import MagicMock

import pytest

from surrox.analysis.analyzer import Analyzer
from surrox.analysis.config import AnalysisConfig
from surrox.exceptions import AnalysisError
from surrox.optimizer.result import OptimizationResult
from surrox.problem.dataset import BoundDataset


class TestAnalyzerValidation:
    def test_invalid_column_raises(
        self,
        single_objective_result: OptimizationResult,
        mock_surrogate_single: MagicMock,
        bound_dataset_single: BoundDataset,
        analysis_config: AnalysisConfig,
    ) -> None:
        analyzer = Analyzer(
            single_objective_result,
            mock_surrogate_single,
            bound_dataset_single,
            analysis_config,
        )
        with pytest.raises(AnalysisError, match="not found in surrogate"):
            analyzer.shap_global("nonexistent_column")

    def test_trade_off_single_objective_raises(
        self,
        single_objective_result: OptimizationResult,
        mock_surrogate_single: MagicMock,
        bound_dataset_single: BoundDataset,
        analysis_config: AnalysisConfig,
    ) -> None:
        analyzer = Analyzer(
            single_objective_result,
            mock_surrogate_single,
            bound_dataset_single,
            analysis_config,
        )
        with pytest.raises(AnalysisError, match="at least 2 objectives"):
            analyzer.trade_off()

    def test_pdp_ice_non_decision_variable_raises(
        self,
        single_objective_result: OptimizationResult,
        mock_surrogate_single: MagicMock,
        bound_dataset_single: BoundDataset,
        analysis_config: AnalysisConfig,
    ) -> None:
        analyzer = Analyzer(
            single_objective_result,
            mock_surrogate_single,
            bound_dataset_single,
            analysis_config,
        )
        with pytest.raises(AnalysisError, match="not a decision variable"):
            analyzer.pdp_ice("nonexistent", "cost")

    def test_shap_local_invalid_index_raises(
        self,
        single_objective_result: OptimizationResult,
        mock_surrogate_single: MagicMock,
        bound_dataset_single: BoundDataset,
        analysis_config: AnalysisConfig,
    ) -> None:
        analyzer = Analyzer(
            single_objective_result,
            mock_surrogate_single,
            bound_dataset_single,
            analysis_config,
        )
        with pytest.raises(AnalysisError, match="point_index"):
            analyzer.shap_local("cost", 999)

    def test_no_feasible_raises_for_what_if(
        self,
        no_feasible_result: OptimizationResult,
        mock_surrogate_single: MagicMock,
        bound_dataset_single: BoundDataset,
        analysis_config: AnalysisConfig,
    ) -> None:
        analyzer = Analyzer(
            no_feasible_result,
            mock_surrogate_single,
            bound_dataset_single,
            analysis_config,
        )
        with pytest.raises(AnalysisError, match="no feasible"):
            analyzer.what_if({"x1": 5.0, "x2": 5.0})


class TestAnalyzerTradeOff:
    def test_trade_off_multi_objective(
        self,
        multi_objective_result: OptimizationResult,
        mock_surrogate_multi: MagicMock,
        bound_dataset_multi: BoundDataset,
        analysis_config: AnalysisConfig,
    ) -> None:
        analyzer = Analyzer(
            multi_objective_result,
            mock_surrogate_multi,
            bound_dataset_multi,
            analysis_config,
        )
        result = analyzer.trade_off()
        assert len(result.objective_pairs) == 1
        assert result.objective_pairs[0] == ("cost", "quality")
        assert result.pareto_objectives.shape[0] == 3
        assert result.pareto_objectives.shape[1] == 2

    def test_trade_off_cached(
        self,
        multi_objective_result: OptimizationResult,
        mock_surrogate_multi: MagicMock,
        bound_dataset_multi: BoundDataset,
        analysis_config: AnalysisConfig,
    ) -> None:
        analyzer = Analyzer(
            multi_objective_result,
            mock_surrogate_multi,
            bound_dataset_multi,
            analysis_config,
        )
        r1 = analyzer.trade_off()
        r2 = analyzer.trade_off()
        assert r1 is r2
