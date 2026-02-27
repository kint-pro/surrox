from unittest.mock import MagicMock

import pytest

from surrox.analysis.analyzer import Analyzer
from surrox.analysis.config import AnalysisConfig
from surrox.analysis.what_if import WhatIfResult
from surrox.optimizer.result import OptimizationResult
from surrox.problem.dataset import BoundDataset


class TestWhatIf:
    def test_returns_predictions(
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
        result = analyzer.what_if({"x1": 5.0, "x2": 5.0})
        assert isinstance(result, WhatIfResult)
        assert "cost" in result.objectives
        assert result.objectives["cost"].predicted == 45.0

    def test_not_cached(
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
        r1 = analyzer.what_if({"x1": 5.0, "x2": 5.0})
        r2 = analyzer.what_if({"x1": 5.0, "x2": 5.0})
        assert r1 is not r2
