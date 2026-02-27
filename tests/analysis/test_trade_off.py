import numpy as np
import pytest

from surrox.analysis.analyzer import Analyzer
from surrox.analysis.config import AnalysisConfig
from surrox.exceptions import AnalysisError


class TestTradeOff:
    def test_marginal_rates_computed(
        self, multi_objective_result, mock_surrogate_multi,
        bound_dataset_multi, analysis_config,
    ) -> None:
        analyzer = Analyzer(
            multi_objective_result, mock_surrogate_multi,
            bound_dataset_multi, analysis_config,
        )
        result = analyzer.trade_off()
        pair = ("cost", "quality")
        assert pair in result.marginal_rates
        rates = result.marginal_rates[pair]
        assert len(rates) == 2

    def test_pareto_sorted_by_first_objective(
        self, multi_objective_result, mock_surrogate_multi,
        bound_dataset_multi, analysis_config,
    ) -> None:
        analyzer = Analyzer(
            multi_objective_result, mock_surrogate_multi,
            bound_dataset_multi, analysis_config,
        )
        result = analyzer.trade_off()
        costs = result.pareto_objectives[:, 0]
        assert np.all(np.diff(costs) >= 0)
