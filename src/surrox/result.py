from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from surrox.analysis.result import AnalysisResult
from surrox.analysis.scenario import ScenarioComparisonResult
from surrox.optimizer.result import OptimizationResult


class SurroxResult(BaseModel):
    """Combined result of a single surrox pipeline run.

    Attributes:
        optimization: Pareto-optimal points and optimization metadata.
        analysis: Summary analysis computed automatically after optimization.
    """

    model_config = ConfigDict(frozen=True)

    optimization: OptimizationResult
    analysis: AnalysisResult


class ScenariosResult(BaseModel):
    """Combined result of a multi-scenario surrox pipeline run.

    Attributes:
        per_scenario: Results keyed by scenario name.
        comparison: Cross-scenario comparison metrics.
    """

    model_config = ConfigDict(frozen=True)

    per_scenario: dict[str, SurroxResult]
    comparison: ScenarioComparisonResult
