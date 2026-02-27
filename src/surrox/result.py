from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from surrox.analysis.result import AnalysisResult
from surrox.analysis.scenario import ScenarioComparisonResult
from surrox.optimizer.result import OptimizationResult


class SurroxResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    optimization: OptimizationResult
    analysis: AnalysisResult


class ScenariosResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    per_scenario: dict[str, SurroxResult]
    comparison: ScenarioComparisonResult
