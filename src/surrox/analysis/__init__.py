from surrox.analysis.analyzer import Analyzer
from surrox.analysis.config import AnalysisConfig
from surrox.analysis.pdp import PDPICEResult
from surrox.analysis.result import AnalysisResult
from surrox.analysis.scenario import (
    ScenarioComparisonResult,
    VariableRobustness,
    compare_scenarios,
)
from surrox.analysis.shap import (
    FeatureImportanceResult,
    ShapGlobalResult,
    ShapLocalResult,
)
from surrox.analysis.summary import (
    BaselineComparison,
    ConstraintStatus,
    ExtrapolationWarning,
    MonotonicityViolation,
    SolutionSummary,
    Summary,
    SurrogateQuality,
    compute_summary,
)
from surrox.analysis.trade_off import TradeOffResult
from surrox.analysis.types import ConstraintStatusKind
from surrox.analysis.what_if import WhatIfPrediction, WhatIfResult
from surrox.optimizer.result import OptimizationResult
from surrox.problem.dataset import BoundDataset
from surrox.surrogate.manager import SurrogateManager


def analyze(
    optimization_result: OptimizationResult,
    surrogate_manager: SurrogateManager,
    bound_dataset: BoundDataset,
    config: AnalysisConfig = AnalysisConfig(),
) -> tuple[AnalysisResult, Analyzer]:
    summary = compute_summary(
        optimization_result, surrogate_manager, bound_dataset, config
    )
    result = AnalysisResult(summary=summary)
    analyzer = Analyzer(optimization_result, surrogate_manager, bound_dataset, config)
    return result, analyzer


__all__ = [
    "Analyzer",
    "AnalysisConfig",
    "AnalysisResult",
    "BaselineComparison",
    "ConstraintStatus",
    "ConstraintStatusKind",
    "ExtrapolationWarning",
    "FeatureImportanceResult",
    "MonotonicityViolation",
    "PDPICEResult",
    "ScenarioComparisonResult",
    "ShapGlobalResult",
    "ShapLocalResult",
    "SolutionSummary",
    "Summary",
    "SurrogateQuality",
    "TradeOffResult",
    "VariableRobustness",
    "WhatIfPrediction",
    "WhatIfResult",
    "analyze",
    "compare_scenarios",
]


def __dir__() -> list[str]:
    return __all__
