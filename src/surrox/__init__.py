from __future__ import annotations

from typing import TYPE_CHECKING

from surrox.analysis import AnalysisConfig, AnalysisResult, Analyzer, analyze
from surrox.analysis.scenario import ScenarioComparisonResult, compare_scenarios
from surrox.exceptions import SurroxError
from surrox.optimizer import OptimizationResult, OptimizerConfig, optimize
from surrox.persistence import load_result, save_result
from surrox.problem import ProblemDefinition
from surrox.problem.dataset import BoundDataset
from surrox.problem.scenarios import Scenario
from surrox.result import ScenariosResult, SurroxResult
from surrox.surrogate import SurrogateManager, TrainingConfig

if TYPE_CHECKING:
    import pandas as pd

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "run",
    "run_scenarios",
    "SurroxResult",
    "ScenariosResult",
    "ProblemDefinition",
    "BoundDataset",
    "Analyzer",
    "AnalysisConfig",
    "AnalysisResult",
    "OptimizerConfig",
    "OptimizationResult",
    "TrainingConfig",
    "SurrogateManager",
    "Scenario",
    "ScenarioComparisonResult",
    "SurroxError",
    "save_result",
    "load_result",
]


def __dir__() -> list[str]:
    return __all__


def run(
    problem: ProblemDefinition,
    dataframe: pd.DataFrame,
    surrogate_config: TrainingConfig | None = None,
    optimizer_config: OptimizerConfig | None = None,
    analysis_config: AnalysisConfig | None = None,
    scenario: Scenario | None = None,
) -> tuple[SurroxResult, Analyzer]:
    bound_dataset = BoundDataset(problem=problem, dataframe=dataframe)

    surrogate_manager = SurrogateManager.train(
        problem=problem,
        dataset=bound_dataset,
        config=surrogate_config or TrainingConfig(),
    )

    optimization_result = optimize(
        bound_dataset=bound_dataset,
        surrogate_manager=surrogate_manager,
        config=optimizer_config,
        scenario=scenario,
    )

    analysis_result, analyzer = analyze(
        optimization_result=optimization_result,
        surrogate_manager=surrogate_manager,
        bound_dataset=bound_dataset,
        config=analysis_config,
    )

    result = SurroxResult(
        optimization=optimization_result,
        analysis=analysis_result,
    )

    return result, analyzer


def run_scenarios(
    problem: ProblemDefinition,
    dataframe: pd.DataFrame,
    scenarios: dict[str, Scenario],
    surrogate_config: TrainingConfig | None = None,
    optimizer_config: OptimizerConfig | None = None,
    analysis_config: AnalysisConfig | None = None,
) -> tuple[ScenariosResult, dict[str, Analyzer]]:
    if len(scenarios) < 2:
        raise SurroxError("run_scenarios requires at least 2 scenarios")

    bound_dataset = BoundDataset(problem=problem, dataframe=dataframe)

    surrogate_manager = SurrogateManager.train(
        problem=problem,
        dataset=bound_dataset,
        config=surrogate_config or TrainingConfig(),
    )

    per_scenario: dict[str, SurroxResult] = {}
    analyzers: dict[str, Analyzer] = {}
    optimization_results: dict[str, OptimizationResult] = {}

    for name, scenario in scenarios.items():
        optimization_result = optimize(
            bound_dataset=bound_dataset,
            surrogate_manager=surrogate_manager,
            config=optimizer_config,
            scenario=scenario,
        )
        optimization_results[name] = optimization_result

        analysis_result, analyzer = analyze(
            optimization_result=optimization_result,
            surrogate_manager=surrogate_manager,
            bound_dataset=bound_dataset,
            config=analysis_config,
        )

        per_scenario[name] = SurroxResult(
            optimization=optimization_result,
            analysis=analysis_result,
        )
        analyzers[name] = analyzer

    comparison = compare_scenarios(
        results=optimization_results,
        problem=problem,
    )

    result = ScenariosResult(
        per_scenario=per_scenario,
        comparison=comparison,
    )

    return result, analyzers
