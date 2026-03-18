from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from surrox._logging import log_duration
from surrox.analysis import AnalysisConfig, AnalysisResult, Analyzer, analyze
from surrox.analysis.scenario import ScenarioComparisonResult, compare_scenarios
from surrox.exceptions import SurroxError
from surrox.optimizer import (
    OptimizationResult,
    OptimizerConfig,
    Strategy,
    TuRBOConfig,
    optimize,
)
from surrox.persistence import load_result, save_result
from surrox.problem import ProblemDefinition
from surrox.problem.constraints import DataConstraint, LinearConstraint
from surrox.problem.dataset import BoundDataset
from surrox.problem.domain_knowledge import MonotonicRelation
from surrox.problem.objectives import Objective
from surrox.problem.scenarios import Scenario
from surrox.problem.types import (
    ConstraintOperator,
    ConstraintSeverity,
    Direction,
    DType,
    MonotonicDirection,
    Role,
)
from surrox.problem.variables import (
    CategoricalBounds,
    ContinuousBounds,
    IntegerBounds,
    OrdinalBounds,
    Variable,
)
from surrox.result import ScenariosResult, SurroxResult
from surrox.suggest import ObjectivePrediction, Suggestion, SuggestionResult, suggest
from surrox.surrogate import FeatureReductionConfig, SurrogateManager, TrainingConfig
from surrox.surrogate.models import EnsembleMemberConfig

if TYPE_CHECKING:
    import pandas as pd

_logger = logging.getLogger(__name__)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "run",
    "run_scenarios",
    "suggest",
    "Suggestion",
    "SuggestionResult",
    "ObjectivePrediction",
    "EnsembleMemberConfig",
    "ProblemDefinition",
    "Variable",
    "ContinuousBounds",
    "IntegerBounds",
    "CategoricalBounds",
    "OrdinalBounds",
    "Objective",
    "DataConstraint",
    "LinearConstraint",
    "MonotonicRelation",
    "Scenario",
    "DType",
    "Role",
    "Direction",
    "ConstraintOperator",
    "ConstraintSeverity",
    "MonotonicDirection",
    "BoundDataset",
    "SurroxResult",
    "ScenariosResult",
    "TrainingConfig",
    "SurrogateManager",
    "OptimizerConfig",
    "Strategy",
    "TuRBOConfig",
    "OptimizationResult",
    "AnalysisConfig",
    "AnalysisResult",
    "Analyzer",
    "ScenarioComparisonResult",
    "FeatureReductionConfig",
    "SurroxError",
    "save_result",
    "load_result",
]


def __dir__() -> list[str]:
    return __all__


def run(
    problem: ProblemDefinition,
    dataframe: pd.DataFrame,
    surrogate_config: TrainingConfig = TrainingConfig(),
    optimizer_config: OptimizerConfig = OptimizerConfig(),
    analysis_config: AnalysisConfig = AnalysisConfig(),
    scenario: Scenario | None = None,
) -> tuple[SurroxResult, Analyzer]:
    """Run the full surrox pipeline: train surrogates, optimize, and analyze.

    Args:
        problem: Declarative problem definition with variables, objectives, and constraints.
        dataframe: Historical data matching the problem definition.
        surrogate_config: Surrogate training configuration.
        optimizer_config: Optimizer configuration.
        analysis_config: Analysis configuration.
        scenario: Optional scenario fixing context variables to specific values.

    Returns:
        A tuple of the optimization and analysis result, and an analyzer for
        on-demand detail analyses (SHAP, PDP/ICE, What-If).
    """
    with log_duration(
        _logger, "surrox.run",
        n_variables=len(problem.variables),
        n_objectives=len(problem.objectives),
        n_constraints=len(problem.data_constraints),
        n_rows=len(dataframe),
    ):
        bound_dataset = BoundDataset(problem=problem, dataframe=dataframe)

        surrogate_manager = SurrogateManager.train(
            problem=problem,
            dataset=bound_dataset,
            config=surrogate_config,
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
    surrogate_config: TrainingConfig = TrainingConfig(),
    optimizer_config: OptimizerConfig = OptimizerConfig(),
    analysis_config: AnalysisConfig = AnalysisConfig(),
) -> tuple[ScenariosResult, dict[str, Analyzer]]:
    """Run the full surrox pipeline for multiple scenarios and compare results.

    Trains surrogates once, then optimizes and analyzes each scenario independently.
    Produces a cross-scenario comparison alongside per-scenario results.

    Args:
        problem: Declarative problem definition with variables, objectives, and constraints.
        dataframe: Historical data matching the problem definition.
        scenarios: Named scenarios mapping to Scenario objects. Must contain at least 2.
        surrogate_config: Surrogate training configuration.
        optimizer_config: Optimizer configuration.
        analysis_config: Analysis configuration.

    Returns:
        A tuple of the scenarios result (per-scenario results + comparison),
        and a dict mapping scenario names to their analyzers.

    Raises:
        SurroxError: If fewer than 2 scenarios are provided.
    """
    if len(scenarios) < 2:
        raise SurroxError("run_scenarios requires at least 2 scenarios")

    with log_duration(
        _logger, "surrox.run_scenarios",
        n_scenarios=len(scenarios),
        scenario_names=list(scenarios.keys()),
    ):
        bound_dataset = BoundDataset(problem=problem, dataframe=dataframe)

        surrogate_manager = SurrogateManager.train(
            problem=problem,
            dataset=bound_dataset,
            config=surrogate_config,
        )

        per_scenario: dict[str, SurroxResult] = {}
        analyzers: dict[str, Analyzer] = {}
        optimization_results: dict[str, OptimizationResult] = {}

        for name, scenario in scenarios.items():
            _logger.info(
                "surrox.scenario started",
                extra={"scenario": name},
            )
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
