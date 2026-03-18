from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from surrox.exceptions import AnalysisError
from surrox.problem.types import DType

if TYPE_CHECKING:
    from surrox.optimizer.result import OptimizationResult
    from surrox.problem.definition import ProblemDefinition


class VariableRobustness(BaseModel):
    """Robustness assessment for a single decision variable across scenarios.

    Attributes:
        variable_name: Name of the decision variable.
        values_per_scenario: Recommended value per scenario.
        is_robust: Whether the variable is stable across scenarios (spread < 5% of bounds range).
        spread: Absolute range of values across scenarios (0.0 for categorical agreement).
    """

    model_config = ConfigDict(frozen=True)

    variable_name: str
    values_per_scenario: dict[str, Any]
    is_robust: bool
    spread: float


class ScenarioComparisonResult(BaseModel):
    """Cross-scenario comparison of recommended decision variable settings.

    Attributes:
        scenario_names: Names of the compared scenarios.
        variable_robustness: Robustness assessment per decision variable.
    """

    model_config = ConfigDict(frozen=True)

    scenario_names: tuple[str, ...]
    variable_robustness: dict[str, VariableRobustness]


def compare_scenarios(
    results: dict[str, OptimizationResult],
    problem: ProblemDefinition,
) -> ScenarioComparisonResult:
    if len(results) < 2:
        raise AnalysisError("at least 2 scenarios required for comparison")

    scenario_names = tuple(results.keys())
    recommended: dict[str, dict[str, Any]] = {}

    for name, opt_result in results.items():
        if not opt_result.has_feasible_solutions:
            raise AnalysisError(f"scenario '{name}' has no feasible solutions")
        if opt_result.compromise_index is not None:
            point = opt_result.feasible_points[opt_result.compromise_index]
        else:
            point = opt_result.feasible_points[0]
        recommended[name] = dict(point.variables)

    robustness: dict[str, VariableRobustness] = {}
    for var in problem.decision_variables:
        values_per_scenario = {
            name: recommended[name][var.name] for name in scenario_names
        }
        is_robust, spread = _compute_robustness(var, values_per_scenario)
        robustness[var.name] = VariableRobustness(
            variable_name=var.name,
            values_per_scenario=values_per_scenario,
            is_robust=is_robust,
            spread=spread,
        )

    return ScenarioComparisonResult(
        scenario_names=scenario_names,
        variable_robustness=robustness,
    )


def _compute_robustness(
    variable: Any,
    values_per_scenario: dict[str, Any],
) -> tuple[bool, float]:
    values = list(values_per_scenario.values())

    if variable.dtype in (DType.CATEGORICAL, DType.ORDINAL):
        unique = set(str(v) for v in values)
        is_robust = len(unique) == 1
        spread = 0.0 if is_robust else 1.0
        return is_robust, spread

    numeric_values = np.array([float(v) for v in values])
    value_range = float(np.ptp(numeric_values))
    bounds_range = float(variable.bounds.upper - variable.bounds.lower)  # type: ignore[union-attr]

    if bounds_range > 0:
        is_robust = value_range / bounds_range < 0.05
    else:
        is_robust = value_range == 0.0

    return is_robust, value_range
