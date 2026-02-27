from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from surrox.problem.definition import ProblemDefinition
from surrox.problem.types import ConstraintSeverity


class ConstraintEvaluation(BaseModel):
    """Evaluation of a single constraint at a candidate point.

    Attributes:
        name: Name of the constraint.
        violation: Constraint violation magnitude (0.0 if satisfied).
        prediction: Predicted value from the surrogate.
        severity: Hard or soft constraint.
        lower_bound: Lower conformal prediction bound (data constraints only).
        upper_bound: Upper conformal prediction bound (data constraints only).
    """

    model_config = ConfigDict(frozen=True)

    name: str
    violation: float
    prediction: float
    severity: ConstraintSeverity = ConstraintSeverity.HARD
    lower_bound: float | None = None
    upper_bound: float | None = None


class EvaluatedPoint(BaseModel):
    """A single evaluated candidate from the optimization.

    Attributes:
        variables: Decision and context variable values.
        objectives: Predicted objective values by name.
        constraints: Constraint evaluations with violation info.
        feasible: Whether all hard constraints are satisfied.
        extrapolation_distance: Distance to the training data manifold.
        is_extrapolating: Whether this point is outside the training domain.
    """

    model_config = ConfigDict(frozen=True)

    variables: dict[str, Any]
    objectives: dict[str, float]
    constraints: tuple[ConstraintEvaluation, ...]
    feasible: bool
    extrapolation_distance: float
    is_extrapolating: bool


def _compute_compromise_index(
    feasible_points: tuple[EvaluatedPoint, ...],
    n_objectives: int,
) -> int | None:
    if n_objectives < 2 or len(feasible_points) < 2:
        return None

    objective_names = list(feasible_points[0].objectives.keys())
    values = np.array(
        [[p.objectives[name] for name in objective_names] for p in feasible_points]
    )

    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0

    normalized = (values - mins) / ranges
    distances = np.linalg.norm(normalized, axis=1)
    return int(np.argmin(distances))


def _compute_hypervolume(
    feasible_points: tuple[EvaluatedPoint, ...],
    n_objectives: int,
) -> float | None:
    if n_objectives < 2 or len(feasible_points) < 2:
        return None

    from pymoo.indicators.hv import HV

    objective_names = list(feasible_points[0].objectives.keys())
    feasible_values = np.array(
        [[p.objectives[name] for name in objective_names] for p in feasible_points]
    )

    ref_point = feasible_values.max(axis=0) * 1.1
    ref_point[ref_point == 0] = 0.1

    indicator = HV(ref_point=ref_point)
    return float(indicator(feasible_values))  # pyright: ignore[reportArgumentType]


class OptimizationResult(BaseModel):
    """Result of the optimization process.

    Attributes:
        feasible_points: Pareto-optimal points satisfying all hard constraints.
        infeasible_points: Points that violated at least one hard constraint.
        has_feasible_solutions: Whether any feasible solutions were found.
        compromise_index: Index into feasible_points of the recommended compromise solution (multi-objective only).
        hypervolume: Hypervolume indicator of the Pareto front (multi-objective only).
        problem: The problem definition used for optimization.
        n_generations: Number of generations executed.
        n_evaluations: Total number of candidate evaluations.
    """

    model_config = ConfigDict(frozen=True)

    feasible_points: tuple[EvaluatedPoint, ...]
    infeasible_points: tuple[EvaluatedPoint, ...]
    has_feasible_solutions: bool
    compromise_index: int | None
    hypervolume: float | None
    problem: ProblemDefinition
    n_generations: int
    n_evaluations: int
