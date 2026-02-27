from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from surrox.problem.definition import ProblemDefinition


class ConstraintEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    violation: float
    prediction: float
    lower_bound: float | None = None
    upper_bound: float | None = None


class EvaluatedPoint(BaseModel):
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
    model_config = ConfigDict(frozen=True)

    feasible_points: tuple[EvaluatedPoint, ...]
    infeasible_points: tuple[EvaluatedPoint, ...]
    has_feasible_solutions: bool
    compromise_index: int | None
    hypervolume: float | None
    problem: ProblemDefinition
    n_generations: int
    n_evaluations: int
