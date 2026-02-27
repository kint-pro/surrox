from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from pymoo.optimize import minimize

from surrox._logging import log_duration
from surrox.optimizer.algorithm import select_algorithm
from surrox.optimizer.config import OptimizerConfig
from surrox.optimizer.extrapolation import ExtrapolationGate
from surrox.optimizer.problem_adapter import (
    SurroxProblem,
    _compute_extrapolation_penalty,
)
from surrox.optimizer.result import (
    EvaluatedPoint,
    OptimizationResult,
    _compute_compromise_index,
    _compute_hypervolume,
)
from surrox.problem.dataset import BoundDataset
from surrox.problem.definition import ProblemDefinition
from surrox.problem.scenarios import Scenario
from surrox.problem.types import ConstraintSeverity, Direction
from surrox.surrogate.manager import SurrogateManager

_logger = logging.getLogger(__name__)


def optimize(
    bound_dataset: BoundDataset,
    surrogate_manager: SurrogateManager,
    config: OptimizerConfig | None = None,
    scenario: Scenario | None = None,
) -> OptimizationResult:
    if config is None:
        config = OptimizerConfig()
    problem = bound_dataset.problem

    gate = ExtrapolationGate(
        training_data=bound_dataset.dataframe,
        decision_variables=problem.decision_variables,
        k=config.extrapolation_k,
        threshold=config.extrapolation_threshold,
    )

    penalty = _compute_extrapolation_penalty(problem, bound_dataset.dataframe)

    pymoo_problem = SurroxProblem(
        problem=problem,
        surrogate_manager=surrogate_manager,
        extrapolation_gate=gate,
        config=config,
        extrapolation_penalty=penalty,
        scenario=scenario,
    )

    algorithm = select_algorithm(problem, config)
    pymoo_problem.clear_diagnostics()

    with log_duration(
        _logger, "optimization",
        algorithm=type(algorithm).__name__,
        population_size=config.population_size,
        n_generations=config.n_generations,
    ):
        result = minimize(
            pymoo_problem,
            algorithm,
            ("n_gen", config.n_generations),
            seed=config.seed,
            verbose=False,
        )

    opt_result = _build_result(pymoo_problem, result, problem, config)
    _logger.info(
        "optimization result",
        extra={
            "n_feasible": len(opt_result.feasible_points),
            "n_infeasible": len(opt_result.infeasible_points),
            "hypervolume": opt_result.hypervolume,
        },
    )
    return opt_result


def _build_result(
    pymoo_problem: SurroxProblem,
    result: object,
    problem: ProblemDefinition,
    config: OptimizerConfig,
) -> OptimizationResult:
    X_raw = result.X  # type: ignore[union-attr]
    F_raw = result.F  # type: ignore[union-attr]
    G_raw = result.G  # type: ignore[union-attr]
    n_evals: int = result.algorithm.evaluator.n_eval  # type: ignore[union-attr]

    if X_raw is None:
        return _empty_result(problem, config)

    F_2d = np.atleast_2d(F_raw)
    G_2d = np.atleast_2d(G_raw) if G_raw is not None else None

    if isinstance(X_raw, np.ndarray) and X_raw.ndim == 1 and X_raw.dtype != object:
        X_raw = X_raw.reshape(1, -1)

    n_points = F_2d.shape[0]
    decision_var_names = [v.name for v in problem.decision_variables]
    objective_names = [o.name for o in problem.objectives]
    objective_directions = [o.direction for o in problem.objectives]

    diagnostics = pymoo_problem.point_diagnostics
    diag_offset = len(diagnostics) - n_points

    feasible: list[EvaluatedPoint] = []
    infeasible: list[EvaluatedPoint] = []

    for i in range(n_points):
        variables = _extract_variables(X_raw, i, decision_var_names)
        objectives = _extract_objectives(F_2d, i, objective_names, objective_directions)

        diag_idx = diag_offset + i
        if 0 <= diag_idx < len(diagnostics):
            constraint_evals, extrap_dist, is_extrap = diagnostics[diag_idx]
        else:
            constraint_evals = ()
            extrap_dist = 0.0
            is_extrap = False

        is_feasible = True
        if G_2d is not None:
            is_feasible = bool(np.all(G_2d[i] <= 0))

        point = EvaluatedPoint(
            variables=variables,
            objectives=objectives,
            constraints=constraint_evals,
            feasible=is_feasible,
            extrapolation_distance=extrap_dist,
            is_extrapolating=is_extrap,
        )

        if is_feasible:
            feasible.append(point)
        else:
            infeasible.append(point)

    infeasible.sort(
        key=lambda p: sum(
            max(0.0, ce.violation)
            for ce in p.constraints
            if ce.severity == ConstraintSeverity.HARD
        )
    )

    feasible_tuple = tuple(feasible)
    infeasible_tuple = tuple(infeasible)
    n_obj = len(problem.objectives)

    return OptimizationResult(
        feasible_points=feasible_tuple,
        infeasible_points=infeasible_tuple,
        has_feasible_solutions=len(feasible) > 0,
        compromise_index=_compute_compromise_index(feasible_tuple, n_obj),
        hypervolume=_compute_hypervolume(feasible_tuple, n_obj),
        problem=problem,
        n_generations=config.n_generations,
        n_evaluations=n_evals,
    )


def _empty_result(
    problem: ProblemDefinition, config: OptimizerConfig
) -> OptimizationResult:
    return OptimizationResult(
        feasible_points=(),
        infeasible_points=(),
        has_feasible_solutions=False,
        compromise_index=None,
        hypervolume=None,
        problem=problem,
        n_generations=config.n_generations,
        n_evaluations=0,
    )


def _extract_variables(
    X_raw: object, i: int, var_names: list[str]
) -> dict[str, object]:
    if isinstance(X_raw, np.ndarray):
        if X_raw.dtype == object:
            x = X_raw[i]
            if isinstance(x, dict):
                return {name: x[name] for name in var_names}
        return {name: float(X_raw[i, j]) for j, name in enumerate(var_names)}

    if isinstance(X_raw, dict):
        return {name: X_raw[name] for name in var_names}

    if isinstance(X_raw, list) and isinstance(X_raw[i], dict):
        return {name: X_raw[i][name] for name in var_names}

    return {name: float(X_raw[i][j]) for j, name in enumerate(var_names)}  # type: ignore[index]


def _extract_objectives(
    F: NDArray, i: int, names: list[str], directions: list[Direction]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for j, (name, direction) in enumerate(zip(names, directions, strict=True)):
        raw = float(F[i, j])
        result[name] = -raw if direction == Direction.MAXIMIZE else raw
    return result
