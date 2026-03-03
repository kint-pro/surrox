from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pymoo.optimize import minimize

from surrox._logging import log_duration
from surrox.exceptions import OptimizationError
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
from surrox.problem.types import ConstraintSeverity, DType
from surrox.surrogate.manager import SurrogateManager

_logger = logging.getLogger(__name__)


def _compute_trust_region_bounds(
    problem: ProblemDefinition,
    training_df: pd.DataFrame,
    config: OptimizerConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    if config.trust_region_margin is None:
        return None

    if any(
        v.dtype in (DType.CATEGORICAL, DType.ORDINAL)
        for v in problem.decision_variables
    ):
        return None

    decision_vars = problem.decision_variables
    margin = config.trust_region_margin
    xl = np.empty(len(decision_vars), dtype=np.float64)
    xu = np.empty(len(decision_vars), dtype=np.float64)
    center = config.trust_region_center

    if center is not None:
        missing = {v.name for v in decision_vars} - set(center.keys())
        if missing:
            raise OptimizationError(
                f"trust_region_center missing keys for decision variables: {missing}"
            )
        for i, var in enumerate(decision_vars):
            var_range = var.bounds.upper - var.bounds.lower
            half_width = margin * var_range
            c = center[var.name]
            xl[i] = max(var.bounds.lower, c - half_width)
            xu[i] = min(var.bounds.upper, c + half_width)
    else:
        for i, var in enumerate(decision_vars):
            col = training_df[var.name]
            data_min = float(col.min())
            data_max = float(col.max())
            data_range = data_max - data_min
            xl[i] = max(var.bounds.lower, data_min - margin * data_range)
            xu[i] = min(var.bounds.upper, data_max + margin * data_range)

    return xl, xu


def _run_minimization(
    bound_dataset: BoundDataset,
    surrogate_manager: SurrogateManager,
    config: OptimizerConfig,
    scenario: Scenario | None,
    log_label: str,
    seed_points: list[dict[str, float]] | None = None,
    **log_extra: object,
) -> tuple[object, SurroxProblem, ExtrapolationGate]:
    problem = bound_dataset.problem

    gate = ExtrapolationGate(
        training_data=bound_dataset.dataframe,
        decision_variables=problem.decision_variables,
        k=config.extrapolation_k,
        threshold=config.extrapolation_threshold,
    )

    penalty = _compute_extrapolation_penalty(problem, bound_dataset.dataframe)

    trust_bounds = _compute_trust_region_bounds(
        problem, bound_dataset.dataframe, config,
    )

    pymoo_problem = SurroxProblem(
        problem=problem,
        surrogate_manager=surrogate_manager,
        extrapolation_gate=gate,
        config=config,
        extrapolation_penalty=penalty,
        scenario=scenario,
        trust_region_bounds=trust_bounds,
    )

    seed_X = None
    if seed_points:
        decision_vars = problem.decision_variables
        seed_X = np.array([
            [point[v.name] for v in decision_vars]
            for point in seed_points
        ], dtype=np.float64)

    algorithm = select_algorithm(problem, config, seed_X=seed_X)
    pymoo_problem.clear_diagnostics()

    with log_duration(
        _logger, log_label,
        algorithm=type(algorithm).__name__,
        population_size=config.population_size,
        n_generations=config.n_generations,
        **log_extra,
    ):
        pymoo_result = minimize(
            pymoo_problem,
            algorithm,
            ("n_gen", config.n_generations),
            seed=config.seed,
            verbose=False,
        )

    return pymoo_result, pymoo_problem, gate


def optimize(
    bound_dataset: BoundDataset,
    surrogate_manager: SurrogateManager,
    config: OptimizerConfig = OptimizerConfig(),  # noqa: B008
    scenario: Scenario | None = None,
) -> OptimizationResult:
    result, pymoo_problem, _ = _run_minimization(
        bound_dataset, surrogate_manager, config, scenario,
        log_label="optimization",
    )

    opt_result = _build_result(
        pymoo_problem, result, bound_dataset.problem, config,
        surrogate_manager, scenario,
    )
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
    surrogate_manager: SurrogateManager,
    scenario: Scenario | None = None,
) -> OptimizationResult:
    X_raw = result.X  # type: ignore[union-attr]
    G_raw = result.G  # type: ignore[union-attr]
    n_evals: int = result.algorithm.evaluator.n_eval  # type: ignore[union-attr]

    if X_raw is None:
        return _empty_result(problem, config)

    G_2d = np.atleast_2d(G_raw) if G_raw is not None else None

    if isinstance(X_raw, np.ndarray) and X_raw.ndim == 1 and X_raw.dtype != object:
        X_raw = X_raw.reshape(1, -1)

    n_points = result.F.shape[0] if np.ndim(result.F) > 1 else 1  # type: ignore[union-attr]
    decision_var_names = [v.name for v in problem.decision_variables]
    objective_names = [o.name for o in problem.objectives]
    context_values = scenario.context_values if scenario is not None else {}

    F_actual = _reevaluate_objectives(
        X_raw, n_points, decision_var_names, problem, surrogate_manager, scenario,
    )

    diagnostics = pymoo_problem.point_diagnostics
    diag_offset = len(diagnostics) - n_points

    feasible: list[EvaluatedPoint] = []
    infeasible: list[EvaluatedPoint] = []

    for i in range(n_points):
        variables = _extract_variables(X_raw, i, decision_var_names)
        variables.update(context_values)
        objectives = {
            name: float(F_actual[i, j])
            for j, name in enumerate(objective_names)
        }

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


def suggest_candidates(
    bound_dataset: BoundDataset,
    surrogate_manager: SurrogateManager,
    n_candidates: int,
    config: OptimizerConfig = OptimizerConfig(),  # noqa: B008
    scenario: Scenario | None = None,
    seed_points: list[dict[str, float]] | None = None,
) -> tuple[EvaluatedPoint, ...]:
    result, _, gate = _run_minimization(
        bound_dataset, surrogate_manager, config, scenario,
        log_label="suggest_candidates",
        seed_points=seed_points,
        n_candidates=n_candidates,
    )

    pop = result.pop
    if pop is None:
        raise OptimizationError("Optimizer produced no population")

    X_pop = pop.get("X")
    F_pop = pop.get("F")
    G_pop = pop.get("G")

    if X_pop is None or F_pop is None:
        raise OptimizationError(
            "Optimizer population contains no decision variables or objectives"
        )

    problem = bound_dataset.problem
    decision_var_names = [v.name for v in problem.decision_variables]
    objective_names = [o.name for o in problem.objectives]
    context_values = scenario.context_values if scenario is not None else {}

    n_pop = X_pop.shape[0]
    X_eval = _reevaluate_objectives(
        X_pop, n_pop, decision_var_names, problem, surrogate_manager, scenario,
    )

    scored = np.zeros(n_pop, dtype=np.float64)
    for j, obj in enumerate(problem.objectives):
        sign = 1.0 if obj.direction.value == "minimize" else -1.0
        scored += sign * X_eval[:, j]

    feasible_mask = np.ones(n_pop, dtype=bool)
    if G_pop is not None:
        feasible_mask = np.all(G_pop <= 0, axis=1)

    feasible_indices = np.where(feasible_mask)[0]
    if len(feasible_indices) == 0:
        feasible_indices = np.arange(n_pop)

    sorted_indices = feasible_indices[np.argsort(scored[feasible_indices])]

    selected = _select_diverse(X_pop, sorted_indices, n_candidates)

    candidates: list[EvaluatedPoint] = []
    for idx in selected:
        variables = _extract_variables(X_pop, idx, decision_var_names)
        variables.update(context_values)
        objectives = {
            name: float(X_eval[idx, j])
            for j, name in enumerate(objective_names)
        }

        point_df = pd.DataFrame([{
            name: variables[name] for name in decision_var_names
        }])
        is_extrap_arr, dist_arr = gate.evaluate(point_df)
        extrap_dist = float(dist_arr[0])
        is_extrap = bool(is_extrap_arr[0])

        candidates.append(EvaluatedPoint(
            variables=variables,
            objectives=objectives,
            constraints=(),
            feasible=bool(feasible_mask[idx]),
            extrapolation_distance=extrap_dist,
            is_extrapolating=is_extrap,
        ))

    return tuple(candidates)


def _select_diverse(
    X: NDArray,
    sorted_indices: NDArray,
    n: int,
) -> list[int]:
    if len(sorted_indices) <= n:
        return sorted_indices.tolist()

    X_norm = X[sorted_indices].astype(np.float64)
    ranges = X_norm.max(axis=0) - X_norm.min(axis=0)
    ranges[ranges == 0] = 1.0
    X_norm = (X_norm - X_norm.min(axis=0)) / ranges

    selected: list[int] = [0]
    for _ in range(n - 1):
        best_idx = -1
        best_min_dist = -1.0
        for candidate_pos in range(len(sorted_indices)):
            if candidate_pos in selected:
                continue
            min_dist = min(
                float(np.linalg.norm(X_norm[candidate_pos] - X_norm[s]))
                for s in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = candidate_pos
        if best_idx == -1:
            break
        selected.append(best_idx)

    return [int(sorted_indices[i]) for i in selected]


def _reevaluate_objectives(
    X_raw: object,
    n_points: int,
    decision_var_names: list[str],
    problem: ProblemDefinition,
    surrogate_manager: SurrogateManager,
    scenario: Scenario | None,
) -> NDArray[np.float64]:
    rows: list[dict[str, object]] = []
    for i in range(n_points):
        variables = _extract_variables(X_raw, i, decision_var_names)
        if scenario is not None:
            variables.update(scenario.context_values)
        rows.append(variables)

    df = pd.DataFrame(rows)
    predictions = surrogate_manager.evaluate(df)

    F = np.zeros((n_points, len(problem.objectives)), dtype=np.float64)
    for j, obj in enumerate(problem.objectives):
        F[:, j] = predictions[obj.column]
    return F
