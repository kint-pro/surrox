from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from surrox.analysis.types import ConstraintStatusKind
from surrox.optimizer.result import ConstraintEvaluation
from surrox.problem.types import ConstraintOperator, Direction, MonotonicDirection
from surrox.problem.variables import ContinuousBounds, IntegerBounds

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from surrox.analysis.config import AnalysisConfig
    from surrox.optimizer.result import EvaluatedPoint, OptimizationResult
    from surrox.problem.dataset import BoundDataset
    from surrox.problem.definition import ProblemDefinition
    from surrox.surrogate.manager import SurrogateManager


_ACTIVE_THRESHOLD = 0.05
_EPSILON = 1e-8


class SolutionSummary(BaseModel):
    """High-level overview of the optimization outcome.

    Attributes:
        n_feasible: Number of feasible Pareto-optimal points.
        n_infeasible: Number of infeasible points.
        best_objectives: Best value found per objective across feasible points.
        compromise_objectives: Objective values at the compromise point (multi-objective only).
        hypervolume: Hypervolume indicator of the Pareto front (multi-objective only).
    """

    model_config = ConfigDict(frozen=True)

    n_feasible: int
    n_infeasible: int
    best_objectives: dict[str, float]
    compromise_objectives: dict[str, float] | None
    hypervolume: float | None


class BaselineComparison(BaseModel):
    """Comparison of the recommended solution against historical data.

    Attributes:
        recommended_objectives: Predicted objectives at the recommended point.
        historical_best_per_objective: Best historical value per objective from the training data.
        improvement: Improvement over historical best (positive = better).
    """

    model_config = ConfigDict(frozen=True)

    recommended_objectives: dict[str, float]
    historical_best_per_objective: dict[str, float]
    improvement: dict[str, float]


class ConstraintStatus(BaseModel):
    """Status of a constraint at the recommended point.

    Attributes:
        evaluation: Raw constraint evaluation with violation and prediction.
        status: Classification as satisfied, active, or violated.
        margin: Distance to the constraint boundary (positive = satisfied).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    evaluation: ConstraintEvaluation
    status: ConstraintStatusKind
    margin: float


class SurrogateQuality(BaseModel):
    """Quality metrics for a trained surrogate ensemble.

    Attributes:
        column: Target column name.
        cv_rmse: Best cross-validation RMSE from the Optuna study.
        conformal_coverage: Conformal prediction interval coverage level.
        ensemble_size: Number of models in the ensemble.
        warning: Quality warning if CV RMSE significantly exceeds conformity scores.
    """

    model_config = ConfigDict(frozen=True)

    column: str
    cv_rmse: float
    conformal_coverage: float
    ensemble_size: int
    warning: str | None


class ExtrapolationWarning(BaseModel):
    """Warning for a feasible point that lies outside the training data domain.

    Attributes:
        point_index: Index into the feasible points list.
        distance: k-NN distance to the training data manifold.
    """

    model_config = ConfigDict(frozen=True)

    point_index: int
    distance: float


class MonotonicityViolation(BaseModel):
    """Detected violation of a declared monotonic relationship.

    Attributes:
        decision_variable: Variable involved in the violation.
        target: Objective or constraint target.
        declared_direction: Expected monotonic direction.
        violation_fraction: Fraction of grid intervals where monotonicity is violated.
        max_reversal: Maximum absolute reversal magnitude.
    """

    model_config = ConfigDict(frozen=True)

    decision_variable: str
    target: str
    declared_direction: MonotonicDirection
    violation_fraction: float
    max_reversal: float


class Summary(BaseModel):
    """Automatic post-optimization summary.

    Attributes:
        solution_summary: Overview of feasible/infeasible solutions and best objectives.
        baseline_comparison: Improvement vs. historical best (None if no feasible solutions).
        constraint_status: Status of each constraint at the recommended point.
        surrogate_quality: Quality metrics per trained surrogate.
        extrapolation_warnings: Feasible points flagged as extrapolating.
        monotonicity_violations: Detected violations of declared monotonic relations.
    """

    model_config = ConfigDict(frozen=True)

    solution_summary: SolutionSummary
    baseline_comparison: BaselineComparison | None
    constraint_status: tuple[ConstraintStatus, ...]
    surrogate_quality: tuple[SurrogateQuality, ...]
    extrapolation_warnings: tuple[ExtrapolationWarning, ...]
    monotonicity_violations: tuple[MonotonicityViolation, ...]


def compute_summary(
    optimization_result: OptimizationResult,
    surrogate_manager: SurrogateManager,
    bound_dataset: BoundDataset,
    config: AnalysisConfig,
) -> Summary:
    problem = optimization_result.problem
    recommended = _get_recommended_solution(optimization_result)

    summary = Summary(
        solution_summary=_compute_solution_summary(optimization_result),
        baseline_comparison=_compute_baseline_comparison(
            optimization_result, bound_dataset, recommended
        ),
        constraint_status=_compute_constraint_status(problem, recommended),
        surrogate_quality=_compute_surrogate_quality(surrogate_manager, problem),
        extrapolation_warnings=_compute_extrapolation_warnings(optimization_result),
        monotonicity_violations=_compute_monotonicity_violations(
            optimization_result, surrogate_manager, config, recommended
        ),
    )
    _logger.info(
        "summary complete",
        extra={
            "n_objectives": len(problem.objectives),
            "n_constraints": len(problem.data_constraints),
            "n_extrapolation_warnings": len(summary.extrapolation_warnings),
            "n_monotonicity_violations": len(summary.monotonicity_violations),
        },
    )
    return summary


def _get_recommended_solution(
    optimization_result: OptimizationResult,
) -> EvaluatedPoint | None:
    if not optimization_result.has_feasible_solutions:
        return None
    if optimization_result.compromise_index is not None:
        return optimization_result.feasible_points[optimization_result.compromise_index]
    return optimization_result.feasible_points[0]


def _compute_solution_summary(
    optimization_result: OptimizationResult,
) -> SolutionSummary:
    problem = optimization_result.problem
    feasible = optimization_result.feasible_points

    best_objectives: dict[str, float] = {}
    for obj in problem.objectives:
        if feasible:
            values = [p.objectives[obj.name] for p in feasible]
            if obj.direction == Direction.MINIMIZE:
                best_objectives[obj.name] = min(values)
            else:
                best_objectives[obj.name] = max(values)

    compromise_objectives: dict[str, float] | None = None
    if optimization_result.compromise_index is not None:
        compromise_point = feasible[optimization_result.compromise_index]
        compromise_objectives = dict(compromise_point.objectives)

    return SolutionSummary(
        n_feasible=len(feasible),
        n_infeasible=len(optimization_result.infeasible_points),
        best_objectives=best_objectives,
        compromise_objectives=compromise_objectives,
        hypervolume=optimization_result.hypervolume,
    )


def _compute_baseline_comparison(
    optimization_result: OptimizationResult,
    bound_dataset: BoundDataset,
    recommended: EvaluatedPoint | None,
) -> BaselineComparison | None:
    if recommended is None:
        return None

    problem = optimization_result.problem
    df = bound_dataset.dataframe

    recommended_objectives = dict(recommended.objectives)
    historical_best: dict[str, float] = {}
    improvement: dict[str, float] = {}

    for obj in problem.objectives:
        col_values = df[obj.column].to_numpy()
        if obj.direction == Direction.MINIMIZE:
            hist_best = float(np.min(col_values))
            historical_best[obj.name] = hist_best
            improvement[obj.name] = hist_best - recommended_objectives[obj.name]
        else:
            hist_best = float(np.max(col_values))
            historical_best[obj.name] = hist_best
            improvement[obj.name] = recommended_objectives[obj.name] - hist_best

    return BaselineComparison(
        recommended_objectives=recommended_objectives,
        historical_best_per_objective=historical_best,
        improvement=improvement,
    )


def _get_constraint_limit(
    problem: ProblemDefinition, constraint_name: str
) -> tuple[float, ConstraintOperator]:
    for dc in problem.data_constraints:
        if dc.name == constraint_name:
            return dc.limit, dc.operator
    for lc in problem.linear_constraints:
        if lc.name == constraint_name:
            return lc.rhs, lc.operator
    raise ValueError(f"constraint '{constraint_name}' not found")


def _compute_margin(
    prediction: float,
    limit: float,
    operator: ConstraintOperator,
) -> float:
    if operator == ConstraintOperator.LE:
        return limit - prediction
    elif operator == ConstraintOperator.GE:
        return prediction - limit
    else:
        return -abs(prediction - limit)


def _classify_constraint_status(
    violation: float, margin: float, limit: float
) -> ConstraintStatusKind:
    if violation > 0:
        return ConstraintStatusKind.VIOLATED
    threshold = _ACTIVE_THRESHOLD * max(abs(limit), _EPSILON)
    if abs(margin) <= threshold:
        return ConstraintStatusKind.ACTIVE
    return ConstraintStatusKind.SATISFIED


def _compute_constraint_status(
    problem: ProblemDefinition,
    recommended: EvaluatedPoint | None,
) -> tuple[ConstraintStatus, ...]:
    if recommended is None:
        return ()

    statuses: list[ConstraintStatus] = []
    for ce in recommended.constraints:
        limit, operator = _get_constraint_limit(problem, ce.name)
        margin = _compute_margin(ce.prediction, limit, operator)
        status = _classify_constraint_status(ce.violation, margin, limit)
        statuses.append(ConstraintStatus(evaluation=ce, status=status, margin=margin))
    return tuple(statuses)


def _compute_surrogate_quality(
    surrogate_manager: SurrogateManager,
    problem: ProblemDefinition,
) -> tuple[SurrogateQuality, ...]:
    qualities: list[SurrogateQuality] = []
    for column in problem.surrogate_columns:
        sr = surrogate_manager.get_surrogate_result(column)
        ensemble = sr.ensemble
        trial_history = sr.trial_history

        best_cv_rmse = min(t.mean_rmse for t in trial_history)

        conformal = sr.conformal
        coverage = conformal._default_coverage

        warning: str | None = None
        median_score = float(np.median(conformal.conformity_scores))
        if median_score > 0 and best_cv_rmse > 2 * median_score:
            warning = (
                f"cv_rmse ({best_cv_rmse:.4f}) significantly exceeds "
                f"median conformity score ({median_score:.4f})"
            )

        qualities.append(
            SurrogateQuality(
                column=column,
                cv_rmse=best_cv_rmse,
                conformal_coverage=coverage,
                ensemble_size=len(ensemble.members),
                warning=warning,
            )
        )
    return tuple(qualities)


def _compute_extrapolation_warnings(
    optimization_result: OptimizationResult,
) -> tuple[ExtrapolationWarning, ...]:
    warnings: list[ExtrapolationWarning] = []
    for i, point in enumerate(optimization_result.feasible_points):
        if point.is_extrapolating:
            warnings.append(
                ExtrapolationWarning(
                    point_index=i,
                    distance=point.extrapolation_distance,
                )
            )
    return tuple(warnings)


def _compute_monotonicity_violations(
    optimization_result: OptimizationResult,
    surrogate_manager: SurrogateManager,
    config: AnalysisConfig,
    recommended: EvaluatedPoint | None,
) -> tuple[MonotonicityViolation, ...]:
    if recommended is None:
        return ()

    problem = optimization_result.problem
    if not problem.monotonic_relations:
        return ()

    violations: list[MonotonicityViolation] = []
    resolution = config.monotonicity_check_resolution

    for mr in problem.monotonic_relations:
        variable = next(
            v for v in problem.decision_variables if v.name == mr.decision_variable
        )
        bounds = variable.bounds
        if not isinstance(bounds, (ContinuousBounds, IntegerBounds)):
            raise ValueError(
                f"monotonicity check requires numeric bounds for '{variable.name}'"
            )

        grid = np.linspace(
            bounds.lower,
            bounds.upper,
            resolution,
        )

        base_values: dict[str, Any] = dict(recommended.variables)
        rows = []
        for val in grid:
            row = dict(base_values)
            row[mr.decision_variable] = val
            rows.append(row)

        df = pd.DataFrame(rows)
        column = problem.target_to_column[mr.objective_or_constraint]
        ensemble = surrogate_manager.get_ensemble(column)
        predictions = ensemble.predict(df)

        diffs = np.diff(predictions)
        if mr.direction == MonotonicDirection.INCREASING:
            violating = diffs < 0
        else:
            violating = diffs > 0

        violation_fraction = float(np.mean(violating))
        if violation_fraction > 0:
            max_reversal = float(np.max(np.abs(diffs[violating])))

            violations.append(
                MonotonicityViolation(
                    decision_variable=mr.decision_variable,
                    target=mr.objective_or_constraint,
                    declared_direction=mr.direction,
                    violation_fraction=violation_fraction,
                    max_reversal=max_reversal,
                )
            )

    return tuple(violations)
