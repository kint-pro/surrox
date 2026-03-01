from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Choice, Integer, Real

from surrox.exceptions import OptimizationError
from surrox.optimizer.config import OptimizerConfig
from surrox.optimizer.extrapolation import ExtrapolationGate
from surrox.optimizer.result import ConstraintEvaluation
from surrox.problem.constraints import LinearConstraint
from surrox.problem.definition import ProblemDefinition
from surrox.problem.scenarios import Scenario
from surrox.problem.types import (
    ConstraintOperator,
    ConstraintSeverity,
    Direction,
    DType,
)
from surrox.surrogate.manager import SurrogateManager

_DEFAULT_PENALTY_MULTIPLIER = 100.0


def _compute_extrapolation_penalty(
    problem: ProblemDefinition, training_df: pd.DataFrame
) -> float:
    ranges: list[float] = []
    for obj in problem.objectives:
        col = training_df[obj.column]
        obj_range = float(col.max() - col.min())
        if obj_range > 0:
            ranges.append(obj_range)
    if not ranges:
        return _DEFAULT_PENALTY_MULTIPLIER
    return _DEFAULT_PENALTY_MULTIPLIER * max(ranges)


def _has_mixed_variables(problem: ProblemDefinition) -> bool:
    return any(
        v.dtype in (DType.CATEGORICAL, DType.ORDINAL)
        for v in problem.decision_variables
    )


def _build_pymoo_variables(
    problem: ProblemDefinition,
) -> dict[str, Real | Integer | Choice]:
    pymoo_vars: dict[str, Real | Integer | Choice] = {}
    for var in problem.decision_variables:
        if var.dtype == DType.CONTINUOUS:
            pymoo_vars[var.name] = Real(
                bounds=(var.bounds.lower, var.bounds.upper)  # type: ignore[union-attr]
            )
        elif var.dtype == DType.INTEGER:
            pymoo_vars[var.name] = Integer(
                bounds=(var.bounds.lower, var.bounds.upper)  # type: ignore[union-attr]
            )
        elif var.dtype in (DType.CATEGORICAL, DType.ORDINAL):
            pymoo_vars[var.name] = Choice(
                options=list(var.bounds.categories)  # type: ignore[union-attr]
            )
    return pymoo_vars


def _count_constraints(problem: ProblemDefinition) -> int:
    n = 0
    for lc in problem.hard_linear_constraints:
        n += 2 if lc.operator == ConstraintOperator.EQ else 1
    for dc in problem.hard_data_constraints:
        n += 2 if dc.operator == ConstraintOperator.EQ else 1
    return n


class SurroxProblem(ElementwiseProblem):
    def __init__(
        self,
        problem: ProblemDefinition,
        surrogate_manager: SurrogateManager,
        extrapolation_gate: ExtrapolationGate,
        config: OptimizerConfig,
        extrapolation_penalty: float,
        scenario: Scenario | None = None,
    ) -> None:
        if problem.context_variables and scenario is None:
            raise OptimizationError(
                "problem has context variables but no scenario was provided"
            )

        self._problem = problem
        self._surrogate = surrogate_manager
        self._gate = extrapolation_gate
        self._config = config
        self._extrapolation_penalty = extrapolation_penalty
        self._scenario = scenario
        self._is_mixed = _has_mixed_variables(problem)

        self._point_diagnostics: list[
            tuple[tuple[ConstraintEvaluation, ...], float, bool]
        ] = []

        n_obj = len(problem.objectives)
        n_constr = _count_constraints(problem)

        if self._is_mixed:
            pymoo_vars = _build_pymoo_variables(problem)
            super().__init__(vars=pymoo_vars, n_obj=n_obj, n_ieq_constr=n_constr)
        else:
            decision_vars = problem.decision_variables
            xl = np.array([v.bounds.lower for v in decision_vars], dtype=np.float64)  # type: ignore[union-attr]
            xu = np.array([v.bounds.upper for v in decision_vars], dtype=np.float64)  # type: ignore[union-attr]
            super().__init__(
                n_var=len(decision_vars),
                n_obj=n_obj,
                n_ieq_constr=n_constr,
                xl=xl,
                xu=xu,
            )

    @property
    def point_diagnostics(
        self,
    ) -> list[tuple[tuple[ConstraintEvaluation, ...], float, bool]]:
        return self._point_diagnostics

    def clear_diagnostics(self) -> None:
        self._point_diagnostics.clear()

    def _evaluate(
        self,
        X: dict[str, Any] | NDArray,
        out: dict[str, Any],
        *args: object,
        **kwargs: object,
    ) -> None:
        row = self._x_to_row(X)

        if self._scenario is not None:
            for key, value in self._scenario.context_values.items():
                row[key] = value

        df = pd.DataFrame([row])

        objectives = self._evaluate_objectives(df)
        constraint_evals, G, soft_penalty = self._evaluate_constraints(df)

        extrap_mask, extrap_distances = self._gate.evaluate(df)
        is_extrapolating = bool(extrap_mask[0])
        extrap_distance = float(extrap_distances[0])

        if is_extrapolating:
            objectives = objectives + self._extrapolation_penalty
        objectives = objectives + soft_penalty

        out["F"] = objectives
        if len(G) > 0:
            out["G"] = G

        self._point_diagnostics.append(
            (tuple(constraint_evals), extrap_distance, is_extrapolating)
        )

    def _x_to_row(self, X: dict[str, Any] | NDArray) -> dict[str, Any]:
        if isinstance(X, dict):
            return dict(X)
        decision_vars = self._problem.decision_variables
        return {var.name: float(X[i]) for i, var in enumerate(decision_vars)}

    def _evaluate_objectives(self, df: pd.DataFrame) -> NDArray[np.float64]:
        predictions = self._surrogate.evaluate(df)
        values = np.zeros(len(self._problem.objectives), dtype=np.float64)
        for i, obj in enumerate(self._problem.objectives):
            pred = float(predictions[obj.column][0])
            values[i] = -pred if obj.direction == Direction.MAXIMIZE else pred
        return values

    def _evaluate_constraints(
        self, df: pd.DataFrame
    ) -> tuple[list[ConstraintEvaluation], NDArray[np.float64], float]:
        evals: list[ConstraintEvaluation] = []
        g_values: list[float] = []
        soft_penalty = 0.0

        if self._problem.data_constraints:
            uncertainty = self._surrogate.evaluate_with_uncertainty(
                df, coverage=self._config.constraint_confidence
            )

            for dc in self._problem.data_constraints:
                pred_data = uncertainty[dc.column]
                point_pred = float(pred_data.mean[0])
                lower = float(pred_data.lower[0])
                upper = float(pred_data.upper[0])

                if dc.operator == ConstraintOperator.LE:
                    violation = upper - dc.limit
                elif dc.operator == ConstraintOperator.GE:
                    violation = dc.limit - lower
                elif dc.operator == ConstraintOperator.EQ:
                    assert dc.tolerance is not None
                    v1 = lower - (dc.limit + dc.tolerance)
                    v2 = (dc.limit - dc.tolerance) - upper
                    violation = max(v1, v2)
                else:
                    raise OptimizationError(f"unknown operator: {dc.operator}")

                evals.append(
                    ConstraintEvaluation(
                        name=dc.name,
                        violation=violation,
                        prediction=point_pred,
                        severity=dc.severity,
                        lower_bound=lower,
                        upper_bound=upper,
                    )
                )

                if dc.severity == ConstraintSeverity.HARD:
                    if dc.operator == ConstraintOperator.EQ:
                        assert dc.tolerance is not None
                        g_values.append(lower - (dc.limit + dc.tolerance))
                        g_values.append((dc.limit - dc.tolerance) - upper)
                    else:
                        g_values.append(violation)
                else:
                    assert dc.penalty_weight is not None
                    if dc.operator == ConstraintOperator.EQ:
                        assert dc.tolerance is not None
                        v1 = lower - (dc.limit + dc.tolerance)
                        v2 = (dc.limit - dc.tolerance) - upper
                        soft_penalty += dc.penalty_weight * (
                            max(0.0, v1) + max(0.0, v2)
                        )
                    else:
                        soft_penalty += dc.penalty_weight * max(0.0, violation)

        for lc in self._problem.linear_constraints:
            row = df.iloc[0]
            lhs = sum(
                coeff * float(row[var_name])
                for var_name, coeff in lc.coefficients.items()
            )
            violation = _linear_constraint_violation(lc, lhs)
            evals.append(
                ConstraintEvaluation(
                    name=lc.name,
                    violation=violation,
                    prediction=lhs,
                    severity=lc.severity,
                )
            )

            if lc.severity == ConstraintSeverity.HARD:
                if lc.operator == ConstraintOperator.EQ:
                    assert lc.tolerance is not None
                    g_values.append(lhs - (lc.rhs + lc.tolerance))
                    g_values.append((lc.rhs - lc.tolerance) - lhs)
                else:
                    g_values.append(violation)
            else:
                assert lc.penalty_weight is not None
                soft_penalty += lc.penalty_weight * max(0.0, violation)

        return evals, np.array(g_values, dtype=np.float64), soft_penalty


def _linear_constraint_violation(lc: LinearConstraint, lhs: float) -> float:
    if lc.operator == ConstraintOperator.LE:
        return lhs - lc.rhs
    elif lc.operator == ConstraintOperator.GE:
        return lc.rhs - lhs
    else:
        assert lc.tolerance is not None
        return abs(lhs - lc.rhs) - lc.tolerance
