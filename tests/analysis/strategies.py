from __future__ import annotations

import hypothesis.strategies as st

from surrox.analysis.config import AnalysisConfig
from surrox.analysis.summary import (
    BaselineComparison,
    ConstraintStatus,
    ExtrapolationWarning,
    MonotonicityViolation,
    SolutionSummary,
    SurrogateQuality,
)
from surrox.analysis.types import ConstraintStatusKind
from surrox.analysis.what_if import WhatIfPrediction
from surrox.optimizer.result import ConstraintEvaluation
from surrox.problem.types import (
    ConstraintOperator,
    ConstraintSeverity,
    Direction,
    MonotonicDirection,
)

safe_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)

positive_floats = st.floats(
    min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False
)

objective_names = st.from_regex(r"obj_[a-z]{1,5}", fullmatch=True)
column_names = st.from_regex(r"col_[a-z]{1,5}", fullmatch=True)
variable_names = st.from_regex(r"[a-z][a-z0-9_]{0,9}", fullmatch=True)


@st.composite
def analysis_configs(draw: st.DrawFn) -> AnalysisConfig:
    lo = draw(st.floats(min_value=0.01, max_value=0.49, allow_nan=False, allow_infinity=False))
    hi = draw(st.floats(min_value=lo + 0.01, max_value=0.99, allow_nan=False, allow_infinity=False))
    return AnalysisConfig(
        shap_background_size=draw(st.integers(min_value=10, max_value=10000)),
        pdp_grid_resolution=draw(st.integers(min_value=10, max_value=500)),
        pdp_percentiles=(lo, hi),
        monotonicity_check_resolution=draw(st.integers(min_value=10, max_value=500)),
    )


@st.composite
def constraint_evaluations(draw: st.DrawFn) -> ConstraintEvaluation:
    return ConstraintEvaluation(
        name=draw(st.from_regex(r"[a-z]{1,10}", fullmatch=True)),
        prediction=draw(safe_floats),
        violation=draw(positive_floats),
        severity=draw(st.sampled_from(ConstraintSeverity)),
    )


@st.composite
def constraint_statuses(draw: st.DrawFn) -> ConstraintStatus:
    return ConstraintStatus(
        evaluation=draw(constraint_evaluations()),
        status=draw(st.sampled_from(ConstraintStatusKind)),
        margin=draw(safe_floats),
    )


@st.composite
def solution_summaries(draw: st.DrawFn) -> SolutionSummary:
    n_feasible = draw(st.integers(min_value=0, max_value=100))
    n_infeasible = draw(st.integers(min_value=0, max_value=100))
    obj_dict = draw(st.dictionaries(objective_names, safe_floats, min_size=1, max_size=3))
    compromise = draw(st.one_of(st.none(), st.just(obj_dict)))
    return SolutionSummary(
        n_feasible=n_feasible,
        n_infeasible=n_infeasible,
        best_objectives=obj_dict,
        compromise_objectives=compromise,
        hypervolume=draw(st.one_of(st.none(), positive_floats)),
    )


@st.composite
def baseline_comparisons(draw: st.DrawFn) -> BaselineComparison:
    names = draw(st.lists(objective_names, min_size=1, max_size=3, unique=True))
    recommended = {n: draw(safe_floats) for n in names}
    historical = {n: draw(safe_floats) for n in names}
    improvement = {n: draw(safe_floats) for n in names}
    return BaselineComparison(
        recommended_objectives=recommended,
        historical_best_per_objective=historical,
        improvement=improvement,
    )


@st.composite
def surrogate_qualities(draw: st.DrawFn) -> SurrogateQuality:
    return SurrogateQuality(
        column=draw(column_names),
        cv_rmse=draw(positive_floats),
        conformal_coverage=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        ensemble_size=draw(st.integers(min_value=1, max_value=20)),
        warning=draw(st.one_of(st.none(), st.text(min_size=1, max_size=50))),
    )


@st.composite
def extrapolation_warnings(draw: st.DrawFn) -> ExtrapolationWarning:
    return ExtrapolationWarning(
        point_index=draw(st.integers(min_value=0, max_value=100)),
        distance=draw(positive_floats),
    )


@st.composite
def monotonicity_violations(draw: st.DrawFn) -> MonotonicityViolation:
    return MonotonicityViolation(
        decision_variable=draw(variable_names),
        target=draw(st.from_regex(r"[a-z]{1,10}", fullmatch=True)),
        declared_direction=draw(st.sampled_from(MonotonicDirection)),
        violation_fraction=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        max_reversal=draw(positive_floats),
    )


@st.composite
def what_if_predictions(draw: st.DrawFn) -> WhatIfPrediction:
    return WhatIfPrediction(
        predicted=draw(safe_floats),
        lower=draw(safe_floats),
        upper=draw(safe_floats),
        recommended_value=draw(safe_floats),
        historical_mean=draw(safe_floats),
    )
