from __future__ import annotations

import hypothesis.strategies as st

from surrox.problem.constraints import DataConstraint, LinearConstraint
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

safe_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)

safe_nonzero_floats = safe_floats.filter(lambda x: x != 0.0)

variable_names = st.from_regex(r"[a-z][a-z0-9_]{0,9}", fullmatch=True)

category_names = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=10,
)

unique_categories = st.lists(
    category_names, min_size=2, max_size=10, unique=True
).map(tuple)


@st.composite
def continuous_bounds(draw: st.DrawFn) -> ContinuousBounds:
    lower = draw(st.floats(min_value=-1e6, max_value=1e6 - 1e-9, allow_nan=False, allow_infinity=False))
    upper = draw(st.floats(min_value=lower + 1e-9, max_value=1e6, allow_nan=False, allow_infinity=False))
    return ContinuousBounds(lower=lower, upper=upper)


@st.composite
def integer_bounds(draw: st.DrawFn) -> IntegerBounds:
    lower = draw(st.integers(min_value=-10000, max_value=9999))
    upper = draw(st.integers(min_value=lower + 1, max_value=10000))
    return IntegerBounds(lower=lower, upper=upper)


@st.composite
def categorical_bounds(draw: st.DrawFn) -> CategoricalBounds:
    cats = draw(unique_categories)
    return CategoricalBounds(categories=cats)


@st.composite
def ordinal_bounds(draw: st.DrawFn) -> OrdinalBounds:
    cats = draw(unique_categories)
    return OrdinalBounds(categories=cats)


def any_bounds() -> st.SearchStrategy:
    return st.one_of(
        continuous_bounds(),
        integer_bounds(),
        categorical_bounds(),
        ordinal_bounds(),
    )


_DTYPE_BOUNDS_MAP: dict[DType, st.SearchStrategy] = {
    DType.CONTINUOUS: continuous_bounds(),
    DType.INTEGER: integer_bounds(),
    DType.CATEGORICAL: categorical_bounds(),
    DType.ORDINAL: ordinal_bounds(),
}


@st.composite
def variables(draw: st.DrawFn, role: Role | None = None) -> Variable:
    dtype = draw(st.sampled_from(DType))
    chosen_role = role if role is not None else draw(st.sampled_from(Role))
    bounds = draw(_DTYPE_BOUNDS_MAP[dtype])
    name = draw(variable_names)
    return Variable(name=name, dtype=dtype, role=chosen_role, bounds=bounds)


def decision_variables() -> st.SearchStrategy[Variable]:
    return variables(role=Role.DECISION)


def context_variables() -> st.SearchStrategy[Variable]:
    return variables(role=Role.CONTEXT)


@st.composite
def objectives(draw: st.DrawFn) -> Objective:
    return Objective(
        name=draw(st.from_regex(r"obj_[a-z]{1,5}", fullmatch=True)),
        direction=draw(st.sampled_from(Direction)),
        column=draw(st.from_regex(r"col_[a-z]{1,5}", fullmatch=True)),
        reference_value=draw(st.one_of(st.none(), safe_floats)),
    )


@st.composite
def _severity_and_weight(
    draw: st.DrawFn,
) -> tuple[ConstraintSeverity, float | None]:
    severity = draw(st.sampled_from(ConstraintSeverity))
    if severity == ConstraintSeverity.SOFT:
        weight = draw(
            st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False)
        )
        return severity, weight
    return severity, None


@st.composite
def linear_constraints(draw: st.DrawFn) -> LinearConstraint:
    coefficients = draw(
        st.dictionaries(variable_names, safe_nonzero_floats, min_size=1, max_size=5)
    )
    severity, penalty_weight = draw(_severity_and_weight())
    return LinearConstraint(
        name=draw(st.from_regex(r"lc_[a-z]{1,5}", fullmatch=True)),
        coefficients=coefficients,
        operator=draw(st.sampled_from(ConstraintOperator)),
        rhs=draw(safe_floats),
        severity=severity,
        penalty_weight=penalty_weight,
    )


@st.composite
def data_constraints(draw: st.DrawFn) -> DataConstraint:
    operator = draw(st.sampled_from(ConstraintOperator))
    tolerance = (
        draw(st.floats(min_value=1e-6, max_value=1e3, allow_nan=False, allow_infinity=False))
        if operator == ConstraintOperator.EQ
        else None
    )
    severity, penalty_weight = draw(_severity_and_weight())
    return DataConstraint(
        name=draw(st.from_regex(r"dc_[a-z]{1,5}", fullmatch=True)),
        column=draw(st.from_regex(r"col_[a-z]{1,5}", fullmatch=True)),
        operator=operator,
        limit=draw(safe_floats),
        tolerance=tolerance,
        severity=severity,
        penalty_weight=penalty_weight,
    )


@st.composite
def monotonic_relations(draw: st.DrawFn) -> MonotonicRelation:
    return MonotonicRelation(
        decision_variable=draw(variable_names),
        objective_or_constraint=draw(st.from_regex(r"[a-z][a-z0-9_]{0,9}", fullmatch=True)),
        direction=draw(st.sampled_from(MonotonicDirection)),
    )


@st.composite
def scenarios(draw: st.DrawFn) -> Scenario:
    context_values = draw(
        st.dictionaries(variable_names, st.text(min_size=1, max_size=10), min_size=1, max_size=5)
    )
    return Scenario(
        name=draw(st.from_regex(r"sc_[a-z]{1,5}", fullmatch=True)),
        context_values=context_values,
    )
