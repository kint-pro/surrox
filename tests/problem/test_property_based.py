from __future__ import annotations

import hypothesis.strategies as st
import pytest
from hypothesis import given

from surrox.exceptions import ProblemDefinitionError
from surrox.problem.constraints import DataConstraint, LinearConstraint
from surrox.problem.definition import ProblemDefinition
from surrox.problem.objectives import Objective
from surrox.problem.scenarios import Scenario
from surrox.problem.types import (
    ConstraintOperator,
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
from tests.strategies import (
    categorical_bounds,
    continuous_bounds,
    data_constraints,
    decision_variables,
    integer_bounds,
    linear_constraints,
    objectives,
    ordinal_bounds,
    safe_floats,
    scenarios,
    unique_categories,
    variable_names,
)


class TestContinuousBoundsProperties:
    @given(bounds=continuous_bounds())
    def test_lower_always_less_than_upper(self, bounds: ContinuousBounds) -> None:
        assert bounds.lower < bounds.upper

    @given(bounds=continuous_bounds())
    def test_roundtrip_json(self, bounds: ContinuousBounds) -> None:
        restored = ContinuousBounds.model_validate_json(bounds.model_dump_json())
        assert restored == bounds

    @given(
        lower=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        upper=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    def test_equal_or_inverted_bounds_rejected(self, lower: float, upper: float) -> None:
        from hypothesis import assume

        assume(lower >= upper)
        with pytest.raises(ProblemDefinitionError):
            ContinuousBounds(lower=lower, upper=upper)


class TestIntegerBoundsProperties:
    @given(bounds=integer_bounds())
    def test_lower_always_less_than_upper(self, bounds: IntegerBounds) -> None:
        assert bounds.lower < bounds.upper

    @given(bounds=integer_bounds())
    def test_roundtrip_json(self, bounds: IntegerBounds) -> None:
        restored = IntegerBounds.model_validate_json(bounds.model_dump_json())
        assert restored == bounds

    @given(
        lower=st.integers(min_value=-10000, max_value=10000),
        upper=st.integers(min_value=-10000, max_value=10000),
    )
    def test_equal_or_inverted_bounds_rejected(self, lower: int, upper: int) -> None:
        from hypothesis import assume

        assume(lower >= upper)
        with pytest.raises(ProblemDefinitionError):
            IntegerBounds(lower=lower, upper=upper)


class TestCategoricalBoundsProperties:
    @given(bounds=categorical_bounds())
    def test_roundtrip_json(self, bounds: CategoricalBounds) -> None:
        restored = CategoricalBounds.model_validate_json(bounds.model_dump_json())
        assert restored == bounds

    @given(bounds=categorical_bounds())
    def test_at_least_two_unique_categories(self, bounds: CategoricalBounds) -> None:
        assert len(bounds.categories) >= 2
        assert len(bounds.categories) == len(set(bounds.categories))


class TestOrdinalBoundsProperties:
    @given(bounds=ordinal_bounds())
    def test_roundtrip_json(self, bounds: OrdinalBounds) -> None:
        restored = OrdinalBounds.model_validate_json(bounds.model_dump_json())
        assert restored == bounds

    @given(bounds=ordinal_bounds())
    def test_at_least_two_unique_categories(self, bounds: OrdinalBounds) -> None:
        assert len(bounds.categories) >= 2
        assert len(bounds.categories) == len(set(bounds.categories))


class TestVariableProperties:
    @given(var=decision_variables())
    def test_dtype_matches_bounds_type(self, var: Variable) -> None:
        expected = {
            DType.CONTINUOUS: "continuous",
            DType.INTEGER: "integer",
            DType.CATEGORICAL: "categorical",
            DType.ORDINAL: "ordinal",
        }
        assert var.bounds.type == expected[var.dtype]

    @given(var=decision_variables())
    def test_roundtrip_json(self, var: Variable) -> None:
        restored = Variable.model_validate_json(var.model_dump_json())
        assert restored == var

    @given(var=decision_variables())
    def test_frozen(self, var: Variable) -> None:
        with pytest.raises(Exception):
            var.name = "mutated"  # type: ignore[misc]

    @given(
        dtype=st.sampled_from(DType),
        bounds=st.one_of(
            continuous_bounds(),
            integer_bounds(),
            categorical_bounds(),
            ordinal_bounds(),
        ),
    )
    def test_mismatched_dtype_bounds_rejected(self, dtype: DType, bounds: object) -> None:
        from hypothesis import assume

        dtype_to_bounds_type = {
            DType.CONTINUOUS: "continuous",
            DType.INTEGER: "integer",
            DType.CATEGORICAL: "categorical",
            DType.ORDINAL: "ordinal",
        }
        assume(bounds.type != dtype_to_bounds_type[dtype])  # type: ignore[union-attr]
        with pytest.raises(ProblemDefinitionError, match="requires .* bounds"):
            Variable(name="x", dtype=dtype, role=Role.DECISION, bounds=bounds)  # type: ignore[arg-type]


class TestObjectiveProperties:
    @given(obj=objectives())
    def test_roundtrip_json(self, obj: Objective) -> None:
        restored = Objective.model_validate_json(obj.model_dump_json())
        assert restored == obj


class TestLinearConstraintProperties:
    @given(lc=linear_constraints())
    def test_roundtrip_json(self, lc: LinearConstraint) -> None:
        restored = LinearConstraint.model_validate_json(lc.model_dump_json())
        assert restored == lc

    @given(lc=linear_constraints())
    def test_no_zero_coefficients(self, lc: LinearConstraint) -> None:
        assert all(v != 0.0 for v in lc.coefficients.values())

    @given(lc=linear_constraints())
    def test_coefficients_not_empty(self, lc: LinearConstraint) -> None:
        assert len(lc.coefficients) >= 1

    @given(
        operator=st.sampled_from(ConstraintOperator),
        rhs=safe_floats,
    )
    def test_empty_coefficients_rejected(
        self, operator: ConstraintOperator, rhs: float
    ) -> None:
        with pytest.raises(ProblemDefinitionError, match="coefficients must not be empty"):
            LinearConstraint(name="lc", coefficients={}, operator=operator, rhs=rhs)


class TestDataConstraintProperties:
    @given(dc=data_constraints())
    def test_roundtrip_json(self, dc: DataConstraint) -> None:
        restored = DataConstraint.model_validate_json(dc.model_dump_json())
        assert restored == dc


class TestScenarioProperties:
    @given(sc=scenarios())
    def test_roundtrip_json(self, sc: Scenario) -> None:
        restored = Scenario.model_validate_json(sc.model_dump_json())
        assert restored == sc

    @given(sc=scenarios())
    def test_context_values_not_empty(self, sc: Scenario) -> None:
        assert len(sc.context_values) >= 1

    @given(name=variable_names)
    def test_empty_context_values_rejected(self, name: str) -> None:
        with pytest.raises(ProblemDefinitionError, match="at least one context variable"):
            Scenario(name=name, context_values={})


class TestProblemDefinitionProperties:
    @given(
        vars=st.lists(decision_variables(), min_size=1, max_size=3, unique_by=lambda v: v.name),
        objs=st.lists(objectives(), min_size=1, max_size=3, unique_by=lambda o: o.name),
    )
    def test_decision_plus_context_equals_variables(
        self, vars: list[Variable], objs: list[Objective]
    ) -> None:
        problem = ProblemDefinition(variables=tuple(vars), objectives=tuple(objs))
        combined = problem.decision_variables + problem.context_variables
        assert set(v.name for v in combined) == set(v.name for v in problem.variables)
        assert len(combined) == len(problem.variables)

    @given(
        vars=st.lists(decision_variables(), min_size=1, max_size=3, unique_by=lambda v: v.name),
        objs=st.lists(objectives(), min_size=1, max_size=3, unique_by=lambda o: o.name),
    )
    def test_surrogate_columns_unique(
        self, vars: list[Variable], objs: list[Objective]
    ) -> None:
        problem = ProblemDefinition(variables=tuple(vars), objectives=tuple(objs))
        columns = problem.surrogate_columns
        assert len(columns) == len(set(columns))

    @given(
        vars=st.lists(decision_variables(), min_size=1, max_size=3, unique_by=lambda v: v.name),
        objs=st.lists(objectives(), min_size=1, max_size=3, unique_by=lambda o: o.name),
        dcs=st.lists(data_constraints(), min_size=0, max_size=2, unique_by=lambda c: c.name),
    )
    def test_surrogate_columns_contain_all_objective_columns(
        self, vars: list[Variable], objs: list[Objective], dcs: list[DataConstraint]
    ) -> None:
        problem = ProblemDefinition(
            variables=tuple(vars), objectives=tuple(objs), data_constraints=tuple(dcs)
        )
        obj_columns = {o.column for o in objs}
        assert obj_columns <= set(problem.surrogate_columns)

    @given(
        vars=st.lists(decision_variables(), min_size=1, max_size=3, unique_by=lambda v: v.name),
        objs=st.lists(objectives(), min_size=1, max_size=3, unique_by=lambda o: o.name),
        dcs=st.lists(data_constraints(), min_size=1, max_size=2, unique_by=lambda c: c.name),
    )
    def test_surrogate_columns_contain_all_data_constraint_columns(
        self, vars: list[Variable], objs: list[Objective], dcs: list[DataConstraint]
    ) -> None:
        problem = ProblemDefinition(
            variables=tuple(vars), objectives=tuple(objs), data_constraints=tuple(dcs)
        )
        dc_columns = {c.column for c in dcs}
        assert dc_columns <= set(problem.surrogate_columns)

    @given(
        vars=st.lists(decision_variables(), min_size=1, max_size=3, unique_by=lambda v: v.name),
        objs=st.lists(objectives(), min_size=1, max_size=3, unique_by=lambda o: o.name),
        target_name=variable_names,
    )
    def test_monotonic_constraints_for_unknown_target_empty(
        self, vars: list[Variable], objs: list[Objective], target_name: str
    ) -> None:
        problem = ProblemDefinition(variables=tuple(vars), objectives=tuple(objs))
        assert problem.monotonic_constraints_for(target_name) == {}

    @given(
        vars=st.lists(decision_variables(), min_size=1, max_size=3, unique_by=lambda v: v.name),
        objs=st.lists(objectives(), min_size=1, max_size=3, unique_by=lambda o: o.name),
    )
    def test_roundtrip_json(
        self, vars: list[Variable], objs: list[Objective]
    ) -> None:
        problem = ProblemDefinition(variables=tuple(vars), objectives=tuple(objs))
        restored = ProblemDefinition.model_validate_json(problem.model_dump_json())
        assert restored == problem

    @given(objs=st.lists(objectives(), min_size=1, max_size=3, unique_by=lambda o: o.name))
    def test_no_decision_variable_rejected(self, objs: list[Objective]) -> None:
        ctx = Variable(
            name="ctx_only",
            dtype=DType.CATEGORICAL,
            role=Role.CONTEXT,
            bounds=CategoricalBounds(categories=("a", "b")),
        )
        with pytest.raises(ProblemDefinitionError, match="at least one decision variable"):
            ProblemDefinition(variables=(ctx,), objectives=tuple(objs))

    @given(
        vars=st.lists(decision_variables(), min_size=1, max_size=3, unique_by=lambda v: v.name),
    )
    def test_no_objectives_rejected(self, vars: list[Variable]) -> None:
        with pytest.raises(ProblemDefinitionError, match="at least one objective"):
            ProblemDefinition(variables=tuple(vars), objectives=())
