import pytest
from pydantic import ValidationError

from surrox.exceptions import ProblemDefinitionError
from surrox.problem.constraints import DataConstraint, LinearConstraint
from surrox.problem.definition import ProblemDefinition
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


def make_decision_var(name: str = "x") -> Variable:
    return Variable(
        name=name,
        dtype=DType.CONTINUOUS,
        role=Role.DECISION,
        bounds=ContinuousBounds(lower=0.0, upper=1.0),
    )


def make_context_var(name: str = "ctx") -> Variable:
    return Variable(
        name=name,
        dtype=DType.CATEGORICAL,
        role=Role.CONTEXT,
        bounds=CategoricalBounds(categories=("a", "b")),
    )


def make_objective(name: str = "obj", column: str = "obj_col") -> Objective:
    return Objective(name=name, direction=Direction.MINIMIZE, column=column)


class TestProblemDefinitionMinimal:
    def test_create_minimal_problem(self) -> None:
        problem = ProblemDefinition(
            variables=(make_decision_var(),),
            objectives=(make_objective(),),
        )
        assert len(problem.variables) == 1
        assert len(problem.objectives) == 1
        assert problem.linear_constraints == ()
        assert problem.data_constraints == ()
        assert problem.monotonic_relations == ()
        assert problem.scenarios == ()

    def test_create_full_problem(self, full_problem: ProblemDefinition) -> None:
        assert len(full_problem.variables) == 2
        assert len(full_problem.objectives) == 2
        assert len(full_problem.linear_constraints) == 1
        assert len(full_problem.data_constraints) == 1
        assert len(full_problem.monotonic_relations) == 1
        assert len(full_problem.scenarios) == 1


class TestProblemDefinitionValidation:
    def test_no_objectives_raises(self) -> None:
        with pytest.raises(ProblemDefinitionError, match="at least one objective is required"):
            ProblemDefinition(
                variables=(make_decision_var(),),
                objectives=(),
            )

    def test_no_decision_variable_raises(self) -> None:
        with pytest.raises(
            ProblemDefinitionError, match="at least one decision variable is required"
        ):
            ProblemDefinition(
                variables=(make_context_var(),),
                objectives=(make_objective(),),
            )

    def test_duplicate_variable_names_raises(self) -> None:
        with pytest.raises(ProblemDefinitionError, match="variable names must be unique"):
            ProblemDefinition(
                variables=(make_decision_var("x"), make_decision_var("x")),
                objectives=(make_objective(),),
            )

    def test_duplicate_objective_names_raises(self) -> None:
        with pytest.raises(ProblemDefinitionError, match="objective names must be unique"):
            ProblemDefinition(
                variables=(make_decision_var(),),
                objectives=(make_objective("obj"), make_objective("obj")),
            )

    def test_duplicate_constraint_names_across_types_raises(self) -> None:
        lc = LinearConstraint(
            name="shared_name",
            coefficients={"x": 1.0},
            operator=ConstraintOperator.LE,
            rhs=1.0,
        )
        dc = DataConstraint(
            name="shared_name",
            column="some_col",
            operator=ConstraintOperator.LE,
            limit=1.0,
        )
        with pytest.raises(ProblemDefinitionError, match="constraint names must be unique"):
            ProblemDefinition(
                variables=(make_decision_var("x"),),
                objectives=(make_objective(),),
                linear_constraints=(lc,),
                data_constraints=(dc,),
            )

    def test_duplicate_scenario_names_raises(self) -> None:
        ctx = make_context_var("ctx")
        s1 = Scenario(name="same", context_values={"ctx": "a"})
        s2 = Scenario(name="same", context_values={"ctx": "b"})
        with pytest.raises(ProblemDefinitionError, match="scenario names must be unique"):
            ProblemDefinition(
                variables=(make_decision_var(), ctx),
                objectives=(make_objective(),),
                scenarios=(s1, s2),
            )

    def test_linear_constraint_unknown_variable_raises(self) -> None:
        lc = LinearConstraint(
            name="bad_ref",
            coefficients={"unknown_var": 1.0},
            operator=ConstraintOperator.LE,
            rhs=1.0,
        )
        with pytest.raises(
            ProblemDefinitionError,
            match="unknown or non-decision variable 'unknown_var'",
        ):
            ProblemDefinition(
                variables=(make_decision_var("x"),),
                objectives=(make_objective(),),
                linear_constraints=(lc,),
            )

    def test_linear_constraint_on_context_variable_raises(self) -> None:
        lc = LinearConstraint(
            name="ctx_ref",
            coefficients={"ctx": 1.0},
            operator=ConstraintOperator.LE,
            rhs=1.0,
        )
        with pytest.raises(
            ProblemDefinitionError,
            match="unknown or non-decision variable 'ctx'",
        ):
            ProblemDefinition(
                variables=(make_decision_var("x"), make_context_var("ctx")),
                objectives=(make_objective(),),
                linear_constraints=(lc,),
            )

    def test_linear_constraint_valid_reference(self) -> None:
        lc = LinearConstraint(
            name="valid",
            coefficients={"x": 1.0},
            operator=ConstraintOperator.LE,
            rhs=1.0,
        )
        problem = ProblemDefinition(
            variables=(make_decision_var("x"),),
            objectives=(make_objective(),),
            linear_constraints=(lc,),
        )
        assert len(problem.linear_constraints) == 1

    def test_monotonic_relation_unknown_variable_raises(self) -> None:
        mr = MonotonicRelation(
            decision_variable="ghost_var",
            objective_or_constraint="obj",
            direction=MonotonicDirection.INCREASING,
        )
        with pytest.raises(
            ProblemDefinitionError,
            match="unknown or non-decision variable 'ghost_var'",
        ):
            ProblemDefinition(
                variables=(make_decision_var("x"),),
                objectives=(make_objective("obj"),),
                monotonic_relations=(mr,),
            )

    def test_monotonic_relation_on_context_variable_raises(self) -> None:
        ctx = make_context_var("ctx")
        mr = MonotonicRelation(
            decision_variable="ctx",
            objective_or_constraint="obj",
            direction=MonotonicDirection.INCREASING,
        )
        with pytest.raises(
            ProblemDefinitionError,
            match="unknown or non-decision variable 'ctx'",
        ):
            ProblemDefinition(
                variables=(make_decision_var("x"), ctx),
                objectives=(make_objective("obj"),),
                monotonic_relations=(mr,),
            )

    def test_monotonic_relation_on_categorical_variable_raises(self) -> None:
        cat_decision = Variable(
            name="method",
            dtype=DType.CATEGORICAL,
            role=Role.DECISION,
            bounds=CategoricalBounds(categories=("a", "b", "c")),
        )
        mr = MonotonicRelation(
            decision_variable="method",
            objective_or_constraint="obj",
            direction=MonotonicDirection.INCREASING,
        )
        with pytest.raises(
            ProblemDefinitionError,
            match="non-numeric variable 'method'",
        ):
            ProblemDefinition(
                variables=(make_decision_var("x"), cat_decision),
                objectives=(make_objective("obj"),),
                monotonic_relations=(mr,),
            )

    def test_monotonic_relation_on_ordinal_variable_raises(self) -> None:
        ord_decision = Variable(
            name="quality",
            dtype=DType.ORDINAL,
            role=Role.DECISION,
            bounds=OrdinalBounds(categories=("low", "high")),
        )
        mr = MonotonicRelation(
            decision_variable="quality",
            objective_or_constraint="obj",
            direction=MonotonicDirection.INCREASING,
        )
        with pytest.raises(
            ProblemDefinitionError,
            match="non-numeric variable 'quality'",
        ):
            ProblemDefinition(
                variables=(make_decision_var("x"), ord_decision),
                objectives=(make_objective("obj"),),
                monotonic_relations=(mr,),
            )

    def test_monotonic_relation_unknown_target_raises(self) -> None:
        mr = MonotonicRelation(
            decision_variable="x",
            objective_or_constraint="nonexistent_target",
            direction=MonotonicDirection.INCREASING,
        )
        with pytest.raises(
            ProblemDefinitionError, match="unknown target 'nonexistent_target'"
        ):
            ProblemDefinition(
                variables=(make_decision_var("x"),),
                objectives=(make_objective("obj"),),
                monotonic_relations=(mr,),
            )

    def test_contradictory_monotonic_relations_raises(self) -> None:
        mr1 = MonotonicRelation(
            decision_variable="x",
            objective_or_constraint="obj",
            direction=MonotonicDirection.INCREASING,
        )
        mr2 = MonotonicRelation(
            decision_variable="x",
            objective_or_constraint="obj",
            direction=MonotonicDirection.DECREASING,
        )
        with pytest.raises(ProblemDefinitionError, match="contradictory monotonic relations"):
            ProblemDefinition(
                variables=(make_decision_var("x"),),
                objectives=(make_objective("obj"),),
                monotonic_relations=(mr1, mr2),
            )

    def test_duplicate_monotonic_relations_same_direction_accepted(self) -> None:
        mr1 = MonotonicRelation(
            decision_variable="x",
            objective_or_constraint="obj",
            direction=MonotonicDirection.INCREASING,
        )
        mr2 = MonotonicRelation(
            decision_variable="x",
            objective_or_constraint="obj",
            direction=MonotonicDirection.INCREASING,
        )
        problem = ProblemDefinition(
            variables=(make_decision_var("x"),),
            objectives=(make_objective("obj"),),
            monotonic_relations=(mr1, mr2),
        )
        assert len(problem.monotonic_relations) == 2

    def test_monotonic_relation_can_target_data_constraint(self) -> None:
        dc = DataConstraint(
            name="pressure_limit",
            column="pressure_col",
            operator=ConstraintOperator.LE,
            limit=100.0,
        )
        mr = MonotonicRelation(
            decision_variable="x",
            objective_or_constraint="pressure_limit",
            direction=MonotonicDirection.INCREASING,
        )
        problem = ProblemDefinition(
            variables=(make_decision_var("x"),),
            objectives=(make_objective(),),
            data_constraints=(dc,),
            monotonic_relations=(mr,),
        )
        constraints = problem.monotonic_constraints_for("pressure_col")
        assert constraints == {"x": MonotonicDirection.INCREASING}

    def test_scenario_unknown_context_variable_raises(self) -> None:
        s = Scenario(name="bad_scenario", context_values={"nonexistent": "value"})
        with pytest.raises(
            ProblemDefinitionError,
            match="unknown context variable 'nonexistent'",
        ):
            ProblemDefinition(
                variables=(make_decision_var("x"),),
                objectives=(make_objective(),),
                scenarios=(s,),
            )

    def test_scenario_non_numeric_value_for_continuous_variable_raises(self) -> None:
        ctx = Variable(
            name="wind",
            dtype=DType.CONTINUOUS,
            role=Role.CONTEXT,
            bounds=ContinuousBounds(lower=0.0, upper=50.0),
        )
        s = Scenario(name="bad", context_values={"wind": "fast"})
        with pytest.raises(ProblemDefinitionError, match="expects numeric value"):
            ProblemDefinition(
                variables=(make_decision_var("x"), ctx),
                objectives=(make_objective(),),
                scenarios=(s,),
            )

    def test_scenario_value_outside_bounds_raises(self) -> None:
        ctx = Variable(
            name="wind",
            dtype=DType.CONTINUOUS,
            role=Role.CONTEXT,
            bounds=ContinuousBounds(lower=0.0, upper=50.0),
        )
        s = Scenario(name="bad", context_values={"wind": 100.0})
        with pytest.raises(ProblemDefinitionError, match="outside bounds"):
            ProblemDefinition(
                variables=(make_decision_var("x"), ctx),
                objectives=(make_objective(),),
                scenarios=(s,),
            )

    def test_scenario_invalid_category_raises(self) -> None:
        ctx = make_context_var("ctx")
        s = Scenario(name="bad", context_values={"ctx": "nonexistent"})
        with pytest.raises(ProblemDefinitionError, match="not in categories"):
            ProblemDefinition(
                variables=(make_decision_var("x"), ctx),
                objectives=(make_objective(),),
                scenarios=(s,),
            )

    def test_scenario_referencing_decision_variable_raises(self) -> None:
        s = Scenario(name="bad", context_values={"x": 0.5})
        with pytest.raises(
            ProblemDefinitionError, match="unknown context variable 'x'"
        ):
            ProblemDefinition(
                variables=(make_decision_var("x"),),
                objectives=(make_objective(),),
                scenarios=(s,),
            )

    def test_scenario_non_integer_value_for_integer_variable_raises(self) -> None:
        ctx = Variable(
            name="level",
            dtype=DType.INTEGER,
            role=Role.CONTEXT,
            bounds=IntegerBounds(lower=0, upper=10),
        )
        s = Scenario(name="bad", context_values={"level": 2.5})
        with pytest.raises(ProblemDefinitionError, match="expects integer value"):
            ProblemDefinition(
                variables=(make_decision_var("x"), ctx),
                objectives=(make_objective(),),
                scenarios=(s,),
            )

    def test_scenario_integer_value_outside_bounds_raises(self) -> None:
        ctx = Variable(
            name="level",
            dtype=DType.INTEGER,
            role=Role.CONTEXT,
            bounds=IntegerBounds(lower=0, upper=10),
        )
        s = Scenario(name="bad", context_values={"level": 20})
        with pytest.raises(ProblemDefinitionError, match="outside bounds"):
            ProblemDefinition(
                variables=(make_decision_var("x"), ctx),
                objectives=(make_objective(),),
                scenarios=(s,),
            )

    def test_scenario_valid_integer_context_value_accepted(self) -> None:
        ctx = Variable(
            name="level",
            dtype=DType.INTEGER,
            role=Role.CONTEXT,
            bounds=IntegerBounds(lower=0, upper=10),
        )
        s = Scenario(name="ok", context_values={"level": 5})
        problem = ProblemDefinition(
            variables=(make_decision_var("x"), ctx),
            objectives=(make_objective(),),
            scenarios=(s,),
        )
        assert len(problem.scenarios) == 1

    def test_scenario_invalid_ordinal_category_raises(self) -> None:
        ctx = Variable(
            name="quality",
            dtype=DType.ORDINAL,
            role=Role.CONTEXT,
            bounds=OrdinalBounds(categories=("low", "medium", "high")),
        )
        s = Scenario(name="bad", context_values={"quality": "extreme"})
        with pytest.raises(ProblemDefinitionError, match="not in categories"):
            ProblemDefinition(
                variables=(make_decision_var("x"), ctx),
                objectives=(make_objective(),),
                scenarios=(s,),
            )

    def test_scenario_valid_ordinal_context_value_accepted(self) -> None:
        ctx = Variable(
            name="quality",
            dtype=DType.ORDINAL,
            role=Role.CONTEXT,
            bounds=OrdinalBounds(categories=("low", "medium", "high")),
        )
        s = Scenario(name="ok", context_values={"quality": "medium"})
        problem = ProblemDefinition(
            variables=(make_decision_var("x"), ctx),
            objectives=(make_objective(),),
            scenarios=(s,),
        )
        assert len(problem.scenarios) == 1

    def test_scenario_valid_values_accepted(self) -> None:
        ctx = Variable(
            name="wind",
            dtype=DType.CONTINUOUS,
            role=Role.CONTEXT,
            bounds=ContinuousBounds(lower=0.0, upper=50.0),
        )
        s = Scenario(name="normal", context_values={"wind": 25.0})
        problem = ProblemDefinition(
            variables=(make_decision_var("x"), ctx),
            objectives=(make_objective(),),
            scenarios=(s,),
        )
        assert len(problem.scenarios) == 1


class TestProblemDefinitionProperties:
    def test_decision_variables_property(self, full_problem: ProblemDefinition) -> None:
        decision_vars = full_problem.decision_variables
        assert all(v.role == Role.DECISION for v in decision_vars)
        assert len(decision_vars) == 1
        assert decision_vars[0].name == "temperature"

    def test_context_variables_property(self, full_problem: ProblemDefinition) -> None:
        context_vars = full_problem.context_variables
        assert all(v.role == Role.CONTEXT for v in context_vars)
        assert len(context_vars) == 1
        assert context_vars[0].name == "mode"

    def test_context_variables_empty_when_all_decision(
        self, minimal_problem: ProblemDefinition
    ) -> None:
        assert minimal_problem.context_variables == ()

    def test_surrogate_columns_contains_objective_columns(
        self, minimal_problem: ProblemDefinition
    ) -> None:
        targets = minimal_problem.surrogate_columns
        assert "cost_col" in targets

    def test_surrogate_columns_contains_data_constraint_columns(
        self, full_problem: ProblemDefinition
    ) -> None:
        targets = full_problem.surrogate_columns
        assert "pressure_col" in targets

    def test_surrogate_columns_order_objectives_first(
        self, full_problem: ProblemDefinition
    ) -> None:
        targets = full_problem.surrogate_columns
        obj_cols = [o.column for o in full_problem.objectives]
        dc_cols = [c.column for c in full_problem.data_constraints]
        expected = tuple(obj_cols + dc_cols)
        assert targets == expected

    def test_surrogate_columns_deduplicates_shared_column(self) -> None:
        dc = DataConstraint(
            name="cost_cap",
            column="cost_col",
            operator=ConstraintOperator.LE,
            limit=200.0,
        )
        problem = ProblemDefinition(
            variables=(make_decision_var(),),
            objectives=(make_objective("cost", "cost_col"),),
            data_constraints=(dc,),
        )
        assert problem.surrogate_columns == ("cost_col",)

    def test_monotonic_constraints_for_known_target(
        self, full_problem: ProblemDefinition
    ) -> None:
        constraints = full_problem.monotonic_constraints_for("cost_col")
        assert "temperature" in constraints
        assert constraints["temperature"] == MonotonicDirection.INCREASING

    def test_monotonic_constraints_for_unknown_target_returns_empty(
        self, full_problem: ProblemDefinition
    ) -> None:
        constraints = full_problem.monotonic_constraints_for("nonexistent")
        assert constraints == {}

    def test_monotonic_constraints_for_target_with_no_relations(
        self, minimal_problem: ProblemDefinition
    ) -> None:
        constraints = minimal_problem.monotonic_constraints_for("cost_col")
        assert constraints == {}


    def test_hard_linear_constraints(self) -> None:
        hard = LinearConstraint(
            name="hard_c", coefficients={"x": 1.0},
            operator=ConstraintOperator.LE, rhs=10.0,
        )
        soft = LinearConstraint(
            name="soft_c", coefficients={"x": 1.0},
            operator=ConstraintOperator.LE, rhs=50.0,
            severity=ConstraintSeverity.SOFT, penalty_weight=1.0,
        )
        problem = ProblemDefinition(
            variables=(make_decision_var(),),
            objectives=(make_objective("cost", "cost_col"),),
            linear_constraints=(hard, soft),
        )
        assert problem.hard_linear_constraints == (hard,)
        assert problem.soft_linear_constraints == (soft,)

    def test_hard_data_constraints(self) -> None:
        hard = DataConstraint(
            name="hard_dc", column="pressure",
            operator=ConstraintOperator.LE, limit=100.0,
        )
        soft = DataConstraint(
            name="soft_dc", column="cost",
            operator=ConstraintOperator.LE, limit=50000.0,
            severity=ConstraintSeverity.SOFT, penalty_weight=10.0,
        )
        problem = ProblemDefinition(
            variables=(make_decision_var(),),
            objectives=(make_objective("obj", "y"),),
            data_constraints=(hard, soft),
        )
        assert problem.hard_data_constraints == (hard,)
        assert problem.soft_data_constraints == (soft,)

    def test_no_soft_constraints_returns_empty(
        self, minimal_problem: ProblemDefinition
    ) -> None:
        assert minimal_problem.soft_linear_constraints == ()
        assert minimal_problem.soft_data_constraints == ()


class TestProblemDefinitionImmutability:
    def test_is_frozen(self, minimal_problem: ProblemDefinition) -> None:
        with pytest.raises(ValidationError):
            minimal_problem.variables = ()  # type: ignore[misc]
