from pydantic import BaseModel, ConfigDict, model_validator

from surrox.problem.constraints import DataConstraint, LinearConstraint
from surrox.problem.domain_knowledge import MonotonicRelation
from surrox.problem.objectives import Objective
from surrox.problem.scenarios import Scenario
from surrox.problem.types import DType, MonotonicDirection, Role
from surrox.problem.variables import Variable


class ProblemDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    variables: tuple[Variable, ...]
    objectives: tuple[Objective, ...]
    linear_constraints: tuple[LinearConstraint, ...] = ()
    data_constraints: tuple[DataConstraint, ...] = ()
    monotonic_relations: tuple[MonotonicRelation, ...] = ()
    scenarios: tuple[Scenario, ...] = ()

    @model_validator(mode="after")
    def _validate_problem(self) -> "ProblemDefinition":
        self._validate_at_least_one_objective()
        self._validate_at_least_one_decision_variable()
        self._validate_unique_names()
        self._validate_linear_constraint_references()
        self._validate_monotonic_relation_references()
        self._validate_monotonic_relation_no_contradictions()
        self._validate_scenario_references()
        return self

    def _validate_at_least_one_objective(self) -> None:
        if not self.objectives:
            raise ValueError("at least one objective is required")

    def _validate_at_least_one_decision_variable(self) -> None:
        if not any(v.role == Role.DECISION for v in self.variables):
            raise ValueError("at least one decision variable is required")

    def _validate_unique_names(self) -> None:
        variable_names = [v.name for v in self.variables]
        if len(variable_names) != len(set(variable_names)):
            raise ValueError("variable names must be unique")

        objective_names = [o.name for o in self.objectives]
        if len(objective_names) != len(set(objective_names)):
            raise ValueError("objective names must be unique")

        constraint_names = [c.name for c in self.linear_constraints] + [
            c.name for c in self.data_constraints
        ]
        if len(constraint_names) != len(set(constraint_names)):
            raise ValueError("constraint names must be unique")

        scenario_names = [s.name for s in self.scenarios]
        if len(scenario_names) != len(set(scenario_names)):
            raise ValueError("scenario names must be unique")

    def _validate_linear_constraint_references(self) -> None:
        decision_variable_names = {
            v.name for v in self.variables if v.role == Role.DECISION
        }
        for lc in self.linear_constraints:
            for var_name in lc.coefficients:
                if var_name not in decision_variable_names:
                    raise ValueError(
                        f"linear constraint '{lc.name}' references "
                        f"unknown or non-decision variable '{var_name}'"
                    )

    def _validate_monotonic_relation_references(self) -> None:
        numeric_decision_variables = {
            v.name: v
            for v in self.variables
            if v.role == Role.DECISION and v.dtype in (DType.CONTINUOUS, DType.INTEGER)
        }
        all_decision_variable_names = {
            v.name for v in self.variables if v.role == Role.DECISION
        }
        valid_targets = {o.name for o in self.objectives} | {
            c.name for c in self.data_constraints
        }

        for mr in self.monotonic_relations:
            if mr.decision_variable not in all_decision_variable_names:
                raise ValueError(
                    f"monotonic relation references unknown or "
                    f"non-decision variable '{mr.decision_variable}'"
                )
            if mr.decision_variable not in numeric_decision_variables:
                raise ValueError(
                    f"monotonic relation references non-numeric "
                    f"variable '{mr.decision_variable}' — only continuous "
                    f"and integer variables support monotonicity constraints"
                )
            if mr.objective_or_constraint not in valid_targets:
                raise ValueError(
                    f"monotonic relation references unknown target "
                    f"'{mr.objective_or_constraint}'"
                )

    def _validate_monotonic_relation_no_contradictions(self) -> None:
        seen: dict[tuple[str, str], MonotonicDirection] = {}
        for mr in self.monotonic_relations:
            key = (mr.decision_variable, mr.objective_or_constraint)
            if key in seen and seen[key] != mr.direction:
                raise ValueError(
                    f"contradictory monotonic relations for "
                    f"variable '{mr.decision_variable}' and "
                    f"target '{mr.objective_or_constraint}'"
                )
            seen[key] = mr.direction

    def _validate_scenario_references(self) -> None:
        context_variables = {
            v.name: v for v in self.variables if v.role == Role.CONTEXT
        }
        for scenario in self.scenarios:
            for var_name, value in scenario.context_values.items():
                if var_name not in context_variables:
                    raise ValueError(
                        f"scenario '{scenario.name}' references unknown "
                        f"context variable '{var_name}'"
                    )
                self._validate_scenario_value(
                    scenario.name, context_variables[var_name], value
                )

    @staticmethod
    def _validate_scenario_value(
        scenario_name: str, variable: Variable, value: object
    ) -> None:
        if variable.dtype in (DType.CONTINUOUS, DType.INTEGER):
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"scenario '{scenario_name}': variable '{variable.name}' "
                    f"expects numeric value, got {type(value).__name__}"
                )
            if value < variable.bounds.lower or value > variable.bounds.upper:  # type: ignore[union-attr]
                raise ValueError(
                    f"scenario '{scenario_name}': value {value} for "
                    f"variable '{variable.name}' is outside bounds "
                    f"[{variable.bounds.lower}, {variable.bounds.upper}]"  # type: ignore[union-attr]
                )
            if variable.dtype == DType.INTEGER and value != int(value):
                raise ValueError(
                    f"scenario '{scenario_name}': variable '{variable.name}' "
                    f"expects integer value, got {value}"
                )
        elif variable.dtype in (DType.CATEGORICAL, DType.ORDINAL):
            valid = variable.bounds.categories  # type: ignore[union-attr]
            if value not in valid:
                raise ValueError(
                    f"scenario '{scenario_name}': value '{value}' for "
                    f"variable '{variable.name}' is not in "
                    f"categories {valid}"
                )

    @property
    def decision_variables(self) -> tuple[Variable, ...]:
        return tuple(v for v in self.variables if v.role == Role.DECISION)

    @property
    def context_variables(self) -> tuple[Variable, ...]:
        return tuple(v for v in self.variables if v.role == Role.CONTEXT)

    @property
    def surrogate_columns(self) -> tuple[str, ...]:
        seen: set[str] = set()
        columns: list[str] = []
        for column in (
            *(o.column for o in self.objectives),
            *(c.column for c in self.data_constraints),
        ):
            if column not in seen:
                seen.add(column)
                columns.append(column)
        return tuple(columns)

    def monotonic_constraints_for(
        self, objective_or_constraint: str
    ) -> dict[str, MonotonicDirection]:
        return {
            mr.decision_variable: mr.direction
            for mr in self.monotonic_relations
            if mr.objective_or_constraint == objective_or_constraint
        }
