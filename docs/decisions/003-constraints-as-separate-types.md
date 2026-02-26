# ADR-003: Constraints as Two Separate Types (LinearConstraint + DataConstraint)

## Status

Accepted (revised 2026-02-26)

## Date

2026-02-26

## Context

Optimization problems have constraints. Two fundamentally different kinds exist:

1. **Analytical constraints on variables**: "A + B ≤ 20" — can be evaluated directly from decision variable values, no model needed.
2. **Data-driven constraints**: "predicted CO₂ emission ≤ limit" — the constraint value must be predicted by a surrogate model, similar to an objective.

### Sub-decision: How to represent analytical constraints

The original design used `VariableConstraint` with a free-form `expression: str` field. This was rejected because:

- Evaluating string expressions requires a parser or unsafe code execution — unnecessary complexity for phase 1
- Security risk if expressions come from user input
- Not type-safe — validation of the expression is deferred to runtime
- pymoo cannot consume string expressions directly

Instead, analytical constraints are modeled as `LinearConstraint` with structured fields: `coefficients: dict[str, float]`, `operator: ConstraintOperator`, `rhs: float`. This covers the most common case (linear constraints) and is directly consumable by pymoo. Non-linear analytical constraints can be added as additional types later.

## Decision

Two separate types: `LinearConstraint` (analytical, structured) and `DataConstraint` (surrogate-based).

## Rationale

- **Different semantics**: An objective is optimized (minimize/maximize). A data constraint is satisfied (value ≤/≥/== limit). These are different concepts that happen to both need surrogate models.
- **Different consumption**: The optimizer treats them differently — objectives go into the fitness function (F), constraints go into the constraint violation function (G). Conflating them forces the optimizer to inspect a flag on every evaluation.
- **Different analysis**: Constraint status (active/slack/violated) is meaningful. Objective "status" is not.
- **LinearConstraint needs no surrogate**: It is analytically evaluable. The SurrogateManager only looks at DataConstraints.
- **Type safety**: `coefficients: dict[str, float]` is validated at construction. No parser, no security risk.
- **pymoo compatibility**: Linear constraints map directly to pymoo's constraint function interface.

### Edge case: Redundancy with variable bounds

A simple bound constraint ("X ≤ 100") is a special case of a LinearConstraint: `coefficients={"X": 1.0}, operator=le, rhs=100.0`. This overlaps with the variable's own bounds (`upper=100`). The framework accepts this redundancy silently — the optimizer handles both correctly (bounds restrict the search space, constraints are checked in G). Detecting and warning about redundancy adds complexity without value. The behavior is documented, not prevented.

### Equality constraints in LinearConstraint

LinearConstraint supports all three operators: `le`, `ge`, and `eq`. Both LinearConstraint and DataConstraint share the same `ConstraintOperator` enum — maintaining two near-identical enums would be unnecessary duplication.

For equality constraints ("A + B == 20"), the optimizer is responsible for translating `eq` into two inequality constraints (≤ and ≥) for pymoo. This is an optimizer-internal concern — the Problem layer describes the semantic intent, the optimizer layer handles the mechanical translation.

## Consequences

- The SurrogateManager trains surrogates for `objectives + data_constraints`, not for `linear_constraints`
- `ProblemDefinition.surrogate_targets` returns columns from objectives and data constraints only
- The optimizer evaluates LinearConstraints directly (sum of coefficient * value vs. rhs) and DataConstraints via surrogate
- The optimizer translates `eq` LinearConstraints into two pymoo inequality constraints
- Constraint names must be unique across both types (validated by ProblemDefinition)
- Redundancy between LinearConstraints and variable bounds is accepted silently
- If non-linear analytical constraints are needed later, a new type (e.g., `NonlinearConstraint`) is added — no changes to existing types
