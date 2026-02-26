# ADR-005: Monotonic Relations as Separate Domain Knowledge Entities

## Status

Accepted

## Date

2026-02-26

## Context

Domain experts often know that certain relationships are monotonic: "more budget never decreases quality", "higher temperature always increases energy consumption". This knowledge can improve surrogate model quality (monotonicity constraints in XGBoost/LightGBM) and serve as post-hoc plausibility checks.

Options for modeling:
1. **Attribute on Variable**: Each variable declares its monotonic relationship to objectives. Problem: a variable can have different monotonic relationships to different objectives (e.g., "more staff" increases throughput but also increases cost).
2. **Attribute on Objective**: Each objective declares which variables are monotonically related. Problem: same information split across objectives, harder to maintain.
3. **Separate entity**: MonotonicRelation as an independent object linking a decision variable to an objective or constraint with a direction.

## Decision

MonotonicRelation is a separate entity in ProblemDefinition, not an attribute of Variable or Objective.

## Rationale

- **Many-to-many relationship**: A single variable can have monotonic relationships with multiple objectives (in different directions). A single objective can have monotonic relationships with multiple variables. Neither side owns the relationship.
- **Clean consumption**: `ProblemDefinition.monotonic_constraints_for(objective_or_constraint)` returns only the relations relevant to a specific surrogate. The SurrogateManager calls this once per TrainingJob.
- **Extensibility**: Future domain knowledge types (interaction effects, known non-linearities) can follow the same pattern — separate entities that reference variables and targets.

## Consequences

- `MonotonicRelation.decision_variable` must reference a decision variable (validated by ProblemDefinition)
- `MonotonicRelation.objective_or_constraint` must reference an objective or data constraint (validated by ProblemDefinition)
- Duplicate (decision_variable, objective_or_constraint) pairs with conflicting directions are rejected at construction time (validated by ProblemDefinition)
- The SurrogateManager translates MonotonicRelations into estimator-specific monotonicity constraint parameters
- The analysis layer validates empirical monotonicity of the trained ensemble against declared relations
