# ADR-011: Monotonicity Constraints Restricted to Numeric Variables

## Status

Accepted

## Date

2026-02-26

## Context

MonotonicRelation declares that a decision variable has a monotonic relationship with an objective or data constraint. The question is: should categorical and ordinal variables be allowed to participate in monotonic relations?

**Option A — Allow all dtypes, filter in SurrogateManager:**
ProblemDefinition accepts monotonic relations on any decision variable. The SurrogateManager silently ignores constraints on categorical/ordinal variables when mapping to estimator parameters. This keeps the Problem layer permissive and pushes the filtering to the consumer.

**Option B — Reject non-numeric variables in ProblemDefinition:**
MonotonicRelation.decision_variable must reference a continuous or integer variable. Categorical and ordinal variables are rejected at construction time.

### The semantic argument

Monotonicity requires a numeric ordering: "as X increases, Y increases (or stays equal)." Categorical variables have no inherent ordering — "more Schichtmodell" is meaningless. Ordinal variables have an ordering, but it is an ordering of categories, not a numeric axis. Neither XGBoost nor LightGBM supports monotonicity constraints on categorical features (LightGBM raises an error, XGBoost produces semantically undefined behavior).

## Decision

MonotonicRelation.decision_variable must reference a decision variable with dtype continuous or integer. Categorical and ordinal variables are rejected at ProblemDefinition construction time.

## Rationale

- **Fail-fast principle**: A monotonic relation on a categorical variable is a user error — it indicates a misunderstanding of the variable's dtype or the meaning of monotonicity. Rejecting it immediately is more helpful than silently ignoring it downstream.
- **Semantic correctness**: Monotonicity is a property of numeric dimensions. Allowing it on non-numeric variables creates a false sense of domain knowledge being applied.
- **LightGBM hard requirement**: LightGBM raises an error when a monotonicity constraint is set on a categorical feature position. Silent filtering in the SurrogateManager would mask this.
- **Validation belongs in the Problem layer**: The Problem layer validates structural correctness. Whether a monotonic relation makes sense for a given variable type is a structural question, not a training-time concern.

## Consequences

- ProblemDefinition validates that `MonotonicRelation.decision_variable` references a variable with `dtype in (DType.CONTINUOUS, DType.INTEGER)`.
- The error message explicitly states that only continuous and integer variables support monotonicity constraints.
- The SurrogateManager does not need to filter monotonic relations by dtype — it can trust that all relations from the ProblemDefinition are valid for numeric features.
- If ordinal monotonicity support is needed in the future (e.g., via custom ordinal encoding), this decision can be revised by extending the allowed dtypes in ProblemDefinition.
