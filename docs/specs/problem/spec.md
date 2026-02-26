# Problem Layer — Specification

## Purpose

The Problem layer provides a declarative, immutable description of a blackbox optimization problem. It is the central object consumed by all downstream layers (Surrogate, Optimizer, Analysis). It describes WHAT is being optimized — never HOW.

## Enumerations

### DType

Data type of a variable.

| Value | Description |
|-------|-------------|
| `continuous` | Real-valued variable |
| `integer` | Integer-valued variable |
| `categorical` | Unordered discrete variable |
| `ordinal` | Ordered discrete variable |

### Role

Role of a variable in the optimization.

| Value | Description |
|-------|-------------|
| `decision` | Optimizable — the optimizer varies this |
| `context` | Fixed during optimization, but used as a feature in the surrogate |

### Direction

Optimization direction of an objective.

| Value | Description |
|-------|-------------|
| `minimize` | Lower is better |
| `maximize` | Higher is better |

### MonotonicDirection

Direction of a monotonic relationship.

| Value | Description |
|-------|-------------|
| `increasing` | As the variable increases, the target increases (or stays equal) |
| `decreasing` | As the variable increases, the target decreases (or stays equal) |

### ConstraintOperator

Comparison operator for constraints.

| Value | Symbol | Description |
|-------|--------|-------------|
| `le` | ≤ | Less than or equal |
| `ge` | ≥ | Greater than or equal |
| `eq` | == | Equal |

## Domain Entities

### Variable

A named input dimension of the problem.

| Field | Type | Description |
|-------|------|-------------|
| name | str | Unique identifier, maps to a dataset column |
| dtype | DType | Data type |
| role | Role | `decision` or `context` |
| bounds | Bounds | Type-specific bounds, must match dtype |

**Bounds types:**

- `ContinuousBounds`: lower (float), upper (float). lower < upper.
- `IntegerBounds`: lower (int), upper (int). lower < upper.
- `CategoricalBounds`: categories (tuple[str, ...]). At least 2, unique.
- `OrdinalBounds`: categories (tuple[str, ...]). At least 2, unique. Order is defined by tuple position.

**Validation rules:**
- Bounds type must match dtype (continuous variable requires ContinuousBounds, etc.)
- Construction fails immediately on violation (fail-fast)

### Objective

A quantity to be optimized. Each objective requires its own surrogate model.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| name | str | yes | Unique identifier |
| direction | Direction | yes | `minimize` or `maximize` |
| column | str | yes | Reference to a dataset column |
| reference_value | float | no | Reference point for normalization in multi-objective (e.g., current baseline value). Used for hypervolume computation and improvement calculation. |

### LinearConstraint

An analytical linear constraint on decision variables. Directly evaluable, no surrogate needed. Represents constraints of the form: Σ(coefficient_i * variable_i) ≤/≥/== rhs.

| Field | Type | Description |
|-------|------|-------------|
| name | str | Unique identifier |
| coefficients | dict[str, float] | Mapping of decision variable names to their coefficients |
| operator | ConstraintOperator | `le` (≤), `ge` (≥), or `eq` (==) |
| rhs | float | Right-hand side value |

Example: "A + B ≤ 20" becomes `coefficients={"A": 1.0, "B": 1.0}, operator=le, rhs=20.0`.

All three operators (`le`, `ge`, `eq`) are supported. For `eq`, the optimizer translates to two inequality constraints internally.

**Redundancy with variable bounds:** A LinearConstraint like `{"X": 1.0}, le, 100.0` may overlap with a variable's upper bound of 100. This redundancy is accepted silently — the optimizer handles both correctly (bounds restrict the search space, constraints are checked in G).

**Validation rules:**
- coefficients must not be empty
- No coefficient may be zero (a zero coefficient references a variable with no effect — indicates a user error)
- All keys in coefficients must reference existing decision variables (not context variables — a constraint on a fixed variable is meaningless during optimization). Validated at ProblemDefinition level.

### DataConstraint

A data-driven constraint. Requires its own surrogate model (like an objective, but with a threshold instead of an optimization direction).

| Field | Type | Description |
|-------|------|-------------|
| name | str | Unique identifier |
| column | str | Reference to a dataset column |
| operator | ConstraintOperator | `le` (≤), `ge` (≥), or `eq` (==) |
| limit | float | Threshold value |

### MonotonicRelation

Domain knowledge: a known monotonic relationship between a decision variable and an objective or data constraint.

| Field | Type | Description |
|-------|------|-------------|
| decision_variable | str | Reference to a decision variable |
| objective_or_constraint | str | Reference to an objective or data constraint (by name) |
| direction | MonotonicDirection | `increasing` or `decreasing` |

Passed to estimators that support monotonicity constraints during surrogate training. Validated post-hoc by the analysis layer.

**Validation rules:**
- decision_variable must reference an existing decision variable with dtype continuous or integer (categorical and ordinal variables do not support monotonicity constraints — the concept of monotonicity requires a numeric ordering)
- objective_or_constraint must reference an existing objective or data constraint
- No contradictions: the same (decision_variable, objective_or_constraint) pair must not appear with conflicting directions

### Scenario

A named assignment of context variable values. Defines the operating conditions under which optimization is performed.

| Field | Type | Description |
|-------|------|-------------|
| name | str | Unique identifier (e.g., "normal_operation", "peak_load") |
| context_values | dict[str, Any] | Mapping of context variable names to fixed values |

**Validation rules:**
- Must define at least one context variable value
- All keys must reference existing context variables (validated at ProblemDefinition level)
- Values are validated against the referenced variable's type and bounds:
  - Continuous/integer variables: value must be numeric and within declared bounds
  - Integer variables: value must be a whole number
  - Categorical/ordinal variables: value must be one of the declared categories

**Behavior when no scenarios are defined:**
- A utility function `create_default_scenario(BoundDataset)` computes median values for all context variables from the historical data and returns a standard Scenario object. This function lives outside the Problem layer — it is a bridge between BoundDataset (data) and Scenario (structure). The optimizer receives a Scenario, it never computes one.

### ProblemDefinition

The top-level container. Immutable after construction. A single ProblemDefinition can be used with multiple BoundDatasets (e.g., training data and new data for retraining).

| Field | Type | Default |
|-------|------|---------|
| variables | tuple[Variable, ...] | required |
| objectives | tuple[Objective, ...] | required |
| linear_constraints | tuple[LinearConstraint, ...] | () |
| data_constraints | tuple[DataConstraint, ...] | () |
| monotonic_relations | tuple[MonotonicRelation, ...] | () |
| scenarios | tuple[Scenario, ...] | () |

**Cross-field validation (all fail-fast):**
- At least one objective
- At least one decision variable
- All variable names unique
- All objective names unique
- All constraint names unique (across linear + data constraints)
- All scenario names unique
- LinearConstraint.coefficients keys reference existing decision variables (not context variables)
- MonotonicRelation.decision_variable references an existing decision variable with dtype continuous or integer
- MonotonicRelation.objective_or_constraint references an existing objective or data constraint
- No contradictory MonotonicRelations (same decision_variable+objective_or_constraint pair with conflicting directions)
- Scenario context_values keys reference existing context variables
- Scenario context_values values match the referenced variable's type and lie within its bounds

**Derived properties (read-only, computed):**
- `decision_variables` — tuple of variables with role=decision
- `context_variables` — tuple of variables with role=context
- `surrogate_columns` — deduplicated column names from objectives + data constraints (one surrogate per unique column, order: objectives first, then data constraints)
- `monotonic_constraints_for(objective_or_constraint)` — dict of decision_variable→direction for a specific target

### BoundDataset

A pandas DataFrame bound to a ProblemDefinition. Validates that the data matches the problem structure. A single ProblemDefinition can be bound to multiple BoundDatasets — this is explicitly supported for use cases like training data vs. new data for retraining.

| Field | Type | Description |
|-------|------|-------------|
| problem | ProblemDefinition | The problem this dataset belongs to |
| dataframe | pd.DataFrame | The validated dataset |

**Validation on construction (all fail-fast):**
- Every variable has a corresponding column in the DataFrame
- Numeric variables (continuous, integer) have numeric dtype in the DataFrame
- Integer variables contain only integer values
- Numeric values lie within declared bounds
- Categorical/ordinal values are within declared categories (values outside declared categories cause an error)
- No column may contain missing values — neither variable columns nor target columns. The framework receives clean data from kint. Missing values are never silently handled.
- Every objective column exists and has numeric dtype
- Every data constraint column exists and has numeric dtype

**What BoundDataset does NOT do:**
- No imputation of missing values
- No type coercion
- No outlier removal
- No feature engineering

Errors are reported, never silently fixed. Data cleaning is kint's responsibility.

## Consumers

| Layer | What it reads |
|-------|---------------|
| Surrogate | `ProblemDefinition.surrogate_columns` (what to train), all variables (features), `monotonic_constraints_for()` (training constraints), `BoundDataset.dataframe` (training data) |
| Optimizer | `ProblemDefinition.decision_variables` (search space), `linear_constraints` (analytic constraints), `data_constraints` (surrogate-checked constraints), `objectives` (single vs. multi, reference_value for hypervolume), scenarios (context variable values) |
| Analysis | `ProblemDefinition.decision_variables` (for PDP/ICE), `objectives` (for SHAP per objective), `monotonic_relations` (for plausibility checks), scenarios (for comparison) |

## Acceptance Criteria

1. All domain objects are immutable after construction
2. Every validation rule listed above is enforced at construction time
3. Invalid input raises immediately — no silent correction, no fallback
4. ProblemDefinition can be serialized to/from JSON (Pydantic provides this)
5. All derived properties return correct results
6. BoundDataset rejects datasets with missing values in any column
7. BoundDataset rejects datasets with values outside declared bounds/categories
8. Contradictory monotonic relations are rejected at construction
9. A single ProblemDefinition can be bound to multiple BoundDatasets
10. LinearConstraint rejects zero coefficients and references to non-decision variables
11. Scenario values are validated against variable types and bounds
