# ADR-013: Explicit Tolerance for Equality Data Constraints

## Status

Accepted

## Date

2026-02-26

## Context

`DataConstraint` supports `ConstraintOperator.EQ` for equality constraints on surrogate-predicted outputs (e.g., "predicted temperature == 95"). The optimizer evaluates data constraints conservatively using conformal prediction intervals instead of point predictions.

For inequality constraints (LE, GE), conservative evaluation is straightforward: use the upper bound for LE, lower bound for GE. For equality, the naive approach is to split into two inequalities with a small epsilon: `|predicted - limit| <= eps`. But which epsilon? A hardcoded value is arbitrary and domain-dependent.

Worse: with conformal intervals, the entire prediction interval must satisfy both inequalities. If the interval is [88, 102] and the constraint is "== 95 +/- 0.5", the interval width (14) far exceeds the tolerance (1.0), making the constraint unsatisfiable — not because the model is wrong, but because the model is uncertain.

### Options

**Option A — Reject EQ on DataConstraints:** Remove EQ from the valid operators for DataConstraint. Users model equality as two explicit LE/GE constraints with their own limits. Simple but forces users to think in terms of inequalities when they mean equality.

**Option B — Explicit tolerance field:** Add `tolerance: float | None` to DataConstraint. Required when `operator=EQ`, forbidden otherwise. The constraint becomes `|predicted - limit| <= tolerance`, mapping to two pymoo G <= 0 constraints. The domain expert decides what "equal" means.

**Option C — Auto-derive tolerance from conformal interval width:** Use the model's uncertainty to set tolerance automatically. "Smart" but unpredictable — the tolerance would change every time the model is retrained, making optimization results non-reproducible.

## Decision

Option B: `DataConstraint` gains a `tolerance: float | None = None` field with validation:
- `operator=EQ` requires `tolerance` (positive float)
- `operator != EQ` requires `tolerance=None`
- `tolerance <= 0` is rejected

The optimizer maps EQ constraints to two pymoo constraints:
- `G1 = lower_bound - (limit + tolerance)`
- `G2 = (limit - tolerance) - upper_bound`

This means the entire conformal interval must fit within `[limit - tolerance, limit + tolerance]`.

## Rationale

- **Explicit over implicit**: The domain expert defines what "equal" means in their context. A chemical process where temperature must be 95 +/- 0.5 degC is different from one where +/- 5 degC is acceptable.
- **Reproducible**: Tolerance is a fixed problem parameter, not derived from model uncertainty. Same problem definition always gives the same constraint semantics.
- **Conservative by design**: With conformal intervals, wide model uncertainty makes EQ hard to satisfy. This is correct behavior for safety-critical applications. The `OptimizationResult` returns infeasible points with diagnostic information so the user can decide to increase tolerance, collect more data, or reduce constraint confidence.

## Consequences

- Breaking change to `DataConstraint` for EQ usage: existing code creating `DataConstraint(operator=EQ)` without `tolerance` will fail validation.
- `LinearConstraint` is unaffected — linear constraints are evaluated analytically (exact), not via surrogates.
- The analysis layer must explain to users why EQ constraints with tight tolerances may produce no feasible solutions.
