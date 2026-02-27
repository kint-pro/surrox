# ADR-016: Conservative Constraint Evaluation via Conformal Prediction Intervals

## Status

Accepted

## Date

2026-02-26

## Context

Data constraints are evaluated by surrogate models, not exact formulas. A point prediction of "pressure = 99.5" when the constraint is "pressure <= 100" looks feasible, but if the model is uncertain (prediction interval [88, 112]), the true value might violate the constraint.

For safety-critical applications (TenneT grid operations, industrial processes), constraint violations can have serious real-world consequences. The optimizer must account for prediction uncertainty when evaluating constraints.

### Approach

Use conformal prediction intervals (from Layer 2) to evaluate constraints conservatively:

- **LE (<=)**: Use the **upper bound** of the prediction interval. `G = upper_bound - limit`. If even the worst-case prediction satisfies the constraint, the true value almost certainly does.
- **GE (>=)**: Use the **lower bound**. `G = limit - lower_bound`.
- **EQ (== with tolerance)**: The entire interval must lie within the tolerance band. `G1 = lower_bound - (limit + tolerance)`, `G2 = (limit - tolerance) - upper_bound`.

The confidence level is configurable via `OptimizerConfig.constraint_confidence` (default 0.95 = 95% coverage).

## Decision

All data constraint evaluations in the optimizer use conformal prediction intervals, not point predictions. The coverage level is a user-configurable parameter, not hardcoded.

## Rationale

- **Safety-first default**: The default of 0.95 means a 95% probability that the true value falls within the interval (distribution-free guarantee from conformal prediction). This is deliberately conservative — a framework claiming safety-critical capability should not ship with a permissive default. Users who want exploratory optimization explicitly lower it.
- **Tunable conservatism**: `constraint_confidence=0.99` for the most critical applications, `0.8` for exploratory optimization. The user decides the risk tolerance.
- **EQ is intentionally strict**: The entire interval must fit within the tolerance band. This is often hard to satisfy with uncertain models. This is correct — it surfaces that the model is not precise enough for equality constraints. The result includes diagnostic information (interval width per constraint) so the user can understand why.

## Consequences

- Conservative evaluation shrinks the feasible region. Some truly feasible solutions may be rejected. This is the intentional tradeoff: fewer but more reliable solutions.
- With high confidence levels (0.99) and uncertain models, it is possible that no feasible solutions are found. The `OptimizationResult` handles this gracefully (see ADR-017).
- Linear constraints are unaffected — they are evaluated analytically (exact values, no uncertainty).
- **Interaction with Extrapolation Gate (ADR-015)**: Both safety mechanisms operate simultaneously and their effects compound. Near the boundary of the training data, the surrogate model is more uncertain, producing wider conformal intervals. Wider intervals make conservative constraint evaluation stricter, shrinking the effective feasible region more than either mechanism alone. This is emergent but correct behavior: regions where the model is uncertain should be treated cautiously for both extrapolation and constraint satisfaction. The practical effect is that the feasible region contracts toward the interior of the well-observed training data — exactly where surrogate predictions are most reliable. Users observing unexpectedly few feasible solutions near training data boundaries should consider this multiplicative interaction.
