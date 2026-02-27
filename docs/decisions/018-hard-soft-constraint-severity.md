# ADR-018: Hard vs. Soft Constraint Severity

## Status

Accepted

## Date

2026-02-27

## Context

All constraints (LinearConstraint, DataConstraint) are treated uniformly as hard constraints — mapped to pymoo's `G ≤ 0` convention. A point violating any constraint is infeasible, regardless of whether the constraint represents a safety-critical limit ("pressure ≤ 100 bar") or a preference ("cost ≤ 50k").

Industrial optimization requires distinguishing between:
- **Hard constraints**: Must be satisfied. Violations make a solution infeasible and unusable (e.g., safety limits, physical boundaries, regulatory requirements).
- **Soft constraints**: Should be satisfied. Violations are undesirable but acceptable — the optimizer steers away from them via penalty, but solutions violating soft constraints remain feasible.

### Options

**Option A — All constraints are hard (status quo):** Simple, but forces users to model preferences as objectives or accept binary feasible/infeasible classification. A "cost ≤ 50k" constraint makes all solutions above 50k completely infeasible, even if 51k would be acceptable.

**Option B — Per-constraint severity with penalty:** Each constraint carries a `severity` field (hard/soft). Hard constraints use pymoo's native constraint handling (G ≤ 0, Feasibility First). Soft constraints add a weighted penalty to all objectives, steering the optimizer away from violations without rejecting solutions outright.

**Option C — Global constraint handling strategy:** Use pymoo's `ConstraintsAsPenalty` or `ConstraintsAsObjective` wrappers. These apply to all constraints uniformly — no per-constraint control.

## Decision

Option B. Each constraint has `severity: ConstraintSeverity` (default HARD) and an optional `penalty_weight: float` (required when SOFT).

### Penalty mechanism

Soft constraint penalty follows pymoo's own `ConstraintsAsPenalty` implementation (verified in pymoo source, `constraints/as_penalty.py` line 36):

```
objectives += penalty_weight × max(0, violation)
```

Applied to **all** objectives equally. This is the standard approach in multi-objective evolutionary optimization: a uniform penalty degrades the point in all objective dimensions, causing it to be dominated by non-violating points in Pareto selection.

No automatic scaling — the user sets `penalty_weight` to match their objective scale. This is transparent and predictable, unlike hidden normalization heuristics.

### Validation rules

- `severity` defaults to `HARD` (safety-first)
- `penalty_weight` is required when `severity=SOFT` (same pattern as `tolerance` for EQ operator)
- `penalty_weight` must be `None` when `severity=HARD`
- `penalty_weight` must be positive

### Optimizer behavior

- `_count_constraints()` counts only HARD constraints for pymoo's `n_ieq_constr`
- Hard constraints produce G values (Feasibility First selection)
- Soft constraints produce penalty values added to objectives (no G values)
- Both use conformal prediction intervals for conservative evaluation (uncertainty is orthogonal to severity)
- Feasibility is determined only by hard constraints

## Rationale

- **Safety-first default**: `severity=HARD` ensures existing behavior is preserved and new users don't accidentally create soft constraints.
- **Per-constraint granularity**: pymoo's `ConstraintsAsPenalty` is all-or-nothing — it converts all constraints to penalties. Industrial problems need mixed strategies (some hard, some soft).
- **No automatic penalty scaling**: Any automatic normalization (× objective range, etc.) is a hidden heuristic that can produce surprising behavior. The user knows their objective and constraint scales. `penalty_weight=10.0` means "a violation of 1.0 adds 10.0 to each objective" — predictable.
- **Required penalty_weight**: Forces the user to think about the scale when creating a soft constraint, preventing accidental meaningless defaults.

## Consequences

- The penalty weight choice requires domain knowledge. Users must understand the scale of their objectives to set meaningful weights. This is inherent to penalty methods and not something we should hide.
- Soft constraints still appear in `ConstraintEvaluation` diagnostics with full violation, prediction, and conformal interval data. The Analysis layer can report which soft constraints are violated and by how much.
- A point violating only soft constraints is classified as feasible in `OptimizationResult.feasible_points`. Its objectives are penalized (worse values) but it is not rejected.
- Hard and soft constraints can coexist on the same problem. The optimizer handles them through separate mechanisms (G values vs. penalty addition).
