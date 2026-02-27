# ADR-017: Optimization Result Always Returned, Never Error on Infeasibility

## Status

Accepted

## Date

2026-02-26

## Context

The optimizer may find no feasible solutions due to: tight constraints, conservative conformal evaluation, extrapolation penalties, insufficient population/generations, or genuinely infeasible constraint combinations. The question is whether this is an error or a valid result.

### Options

**Option A — Raise OptimizationError:** The optimizer throws an exception when no feasible solution is found. Simple but loses all computed information. The caller cannot inspect why or use the best infeasible solutions.

**Option B — Return result with diagnostic information:** The result always contains everything the optimizer computed: feasible points (if any), infeasible points (sorted by violation), and per-constraint diagnostics. A `has_feasible_solutions` flag lets the caller decide.

## Decision

Option B. `OptimizationResult` always returns, never raises on infeasibility.

Result structure:
- `feasible_points: tuple[EvaluatedPoint, ...]` — Pareto front (multi-objective) or best solution (single)
- `infeasible_points: tuple[EvaluatedPoint, ...]` — sorted by total constraint violation ascending (least-violated first)
- `has_feasible_solutions: bool` — convenience flag
- Each `EvaluatedPoint` contains `constraints: tuple[ConstraintEvaluation, ...]` with violation value, point prediction, and conformal interval bounds per constraint
- Each `EvaluatedPoint` contains `extrapolation_distance` and `is_extrapolating` for extrapolation diagnostics

For multi-objective problems with feasible solutions:
- `compromise_index: int | None` — index of the compromise solution (closest to utopia in normalized objective space)
- `hypervolume: float | None` — hypervolume indicator of the feasible Pareto front (only when >= 2 feasible points). Reference point = worst per-objective across feasible points × 1.1.

## Rationale

- **No information loss**: The optimizer invested computation; the result should preserve it. Best infeasible solutions are often valuable — they show what is close to feasible and which constraints are the bottleneck.
- **Caller decides**: kint as a platform knows the application context. For TenneT: "no feasible solution" is a critical finding that needs user action (loosen constraints, increase data, reduce confidence). For less critical applications: "best infeasible candidates" may be useful with caveats.
- **Diagnostic depth**: Per-constraint violation + conformal bounds enable the analysis layer to explain why solutions are infeasible — is it the constraint limit, the model uncertainty, or the conservative evaluation?

## Consequences

- The caller must check `has_feasible_solutions` before using `feasible_points`. This is explicit and intentional.
- `compromise_index` and `hypervolume` are `None` in degenerate cases (single-objective, no feasible points, < 2 Pareto points). Callers must handle `None`.
- Hypervolume reference point uses worst values across feasible points only (× 1.1 offset). Using all evaluated points (including infeasible) would be unstable: infeasible outliers with penalized objectives shift the reference point, making hypervolume incomparable across runs.
