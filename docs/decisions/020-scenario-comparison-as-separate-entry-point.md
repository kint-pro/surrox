# ADR-020: Scenario Comparison as Separate Entry Point

## Status

Accepted

## Date

2026-02-27

## Context

The Analysis layer provides single-run analyses (SHAP, PDP/ICE, What-If, etc.) that operate on one `OptimizationResult`. Scenario comparison is fundamentally different — it compares results across multiple optimization runs (one per scenario).

In the original spec, `scenario_comparison(results)` was a method on `AnalysisResult`, taking a `dict[str, OptimizationResult]` as argument. This is inconsistent: all other methods operate on the single result that was passed to `analyze()`, while `scenario_comparison` requires external inputs.

### Options

**Option A — Method on AnalysisResult/Analyzer:** Single entry point, but inconsistent input semantics. The method takes different inputs than all other methods.

**Option B — Separate top-level function:** `compare_scenarios(results, problem) -> ScenarioComparisonResult`. Clear separation between single-run and cross-run analysis. Consistent with how the optimization literature treats scenario analysis — as a separate phase, not a sub-analysis of a single run.

## Decision

Option B. `compare_scenarios()` is a standalone function in `src/surrox/analysis/scenario.py`, exported from `src/surrox/analysis/__init__.py`.

```python
def compare_scenarios(
    results: dict[str, OptimizationResult],
    problem: ProblemDefinition,
) -> ScenarioComparisonResult
```

The function takes `OptimizationResult` (not `AnalysisResult`) because it needs access to the actual solution points (`feasible_points`, `compromise_index`) to extract recommended variable values per scenario.

## Rationale

- Single-run analysis and cross-run comparison have different inputs, different semantics, and different consumers. Separating them makes the API honest about what each function needs.
- `compare_scenarios()` operates on `OptimizationResult` objects because it needs the solution points directly. `AnalysisResult` only contains the Summary (no points), so it would be insufficient.
- Adding future cross-run analyses (e.g., Pareto front comparison, robustness metrics) follows the same pattern — top-level functions, not methods on a single-run result.

## Consequences

- Two entry points instead of one: `analyze()` for single-run, `compare_scenarios()` for cross-run.
- kint calls `compare_scenarios()` with the `OptimizationResult` objects from each scenario run. It does not need to call `analyze()` first — the two entry points are independent.
