# ADR-024: suggest() API for Closed-Loop Optimization

## Status

Accepted

## Date

2026-03-01

## Context

surrox's current API (`surrox.run()`) is a one-shot pipeline: train surrogates, optimize, analyze. This serves offline model-based optimization well, but Kint needs a closed-loop feedback cycle:

1. Suggest next experiments to evaluate
2. User evaluates (runs real experiments/simulations)
3. Add results to dataset
4. Re-suggest based on accumulated data
5. Repeat

This iterative loop is Kint's core moat against standalone LLMs. An LLM can suggest parameters once; Kint orchestrates the entire feedback cycle with principled exploration-exploitation trade-offs and uncertainty quantification.

### Options

**Option A — Session object with incremental updates:** A stateful `Session` class that maintains the surrogate manager across iterations and incrementally updates it with new data. Lower latency per iteration but complex state management, and tree-based models (XGBoost, LightGBM) don't support true incremental training.

**Option B — Stateless `suggest()` function:** Each call trains the surrogate from scratch on the full accumulated dataset. Simple, reproducible, no state to manage. For n < 1000 (typical in experimental optimization), training takes 30–60 seconds — acceptable for a workflow where real experiments take hours to days.

**Option C — Extend `run()` with a "next suggestions" mode:** Overload the existing API. Muddies the distinction between analysis (one-shot) and suggestion (iterative).

## Decision

Option B. A new top-level `surrox.suggest()` function that is stateless and composable.

```python
result = surrox.suggest(
    problem=problem,
    dataframe=accumulated_data,
    n_suggestions=5,
    surrogate_config=config,
    optimizer_config=optimizer_config,
    coverage=0.9,
)
```

Returns a `SuggestionResult` containing:
- `suggestions`: Tuple of `Suggestion` objects, each with variable values, predicted objectives (mean, std, lower, upper bounds), and extrapolation distance.
- `surrogate_quality`: R² per surrogate column, so the caller knows how trustworthy the suggestions are.

Internally, `suggest()` composes existing building blocks: `SurrogateManager.train()` → `suggest_candidates()` → uncertainty enrichment. No new surrogate or optimizer machinery.

The optimizer-level `suggest_candidates()` function extracts diverse top-N candidates from pymoo's full final population (not just the Pareto front) using greedy distance-based diversity selection.

## Rationale

- **Stateless = simple.** The statefulness lives in Kint (accumulated DataFrame), not in surrox. Each `suggest()` call is independent and reproducible.
- **Full re-training per iteration is acceptable.** Real experiments (pharma, chemical, manufacturing) take hours to days. 30–60s of surrogate training is negligible.
- **Composable with LLM integration.** Kint can: (1) use LLM to suggest initial warm-start points before the first `suggest()`, (2) show `Suggestion` objects to an LLM for critique/filtering, (3) let the LLM inject domain constraints into `ProblemDefinition`. surrox remains LLM-agnostic.
- **Uncertainty per suggestion.** Each `Suggestion` includes conformal prediction intervals (`lower`/`upper`) at the configured coverage level. This is the uncertainty quantification that standalone LLMs cannot provide — when Kint says "predicted cost = 42.3 [38.1, 46.5]", that interval has statistical meaning.
- **Diversity selection.** For batch suggestions, the greedy max-min-distance algorithm ensures candidates span the promising region rather than clustering at a single optimum. This balances exploration and exploitation across the batch.

## Consequences

- `surrox.suggest()` is exported as a public API alongside `surrox.run()` and `surrox.run_scenarios()`.
- New types `Suggestion`, `ObjectivePrediction`, and `SuggestionResult` are exported from `surrox`.
- `suggest_candidates()` is added to the optimizer module as an internal function used by `suggest()`.
- Kint's feedback loop becomes: `suggest()` → evaluate → concat → `suggest()` → repeat.
- For problems where re-training is too slow (n > 5000, many columns), a future stateful session API can be added without breaking the stateless API.
