# ADR-019: Frozen AnalysisResult + Stateful Analyzer

## Status

Accepted

## Date

2026-02-27

## Context

The Analysis layer computes a Summary (cheap, always) and Detail Analyses (expensive, on-demand). Detail analyses like SHAP and PDP/ICE should be lazy and cached — computed on first access, then reused.

All other layers in surrox return frozen Pydantic models (ADR-002). The natural approach — putting lazy cached methods on the result model — conflicts with this:

1. **`cached_property` on frozen Pydantic models** has known bugs: `model_copy()` copies the cached value to the new instance, breaking independence. `__eq__` fails on models with `cached_property` — equal instances compare unequal.

2. **`functools.lru_cache` on methods** requires the instance to be hashable. Frozen Pydantic models are hashable, but `lru_cache` on `self` keeps the instance alive forever (memory leak for long-lived caches).

3. **Making `AnalysisResult` non-frozen** breaks the architectural invariant that all layer outputs are immutable.

### Options

**Option A — Frozen Result + separate Analyzer class:** `analyze()` returns a frozen `AnalysisResult` (Summary only) and an `Analyzer` (plain Python class with `dict`-based cache for detail analyses). The `Analyzer` is stateful by design — it holds references to the inputs (SurrogateManager, OptimizationResult, etc.) and computes detail analyses on demand.

**Option B — Non-frozen Pydantic model with `cached_property`:** Simple API, but breaks the frozen pattern and triggers Pydantic bugs.

**Option C — Plain class (no Pydantic) for the entire result:** Full control over caching, but loses schema validation and serialization for the Summary.

## Decision

Option A. `analyze()` returns `tuple[AnalysisResult, Analyzer]`.

- `AnalysisResult` is a frozen Pydantic model containing only the `Summary`.
- `Analyzer` is a plain Python class that holds the inputs and an internal `dict` cache. Its methods (`shap_global()`, `pdp_ice()`, `what_if()`, etc.) compute results lazily and cache them.

## Rationale

- Preserves the frozen-output invariant for all layer results (ADR-002 consistency).
- Avoids all known Pydantic caching bugs by not mixing caching into the model.
- The `Analyzer` is explicitly stateful — no hidden mutation on a supposedly-immutable object.
- kint calls `analyzer.shap_global("column")` instead of `result.shap_global("column")` — minimal API difference.
- The `Summary` (always needed for dashboards) is cleanly separated from expensive detail analyses (only needed on user interaction).

## Consequences

- Consumers receive two objects instead of one. The `AnalysisResult` is for immediate display (dashboard), the `Analyzer` is for interactive exploration.
- The `Analyzer` is not serializable — it holds references to trained models and large datasets. This is intentional: detail analyses are computed in-process, not transferred over the wire. kint serializes the individual result objects (e.g., `ShapGlobalResult`) after computation.
- The `Analyzer` cache is not bounded. For the expected usage pattern (a handful of columns, one optimization run), this is acceptable. If memory becomes a concern, an LRU eviction policy can be added later.
- The `Analyzer` is not thread-safe — its internal `dict` cache can corrupt under concurrent writes. This is by design: in a multi-tenant deployment, each user/session gets its own `Analyzer` instance. Sharing an `Analyzer` between concurrent requests is not supported and not necessary.
- `what_if()` is not cached — each call has different variable values and the computation is cheap (single surrogate evaluation). All other detail analysis methods are cached.
