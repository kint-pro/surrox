# ADR-014: Type-Based Algorithm Auto-Selection

## Status

Accepted

## Date

2026-02-26

## Context

pymoo offers multiple optimization algorithms. The correct choice depends on (1) number of objectives and (2) variable types. Users of surrox (via kint) should not need to understand pymoo's algorithm zoo — the framework should select the right algorithm automatically.

The information needed for selection is fully available in `ProblemDefinition`: objective count and declared variable types (continuous, integer, categorical, ordinal). This is deterministic — no heuristics, no data analysis.

### Key constraints

- **DE (Differential Evolution)** operates on continuous variables only. Integer variables require rounding, which pymoo supports via `RoundingRepair` on GA crossover/mutation operators — but not on DE's internal operators.
- **MixedVariableGA** handles categorical/ordinal variables natively with type-specific operators.
- **NSGA-II** works for 2-3 objectives. **NSGA-III** with reference directions is needed for 4+ objectives.
- **Ordinal variables** have order but no defined metric distance. Treating them as integers (0, 1, 2) implies equal spacing, which is often wrong. Safer to treat as categorical.

## Decision

Algorithm selection follows a 3x3 decision matrix based on declared variable types:

| Objectives | All continuous | Has integer (no categorical/ordinal) | Has categorical or ordinal |
|---|---|---|---|
| 1 | DE | GA + RoundingRepair | MixedVariableGA |
| 2-3 | NSGA-II | NSGA-II + RoundingRepair | MixedVariableGA + RankAndCrowdingSurvival |
| 4+ | NSGA-III | NSGA-III + RoundingRepair | MixedVariableGA + ReferenceDirectionSurvival |

- Ordinal is classified with categorical (no assumed equal spacing — a custom OrdinalRepair that exploits ordering without assuming equal distance was considered but rejected as premature: ordinal variables typically have few levels (3-5), and MixedVariableGA's categorical operators explore them efficiently)
- Only decision variables are considered (context variables are injected, not optimized)
- All algorithms seeded from `OptimizerConfig.seed`
- Reference directions: `get_reference_directions("das-dennis", n_objectives, n_partitions=max(4, 16-2*n_objectives))` — partitions scale down with objectives to keep reference point count manageable
- **Population size minimum**: `OptimizerConfig.population_size` is the user's requested size. For NSGA-III, the selector enforces `pop_size = max(user_pop_size, n_reference_directions)` since NSGA-III requires population ≥ reference directions. The selector adjusts upward silently, never reduces.

## Rationale

- **Deterministic**: No runtime heuristics. Same problem definition always selects the same algorithm.
- **Correct by construction**: DE is only used when all variables are continuous. Integer rounding happens at the operator level, not as a post-hoc hack. Categorical variables get proper type-specific operators.
- **Extensible**: Adding a new algorithm (e.g., MOEA/D) means adding a row or column to the matrix, not modifying existing selection logic.

## Consequences

- Users cannot override algorithm selection. If this becomes a requirement, `OptimizerConfig` can gain an optional `algorithm_override` field.
- The 4+ objectives case with categorical variables uses MixedVariableGA with `ReferenceDirectionSurvival` from NSGA-III. This gives NSGA-III's reference-direction-based selection with MixedVariableGA's type-aware operators.
- Performance implications: GA is slower than DE for continuous-only problems. The matrix ensures DE is used whenever possible.
