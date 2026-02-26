# ADR-004: Scenarios as First-Class Entities

## Status

Accepted

## Date

2026-02-26

## Context

Industrial optimization problems involve context variables — factors that influence outcomes but cannot be controlled (e.g., wind speed, market price, equipment degradation). During optimization, these must be fixed to specific values.

Options:
1. **No scenario concept**: Users pass context values as raw dicts to the optimizer. No validation, no naming, no cross-scenario comparison.
2. **Scenarios as metadata**: Named context assignments stored outside the ProblemDefinition. Loose coupling, but no validation against the variable definitions.
3. **Scenarios as first-class entities in ProblemDefinition**: Named, validated, integrated into the problem structure.

Reference:
- Ax, BoTorch, pymoo: No explicit scenario concept.
- SMAC3: Instance features (similar idea — each problem instance has fixed feature values).

## Decision

Scenarios are first-class entities within ProblemDefinition. Each scenario is a named mapping of context variable names to values, validated against the declared context variables.

## Rationale

- **Industrial requirement**: Customers like TenneT need to compare optimization results across operating conditions ("normal operation" vs. "peak load" vs. "equipment failure"). This is not an edge case — it is a core use case.
- **Validation**: A scenario referencing a non-existent context variable is a bug. Catching it at ProblemDefinition construction time (fail-fast) prevents silent errors downstream.
- **Cross-scenario analysis**: The analysis layer can compare OptimizationResults across scenarios because each result carries its scenario reference. This enables questions like "which decisions are robust across all scenarios?"
- **Optimizer contract**: Each optimization run is bound to exactly one scenario. The evaluation function prepends the fixed context values to every candidate. This is clean and explicit.

## Consequences

- ProblemDefinition validates scenario keys against context variable names
- Each OptimizationResult carries a reference to the scenario it was optimized under
- When no scenarios are defined, a utility function computes a default scenario from the BoundDataset (median of each context variable's historical values) and returns a standard Scenario object. The optimizer always receives a Scenario — it never computes one
- Future: robust optimization can evaluate candidates across multiple scenarios (not phase 1, but the design supports it)
