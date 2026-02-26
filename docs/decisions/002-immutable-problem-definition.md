# ADR-002: Immutable Problem Definition as Central Object

## Status

Accepted

## Date

2026-02-26

## Context

The framework has four layers (Problem, Surrogate, Optimizer, Analysis) that all need access to the problem description. The question is whether the problem definition should be mutable (layers can modify it) or immutable (layers consume it read-only).

Reference frameworks:
- **Ax (Meta)**: Experiment is mutable — trials are added over time. But SearchSpace and OptimizationConfig are effectively set once.
- **SMAC3**: Scenario is immutable after construction.
- **pymoo**: Problem is mutable (subclass with overridden methods).

## Decision

ProblemDefinition is frozen (immutable) after construction. All collections use `tuple` instead of `list`. All contained objects (Variable, Objective, etc.) are also frozen.

## Rationale

- **Predictability**: If the problem definition can change mid-pipeline, every layer must defensively copy or re-validate. Immutability eliminates this class of bugs.
- **Shareability**: The same ProblemDefinition can be passed to multiple SurrogateManagers or Optimizers without risk of interference.
- **Debuggability**: When something goes wrong, the problem definition at the time of the error is exactly the problem definition at the time of creation. No "it was modified somewhere" hunting.
- **Serialization**: An immutable object can be serialized once and the serialized form is always valid. Mutable objects require re-validation after deserialization.

## Consequences

- To change a problem definition, a new one must be created (Pydantic's `model_copy(update={...})` makes this ergonomic)
- Derived properties (`decision_variables`, `surrogate_targets`) may be computed eagerly at construction or lazily on first access — both are safe because the object is immutable. Caching is permitted and recommended for frequently accessed properties
- All downstream layers receive the ProblemDefinition by reference, never modify it
