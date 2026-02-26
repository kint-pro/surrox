# ADR-007: Deduplicated Surrogate Columns

## Status

Accepted

## Date

2026-02-26

## Context

An Objective and a DataConstraint can reference the same DataFrame column. Example: "minimize emissions" (Objective) and "emissions <= 500t" (DataConstraint) both point to the `emissions` column.

The SurrogateManager trains one model per column. If `surrogate_columns` returned duplicates, the manager would train two identical models for the same column — wasting compute and introducing inconsistency (two models with different predictions for the same quantity).

## Decision

`ProblemDefinition.surrogate_columns` returns **deduplicated** column names. Objectives are listed first, followed by data constraints. If a column appears in both, it appears only once (at the position of the first occurrence).

The SurrogateManager trains exactly one surrogate per unique column. Both the Objective and the DataConstraint consume the same surrogate's predictions.

## Consequences

- One surrogate per unique column — no redundant training, no prediction inconsistency
- The Optimizer must resolve which surrogate belongs to which Objective/DataConstraint by column name, not by position in `surrogate_columns`
- The Analysis layer computes explanations (Feature Importance, SHAP, PDP) per surrogate, not per consumer. When an Objective and a DataConstraint share a column, queries for either return identical results because the same model underlies both. The Analysis layer must resolve by column name to avoid redundant computation.
- The property was renamed from `surrogate_targets` to `surrogate_columns` to reflect that it returns column names, not target objects
