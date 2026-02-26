# ADR-008: Joint Optuna Study Across Estimator Families

## Status

Accepted

## Date

2026-02-26

## Context

The SurrogateManager trains surrogates using two estimator families: XGBoost and LightGBM. The question is whether to run one Optuna study per family (two studies per surrogate column) or one joint study that explores both families.

**Option A — Separate studies per family:**
Each family gets its own study with its own trial budget (e.g., 25 trials each). The best model from each study is compared afterward. Simple, no cross-family interaction during search.

**Option B — Joint study:**
The estimator family is a categorical hyperparameter within a single study. Optuna's TPE sampler explores both families and their respective hyperparameter spaces jointly. The trial budget is shared (e.g., 50 trials total).

## Decision

One joint Optuna study per surrogate column, with `estimator_family` as a `suggest_categorical` parameter.

## Rationale

- **Better budget allocation**: Optuna's TPE sampler naturally allocates more trials to the family that performs better. If XGBoost dominates for a given target, Optuna spends more trials exploring XGBoost configurations. Separate studies would split the budget rigidly 50/50.
- **Cross-family diversity for ensembles**: The top-K trials from a joint study naturally include both families when both are competitive. This produces structurally diverse ensembles (depth-wise vs. leaf-wise growth) without explicit cross-family balancing logic.
- **Simpler orchestration**: One study per column instead of two. One trial history, one leaderboard, one sorted list to pick top-K from.
- **Pruning works across families**: A poor LightGBM trial is pruned based on all completed trials including XGBoost trials, and vice versa. This raises the pruning bar — only genuinely competitive configs survive.

## Consequences

- Each trial's hyperparameter space is conditional on the `estimator_family` value. Each family defines its own search space via the `EstimatorFamily` protocol (ADR-012). Optuna handles conditional spaces natively.
- The TPE sampler's `multivariate=True` setting models correlations across the joint space, including the family dimension.
- Trial records store the estimator family name as a string, enabling per-family filtering for leaderboard display.
- Adding a new estimator family to the joint study requires only adding it to `TrainingConfig.estimator_families` — no pipeline changes.
