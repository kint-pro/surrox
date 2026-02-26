# ADR-009: Ensemble Construction with Diversity Filter and Softmax Weights

## Status

Accepted

## Date

2026-02-26

## Context

After an Optuna study completes, the top-K models are combined into an ensemble. Two decisions are involved: how to select which models to include, and how to weight them.

### Selection

**Option A — Pure top-K by validation score:**
Take the K best models by CV RMSE. Simple, but risks a degenerate ensemble where multiple models have near-identical predictions (correlation > 0.99), providing no diversity benefit.

**Option B — Greedy diversity filter:**
Iterate sorted candidates. Add a model only if its out-of-fold prediction correlation with every already-selected member is below a threshold τ. This guarantees ensemble members are structurally different.

### Weighting

**Option A — Inverse RMSE:**
`w_i = (1/RMSE_i) / sum(1/RMSE_j)`. Simple but numerically unstable when a dominant model has near-zero RMSE.

**Option B — Softmax of negative RMSE:**
`weights = softmax(-RMSE / temperature)`. All weights strictly positive, sum to 1, numerically stable.

**Option C — Uniform weights:**
Equal weight for all members. Robust when models are similar quality, but ignores genuine quality differences.

## Decision

Greedy diversity filter (τ = 0.95 default) for selection. Softmax of negative RMSE (temperature = 1.0 default) for weighting.

## Rationale

### Diversity filter
- Two models with Pearson correlation > 0.99 on OOF predictions are effectively duplicates — the second adds no error reduction. The filter prevents this.
- Greedy selection preserves the best model first, then adds diversity. This is a conservative strategy that never sacrifices the best model for diversity.
- Cross-family inclusion (XGBoost + LightGBM) happens naturally when both families are competitive — their structural differences produce lower correlation.

### Softmax weights
- Numerically stable — no division by near-zero values.
- Temperature provides a single knob: higher temperature → more uniform, lower → winner-takes-more.
- Both τ and temperature are configurable in TrainingConfig.

### Known limitation
Softmax on raw RMSE is scale-dependent. A 1% relative RMSE difference produces different weight distributions depending on the absolute scale. A future improvement would normalize RMSE values before applying softmax (e.g., by range or by applying softmax to ranks).

## Consequences

- OOF predictions must be stored in memory during the Optuna study for correlation computation. Memory cost: one float array per completed trial × train_pool size.
- If all candidates correlate above τ with the best model, the ensemble degrades to a single model. This is correct behavior, not an error.
- The diversity threshold τ and softmax temperature are exposed in TrainingConfig for tuning.
