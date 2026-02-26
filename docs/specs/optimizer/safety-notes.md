# Optimizer Layer — Safety Requirements

These requirements were identified during the Surrogate layer design and must be addressed in the Optimizer spec.

## 1. Extrapolation Constraint

Points too far from training data must be treated as infeasible — not warned about.

- Distance metric on decision space (Mahalanobis or k-NN distance to nearest training points)
- Configurable threshold beyond which the Optimizer rejects the point
- Restricts search space, but prevents recommendations without empirical basis
- Default: enabled. User can adjust threshold or disable.

## 2. Conservative Constraint Evaluation

DataConstraint feasibility must be checked against the worst-case conformal bound, not the point prediction.

- Example: constraint "pressure ≤ 100 bar", prediction 95 bar, interval [88, 102] → infeasible (102 > 100)
- For ≤ constraints: check upper bound. For ≥ constraints: check lower bound.
- Coverage level for this check should be configurable (default: same as `TrainingConfig.default_coverage`)
- Default: enabled. User can switch to point-prediction evaluation.

## 3. Interaction with Surrogate Quality Gate

The Surrogate layer already enforces `min_r2` — if the surrogate is too weak, training fails before the Optimizer ever runs. The Optimizer can trust that all surrogates it receives passed the quality gate.
