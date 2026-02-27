# Concepts

## Four-Layer Architecture

surrox processes optimization problems through four sequential layers:

1. **Problem** — Declarative, immutable problem definition. All validation happens at construction time.
2. **Surrogate** — Trains one ensemble per unique target column using Optuna HPO over XGBoost and LightGBM. Includes conformal prediction for uncertainty quantification.
3. **Optimizer** — pymoo-based multi-objective optimization on the trained surrogates. Auto-selects the algorithm (DE, GA, NSGA-II, NSGA-III) based on the problem structure.
4. **Analysis** — Summary analysis runs automatically. Detail analyses (SHAP, PDP/ICE, What-If) are lazy and cached via the Analyzer.

Each layer consumes the output of the previous layer plus the Problem object.

## Variables

Variables are typed (`continuous`, `integer`, `categorical`, `ordinal`) and have a role:

- **Decision variables** — the optimizer searches over these
- **Context variables** — fixed during optimization, varied across scenarios

## Surrogates and Ensembles

For each unique target column (objectives + data constraints), surrox trains an ensemble of gradient boosting models:

- **Optuna HPO** searches over XGBoost and LightGBM hyperparameters
- **Ensemble selection** picks diverse, high-performing models weighted by softmax of CV performance
- **Conformal Prediction** provides distribution-free prediction intervals with guaranteed coverage

## Constraints

Two types of constraints control the feasibility of solutions:

- **LinearConstraint** — analytical constraint on decision variables (`sum(coeff * x) <= rhs`), evaluated exactly
- **DataConstraint** — surrogate-based constraint on a predicted column (`prediction <= limit`), evaluated via the trained surrogate with uncertainty

Both support `hard` (must satisfy) and `soft` (penalized) severity levels.

## Monotonic Relations

Domain knowledge about monotonic relationships (e.g., "increasing temperature always increases yield") can be encoded as [`MonotonicRelation`][surrox.MonotonicRelation] objects. These constrain the surrogate training to respect known physical relationships.

## Scenarios

Scenarios fix context variables to specific values, enabling what-if comparisons across different operating conditions (e.g., summer vs. winter). Surrogates are trained once; optimization runs independently per scenario.

## Extrapolation Detection

The optimizer flags candidate solutions that lie outside the training data manifold using k-nearest-neighbor distance. Points beyond the threshold are marked as extrapolating, providing a trust signal for recommendations.
