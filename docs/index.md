# surrox

**Blackbox surrogate-based optimization framework.**

surrox takes a declarative problem description, trains surrogate models on historical data, optimizes on the surrogates, and delivers explained, uncertainty-assessed recommendations.

## Features

- **Declarative problem definition** — variables, objectives, constraints, and domain knowledge as immutable Pydantic models
- **Automated surrogate training** — XGBoost/LightGBM ensembles with Optuna HPO and conformal prediction intervals
- **Multi-objective optimization** — pymoo-based, auto-selects algorithm from problem structure
- **Built-in explainability** — SHAP, PDP/ICE, feature importance, trade-off analysis, what-if predictions
- **Scenario comparison** — run multiple scenarios and compare results
- **Persistence** — save and load results and trained surrogates

## Installation

```bash
pip install surrox[all]
```

## Quick Example

```python
import pandas as pd
import surrox

problem = surrox.ProblemDefinition(
    variables=(
        surrox.Variable(
            name="temperature",
            dtype=surrox.DType.CONTINUOUS,
            role=surrox.Role.DECISION,
            bounds=surrox.ContinuousBounds(lower=150.0, upper=300.0),
        ),
        surrox.Variable(
            name="pressure",
            dtype=surrox.DType.CONTINUOUS,
            role=surrox.Role.DECISION,
            bounds=surrox.ContinuousBounds(lower=1.0, upper=10.0),
        ),
    ),
    objectives=(
        surrox.Objective(
            name="maximize_yield",
            direction=surrox.Direction.MAXIMIZE,
            column="yield",
        ),
    ),
)

df = pd.read_csv("experiments.csv")
result, analyzer = surrox.run(problem=problem, dataframe=df)

# Pareto-optimal points
for point in result.optimization.feasible_points:
    print(point.variables, point.objectives)

# On-demand detail analysis
shap = analyzer.shap_global("yield")
pdp = analyzer.pdp_ice("temperature", "yield")
```

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
