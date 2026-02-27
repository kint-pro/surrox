# surrox

**Blackbox surrogate-based optimization framework for the kint platform.**

surrox takes a declarative problem description, rains surrogate models on historical datta, optimizes on the surrogates, and delivers explained, uncertainty-assessed recommendations.

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
