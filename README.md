# surrox

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)

Your simulation takes 4 hours per run. You can't afford grid search. surrox trains surrogates on your existing data, optimizes on the cheap models instead, and tells you exactly why it picked each solution.

```
Data → Ensemble Surrogates → Multi-Objective Optimization → Explained Results
```

## Quick Start

```bash
pip install surrox[all]
```

```python
import numpy as np
import pandas as pd
from surrox import run, ProblemDefinition, Variable, Objective
from surrox import ContinuousBounds, DType, Direction, Role

rng = np.random.default_rng(42)
temperature = rng.uniform(150, 300, 500)
pressure = rng.uniform(1, 10, 500)
df = pd.DataFrame({
    "temperature": temperature,
    "pressure": pressure,
    "yield": 0.2 + 0.002 * temperature + 0.03 * pressure + rng.normal(0, 0.05, 500),
})

problem = ProblemDefinition(
    variables=(
        Variable(name="temperature", dtype=DType.CONTINUOUS, role=Role.DECISION,
                 bounds=ContinuousBounds(lower=150.0, upper=300.0)),
        Variable(name="pressure", dtype=DType.CONTINUOUS, role=Role.DECISION,
                 bounds=ContinuousBounds(lower=1.0, upper=10.0)),
    ),
    objectives=(
        Objective(name="maximize_yield", direction=Direction.MAXIMIZE, column="yield"),
    ),
)

result, analyzer = run(problem=problem, dataframe=df)

best = result.optimization.feasible_points[0]
print(best.variables)    # {"temperature": 284.0, "pressure": 9.06}
print(best.objectives)   # {"maximize_yield": 1.066}

importance = analyzer.feature_importance("yield")
print(importance.importances)  # {"temperature": 0.083, "pressure": 0.070}
```

## What it does

You define the problem declaratively. surrox handles the rest.

**Surrogates** — Trains an ensemble per target column (XGBoost, LightGBM, Gaussian Process). Optuna picks the hyperparameters. Conformal Prediction gives you calibrated uncertainty intervals, not just point estimates.

**Optimization** — Runs pymoo under the hood. Auto-selects the right algorithm based on your problem structure: DE for single-objective, NSGA-II for two objectives, NSGA-III for three or more. Supports linear and data-driven constraints.

**Analysis** — Every result comes with a summary: surrogate quality, constraint status, baseline comparison, extrapolation warnings. For deeper inspection, the `Analyzer` provides SHAP explanations, PDP/ICE curves, feature importance, what-if predictions, and trade-off analysis. All computed lazily and cached.

**Scenarios** — Fix context variables to specific values and compare outcomes across scenarios. Train surrogates once, optimize each scenario independently.

## Suggest mode

Don't need the full pipeline? `suggest` gives you the top N candidates with uncertainty bounds.

```python
from surrox import suggest

result = suggest(problem=problem, dataframe=df, n_suggestions=5)

for s in result.suggestions:
    print(s.variables)                            # {"temperature": 284.0, "pressure": 9.06}
    print(s.objectives["maximize_yield"].mean)    # 1.066
    print(s.objectives["maximize_yield"].lower)   # 0.978 (90% CI lower bound)
    print(s.is_extrapolating)                     # False
```

## Problem definition

Everything is a Pydantic model. Immutable, validated at construction, type-safe.

```python
from surrox import (
    ProblemDefinition, Variable, Objective, DataConstraint, Scenario,
    ContinuousBounds, IntegerBounds, CategoricalBounds,
    DType, Role, Direction, ConstraintOperator,
)

problem = ProblemDefinition(
    variables=(
        Variable(name="speed", dtype=DType.CONTINUOUS, role=Role.DECISION,
                 bounds=ContinuousBounds(lower=10.0, upper=100.0)),
        Variable(name="material", dtype=DType.CATEGORICAL, role=Role.DECISION,
                 bounds=CategoricalBounds(categories=("steel", "aluminum", "titanium"))),
        Variable(name="batch_size", dtype=DType.INTEGER, role=Role.CONTEXT,
                 bounds=IntegerBounds(lower=50, upper=500)),
    ),
    objectives=(
        Objective(name="max_strength", direction=Direction.MAXIMIZE, column="strength"),
        Objective(name="min_cost", direction=Direction.MINIMIZE, column="cost"),
    ),
    data_constraints=(
        DataConstraint(name="max_defect_rate", column="defect_rate",
                       operator=ConstraintOperator.LE, bound=0.05),
    ),
)
```

## Architecture

| Layer | What it does |
|---|---|
| **Problem** | Immutable problem definition: variables, objectives, constraints, scenarios |
| **SurrogateManager** | Ensemble training, Optuna HPO, Conformal Prediction |
| **Optimizer** | pymoo multi-objective optimization with constraint handling |
| **Analysis** | SHAP, PDP/ICE, What-If, Trade-Off, Scenario Comparison |

## Development

```bash
uv sync
uv run pytest tests/ -v
uv run ruff check src/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

[MIT](LICENSE)
