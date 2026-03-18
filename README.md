# surrox

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)

> [!WARNING]
> This repository is currently being set up. APIs, documentation, and project structure may change without notice.

Blackbox surrogate-based optimization framework for the [kint](https://github.com/kint-pro) platform.

surrox takes a declarative problem description, trains surrogate models, optimizes on them, and delivers explained, uncertainty-assessed recommendations.

## Features

- **Declarative problem definition** — variables, objectives, constraints, scenarios as immutable Pydantic models
- **Ensemble surrogates** — XGBoost, LightGBM, Gaussian Process, TabICL with Optuna HPO
- **Uncertainty quantification** — Conformal Prediction for calibrated prediction intervals
- **Multi-objective optimization** — pymoo-based, auto-selects algorithm (DE/GA/NSGA-II/NSGA-III)
- **Explainability** — SHAP, PDP/ICE, feature importance, what-if analysis, scenario comparison

## Architecture

```
Problem → SurrogateManager → Optimizer → Analysis
```

| Layer | Responsibility |
|---|---|
| **Problem** | Immutable declarative problem definition (variables, objectives, constraints, scenarios) |
| **SurrogateManager** | Ensemble training per target column, Optuna HPO, Conformal Prediction |
| **Optimizer** | pymoo-based multi-objective optimization, auto-selects algorithm from problem structure |
| **Analysis** | SHAP, PDP/ICE, Feature Importance, What-If, Scenario Comparison |

## Installation

Requires Python 3.13.

```bash
pip install surrox            # core (problem definition only)
pip install surrox[all]       # all dependencies (ML + optimization)
```

## Quick Start

```python
from surrox import run
from surrox.problem import (
    ProblemDefinition, Variable, Objective,
    ContinuousBounds, Direction, Role,
)

problem = ProblemDefinition(
    variables=(
        Variable(name="temperature", role=Role.DECISION,
                 bounds=ContinuousBounds(lower=150.0, upper=300.0)),
        Variable(name="pressure", role=Role.DECISION,
                 bounds=ContinuousBounds(lower=1.0, upper=10.0)),
    ),
    objectives=(
        Objective(name="maximize_yield", direction=Direction.MAXIMIZE, column="yield"),
    ),
)

result = run(problem=problem, data=df)
```

## Development

```bash
uv sync
uv run pytest tests/ -v
uv run ruff check src/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

[MIT](LICENSE)
