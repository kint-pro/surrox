# Getting Started

This guide walks through a complete surrox workflow: defining a problem, running the pipeline, and interpreting results.

## 1. Define the Problem

A [`ProblemDefinition`][surrox.ProblemDefinition] declares variables, objectives, and constraints:

```python
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
        surrox.Variable(
            name="catalyst",
            dtype=surrox.DType.CATEGORICAL,
            role=surrox.Role.DECISION,
            bounds=surrox.CategoricalBounds(categories=("A", "B", "C")),
        ),
    ),
    objectives=(
        surrox.Objective(
            name="maximize_yield",
            direction=surrox.Direction.MAXIMIZE,
            column="yield",
        ),
        surrox.Objective(
            name="minimize_cost",
            direction=surrox.Direction.MINIMIZE,
            column="cost",
        ),
    ),
    data_constraints=(
        surrox.DataConstraint(
            name="emission_limit",
            column="co2",
            operator=surrox.ConstraintOperator.LE,
            limit=100.0,
        ),
    ),
)
```

### Linear Constraints

Analytical constraints on decision variables that are evaluated exactly (no surrogate needed):

```python
problem = surrox.ProblemDefinition(
    ...,
    linear_constraints=(
        surrox.LinearConstraint(
            name="budget",
            coefficients={"temperature": 0.5, "pressure": 10.0},
            operator=surrox.ConstraintOperator.LE,
            rhs=200.0,
        ),
    ),
)
```

### Monotonic Relations

Encode domain knowledge that a variable always has a specific directional effect on a target:

```python
problem = surrox.ProblemDefinition(
    ...,
    monotonic_relations=(
        surrox.MonotonicRelation(
            decision_variable="temperature",
            objective_or_constraint="maximize_yield",
            direction=surrox.MonotonicDirection.INCREASING,
        ),
    ),
)
```

This constrains the surrogate training to respect the declared monotonicity.

## 2. Run the Pipeline

Pass the problem and a DataFrame with historical data to [`run()`][surrox.run]:

```python
import pandas as pd

df = pd.read_csv("experiments.csv")
result, analyzer = surrox.run(problem=problem, dataframe=df)
```

This trains surrogates, optimizes, and produces a summary analysis — all in one call.

## 3. Inspect Results

### Pareto-Optimal Points

The optimization returns feasible points on the Pareto front:

```python
opt = result.optimization

print(f"{opt.n_evaluations} evaluations, strategy: {opt.strategy}")
print(f"{len(opt.feasible_points)} feasible, {len(opt.infeasible_points)} infeasible")
```

```
20000 evaluations, strategy: global_surrogate
47 feasible, 153 infeasible
```

Each point contains variable settings, predicted objectives, constraint evaluations, and extrapolation info:

```python
point = opt.feasible_points[0]
print(point.variables)
print(point.objectives)
print(point.feasible, point.extrapolation_distance, point.is_extrapolating)
```

```
{'temperature': 245.3, 'pressure': 6.8, 'catalyst': 'B'}
{'maximize_yield': 92.1, 'minimize_cost': 34.5}
True 0.42 False
```

### Compromise Solution

For multi-objective problems, the compromise solution (closest to the ideal point) is identified automatically:

```python
idx = opt.compromise_index
compromise = opt.feasible_points[idx]
print("Recommended:", compromise.variables)
print("Objectives:", compromise.objectives)
```

```
Recommended: {'temperature': 231.7, 'pressure': 5.4, 'catalyst': 'B'}
Objectives: {'maximize_yield': 87.3, 'minimize_cost': 28.9}
```

### Summary Analysis

The summary is computed automatically and provides a high-level overview:

```python
summary = result.analysis.summary

print(summary.solution_summary.n_feasible, "feasible solutions")
print("Best per objective:", summary.solution_summary.best_objectives)
print("Hypervolume:", summary.solution_summary.hypervolume)
```

```
47 feasible solutions
Best per objective: {'maximize_yield': 95.8, 'minimize_cost': 22.1}
Hypervolume: 1847.3
```

```python
baseline = summary.baseline_comparison
print("Recommended:", baseline.recommended_objectives)
print("Historical best:", baseline.historical_best_per_objective)
print("Improvement:", baseline.improvement)
```

```
Recommended: {'maximize_yield': 87.3, 'minimize_cost': 28.9}
Historical best: {'maximize_yield': 84.1, 'minimize_cost': 31.2}
Improvement: {'maximize_yield': 3.2, 'minimize_cost': 2.3}
```

```python
for cs in summary.constraint_status:
    print(f"{cs.evaluation.name}: {cs.status} (margin={cs.margin:.2f})")
```

```
emission_limit: satisfied (margin=18.40)
```

```python
for sq in summary.surrogate_quality:
    print(f"{sq.column}: RMSE={sq.cv_rmse:.4f}, coverage={sq.conformal_coverage}, ensemble={sq.ensemble_size}")
```

```
yield: RMSE=1.2340, coverage=0.9, ensemble=5
cost: RMSE=0.8910, coverage=0.9, ensemble=5
co2: RMSE=2.1050, coverage=0.9, ensemble=4
```

## Next Steps

- [Detail Analyses](guides/analysis.md) — SHAP, PDP/ICE, trade-offs, what-if predictions
- [Scenarios](guides/scenarios.md) — compare optimization across operating conditions
- [Configuration](guides/configuration.md) — tune surrogate training, optimizer, and analysis
- [Persistence](guides/persistence.md) — save and load results and surrogates
