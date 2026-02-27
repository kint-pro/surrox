# Scenarios

Compare optimization results across different operating conditions using [`run_scenarios()`][surrox.run_scenarios].

Surrogates are trained once. Optimization runs independently per scenario.

## Define Scenarios

Scenarios fix context variables to specific values:

```python
import surrox

scenarios = {
    "summer": surrox.Scenario(
        name="summer",
        context_values={"ambient_temp": 35.0},
    ),
    "winter": surrox.Scenario(
        name="winter",
        context_values={"ambient_temp": -5.0},
    ),
}

result, analyzers = surrox.run_scenarios(
    problem=problem,
    dataframe=df,
    scenarios=scenarios,
)
```

## Per-Scenario Results

Each scenario has its own full result and analyzer:

```python
for name, scenario_result in result.per_scenario.items():
    opt = scenario_result.optimization
    print(f"{name}: {len(opt.feasible_points)} feasible points")
```

```
summer: 42 feasible points
winter: 51 feasible points
```

Detail analyses are available per scenario:

```python
summer_analyzer = analyzers["summer"]
shap = summer_analyzer.shap_global("yield")
```

## Scenario Comparison

The comparison identifies which decision variables are robust across scenarios:

```python
for var_name, robustness in result.comparison.variable_robustness.items():
    print(f"{var_name}:")
    print(f"  values: {robustness.values_per_scenario}")
    print(f"  robust: {robustness.is_robust} (spread={robustness.spread:.2f})")
```

```
temperature:
  values: {'summer': 228.4, 'winter': 241.9}
  robust: False (spread=13.50)
pressure:
  values: {'summer': 5.3, 'winter': 5.5}
  robust: True (spread=0.20)
catalyst:
  values: {'summer': 'B', 'winter': 'B'}
  robust: True (spread=0.00)
```

A variable is considered robust when its spread is less than 5% of the bounds range. Robust variables can be set to a single value regardless of scenario; non-robust variables need scenario-specific tuning.
