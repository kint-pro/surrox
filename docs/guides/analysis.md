# Detail Analyses

The [`Analyzer`][surrox.Analyzer] provides on-demand, cached analyses. All results are computed lazily on first access and cached for subsequent calls.

## Feature Importance

```python
importance = analyzer.feature_importance("yield")
print(importance.importances)
print(importance.decision_importances)
```

```
{'temperature': 4.21, 'pressure': 2.87, 'catalyst': 1.05}
{'temperature': 4.21, 'pressure': 2.87, 'catalyst': 1.05}
```

## SHAP Explanations

Global SHAP values show how each feature contributes across the background dataset:

```python
shap_global = analyzer.shap_global("yield")
print("Features:", shap_global.feature_names)
print("Base value:", shap_global.base_value)
print("SHAP matrix shape:", shap_global.shap_values.shape)
```

```
Features: ('temperature', 'pressure', 'catalyst')
Base value: 72.5
SHAP matrix shape: (100, 3)
```

Local SHAP explains a specific Pareto-optimal point:

```python
shap_local = analyzer.shap_local("yield", point_index=0)
print("Predicted:", shap_local.predicted_value)
print("Base:", shap_local.base_value)
for name, sv in zip(shap_local.feature_names, shap_local.shap_values):
    print(f"  {name}: {sv:+.2f} (value={shap_local.feature_values[name]:.1f})")
```

```
Predicted: 92.1
Base: 72.5
  temperature: +14.30 (value=245.3)
  pressure: +4.80 (value=6.8)
  catalyst: +0.50 (value=1.0)
```

## PDP/ICE Curves

Partial Dependence Plots show the marginal effect of a variable on a target:

```python
pdp = analyzer.pdp_ice("temperature", "yield")
print("Grid:", pdp.grid_values[:5], "...")
print("PDP:", pdp.pdp_values[:5], "...")
print("ICE shape:", pdp.ice_values.shape)  # (n_samples, n_grid_points)
```

```
Grid: [150. 153.06 156.12 159.18 162.24] ...
PDP: [61.2 62.1 63.4 65.0 66.8] ...
ICE shape: (100, 50)
```

## Trade-Off Analysis

For multi-objective problems, trade-off analysis computes marginal rates of substitution between objective pairs along the Pareto front:

```python
trade_off = analyzer.trade_off()
for pair in trade_off.objective_pairs:
    rates = trade_off.marginal_rates[pair]
    print(f"{pair[0]} vs {pair[1]}: median rate = {rates[len(rates)//2]:.2f}")
```

```
maximize_yield vs minimize_cost: median rate = -1.35
```

## What-If Predictions

Predict outcomes for hypothetical variable settings with uncertainty intervals:

```python
what_if = analyzer.what_if({
    "temperature": 250.0,
    "pressure": 5.0,
    "catalyst": "B",
})

print(f"Extrapolating: {what_if.is_extrapolating} (distance={what_if.extrapolation_distance:.2f})")

for name, pred in what_if.objectives.items():
    print(f"{name}:")
    print(f"  predicted={pred.predicted:.1f} [{pred.lower:.1f}, {pred.upper:.1f}]")
    print(f"  vs recommended={pred.recommended_value:.1f}, historical mean={pred.historical_mean:.1f}")
```

```
Extrapolating: False (distance=0.31)
maximize_yield:
  predicted=89.4 [85.2, 93.6]
  vs recommended=87.3, historical mean=71.8
minimize_cost:
  predicted=31.2 [27.8, 34.6]
  vs recommended=28.9, historical mean=42.3
```
