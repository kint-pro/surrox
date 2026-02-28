# Architecture

## Layer Overview

```bash
┌─────────────────────────────────────────────────┐
│                  ProblemDefinition              │
│  Variables, Objectives, Constraints, Scenarios  │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│               SurrogateManager                  │
│  Optuna HPO → Ensemble Selection → Conformal Prediction │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│                  Optimizer                      │
│  pymoo (DE / GA / NSGA-II / NSGA-III)           │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│                  Analysis                       │
│  Summary (auto) + Detail (lazy): SHAP, PDP, ... │
└─────────────────────────────────────────────────┘
```

## Data Flow

1. **ProblemDefinition** is constructed and validated. It flows through every layer as an immutable reference.
2. **BoundDataset** binds a DataFrame to the problem, validating column presence, dtypes, bounds, and missing values.
3. **SurrogateManager** trains one ensemble per `surrogate_column` (deduplicated across objectives and data constraints). Each ensemble is an Optuna-selected set of XGBoost/LightGBM models with conformal calibration.
4. **Optimizer** evaluates candidates on surrogates, applies linear and data constraints, detects extrapolation, and returns Pareto-optimal points.
5. **Analysis** produces a summary automatically. The Analyzer object allows lazy, cached detail analyses.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Immutable models (frozen Pydantic) | Prevents accidental mutation across layers |
| One surrogate per unique column | Objectives and data constraints sharing a column reuse the same surrogate (ADR-007) |
| Conformal prediction | Distribution-free uncertainty intervals with coverage guarantees |
| Extrapolation detection | k-NN distance flags candidates outside the training domain |
| Lazy detail analysis | SHAP/PDP are expensive — computed only when requested and cached |
| Facade functions (`run`, `run_scenarios`) | Simple entry points hiding the layer orchestration |
