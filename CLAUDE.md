# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv run pytest tests/ -v              # all tests
uv run pytest tests/problem/ -v      # problem layer tests
uv run pytest tests/problem/test_definition.py::TestProblemDefinitionValidation::test_no_objectives_raises -v  # single test
uv run ruff check src/               # lint
uv run ruff format --check src/      # format check
uv run ruff format src/              # auto-format
uv run zensical build                # build docs
uv run zensical serve                # serve docs locally
```

Python 3.13 is pinned (LightGBM compatibility ceiling). Use `uv sync` to install dependencies.

## Architecture

Surrox is a blackbox surrogate-based optimization framework for the kint platform. It takes a declarative problem description, trains surrogate models, optimizes on them, and delivers explained, uncertainty-assessed recommendations. No data cleaning, no visualization, no interpretation — that is kint's responsibility.

**4 layers, each consuming the previous + the Problem object:**

1. **Problem** (implemented) — Immutable declarative problem definition. Central object consumed by all layers.
2. **SurrogateManager** (not yet) — Trains one ensemble per unique target column (Optuna HPO, XGBoost/LightGBM, Conformal Prediction).
3. **Optimizer** (not yet) — pymoo-based, auto-selects algorithm from problem structure.
4. **Analysis** (not yet) — Summary (auto) + Detail analyses (lazy/cached): SHAP, PDP/ICE, Feature Importance, What-If.

### Problem Layer (`src/surrox/problem/`)

All models are Pydantic v2 with `frozen=True`. Deep immutability: frozen models + tuple collections + frozen contained objects.

- `types.py` — StrEnum types: DType, Role, Direction, MonotonicDirection, ConstraintOperator
- `variables.py` — Bounds (discriminated union via `Field(discriminator="type")`) + Variable
- `objectives.py` — Objective (name, direction, column, optional reference_value)
- `constraints.py` — LinearConstraint (analytical, coefficients dict) + DataConstraint (surrogate-based)
- `domain_knowledge.py` — MonotonicRelation (decision_variable → objective_or_constraint)
- `scenarios.py` — Scenario (named context variable assignments)
- `definition.py` — ProblemDefinition (central container, cross-field validation, derived properties)
- `dataset.py` — BoundDataset (validates DataFrame against ProblemDefinition)

Key property: `ProblemDefinition.surrogate_columns` returns deduplicated column names — one surrogate per unique column, even when shared by Objective and DataConstraint (ADR-007).

## Design Principles

- **Fail-fast**: All validation at construction time. No silent correction, no fallback, no alternative paths.
- **Spec-driven**: Specs in `docs/specs/`, ADRs in `docs/decisions/`. Code must match specs.
- **No comments in code**: Code must be self-explanatory through descriptive names.
- **Iterative**: One layer at a time — research → spec → implement → test.
- **No dead code, no workarounds, no temporary solutions**.

## Specs and ADRs

- `docs/specs/problem/spec.md` — Authoritative specification for the Problem layer
- `docs/decisions/001-007` — Architectural Decision Records

When specs and the initial prompt (`start-prompt.txt`) conflict, specs and ADRs are the binding reference.
