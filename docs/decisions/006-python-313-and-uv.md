# ADR-006: Python 3.13 and uv Package Manager

## Status

Accepted

## Date

2026-02-26

## Context

The framework depends on ML libraries with varying Python version support:

| Library | Min Python | Max Python |
|---------|-----------|-----------|
| XGBoost | 3.10 | 3.14 |
| LightGBM | 3.7 | 3.13 |
| pymoo | 3.10 | 3.14 |
| SHAP | 3.11 | 3.14 |
| Optuna | 3.9 | 3.14 |

The bottleneck is LightGBM, which officially supports up to Python 3.13.

For package management, uv was chosen over pip/Poetry for development speed, but the project must remain installable via standard pip.

## Decision

- Target Python >=3.13
- Use uv as the development package manager
- Use pyproject.toml (PEP 621) for package metadata — pip-compatible by design

## Rationale

- **Python 3.13**: Highest version supported by all critical dependencies. Python 3.14 would exclude LightGBM.
- **uv**: 10-100x faster than pip for dependency resolution and installation. Uses standard pyproject.toml, so end users can install with `pip install surrox` without uv.
- **pyproject.toml**: PEP 621 standard. uv, pip, Poetry, and hatch all read it. No lock-in to any tool.

## Consequences

- `.python-version` file pins 3.13 for development
- `uv.lock` is committed for reproducible dev environments
- End users install with `pip install surrox` or `uv pip install surrox` — both work
- When LightGBM adds Python 3.14 support, the minimum can be bumped
