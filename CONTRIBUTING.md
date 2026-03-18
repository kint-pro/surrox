# Contributing to surrox

## How to Contribute

- **Bugs**: Open an issue using the [bug report template](https://github.com/kint-pro/surrox/issues/new?template=bug_report.yml)
- **Features**: Open an issue using the [feature request template](https://github.com/kint-pro/surrox/issues/new?template=feature_request.yml)
- **Code**: Fork the repo, create a branch from `main`, and open a pull request

## Development Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/kint-pro/surrox.git
cd surrox
uv sync
```

## Tests

```bash
uv run pytest tests/ -v
```

## Linting

```bash
uv run ruff check src/
uv run ruff format --check src/
uv run ruff format src/          # auto-format
```

## Pull Requests

- One concern per PR
- Branch from `main`
- Reference the related issue
- Ensure tests pass and linting is clean

## Code Style

- All code in English
- No comments — code must be self-explanatory
- Fail-fast: validate at construction time, never silently correct
- Follow existing patterns in the codebase
- Specs live in `docs/specs/`, ADRs in `docs/decisions/` — code must match them
