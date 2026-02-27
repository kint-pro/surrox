# ADR-021: stdlib logging, no structlog dependency

## Status

Accepted

## Date

2026-02-27

## Context

Surrox needs observability for enterprise multi-tenant deployments (TenneT). Training surrogates, optimizing, and analyzing results are long-running operations where operators need visibility into progress, duration, and quality metrics.

Structured logging (key-value context on every log event) is the industry standard for enterprise observability, enabling integration with ELK, Datadog, Grafana Loki, and similar platforms.

### Options

**Option A — structlog as dependency:** Rich structured logging out of the box. Adds a runtime dependency to surrox. Applications that already use a different logging stack must reconcile two systems.

**Option B — stdlib `logging` with structured `extra` dicts:** Zero dependencies. Libraries emit structured context via the `extra` parameter on log calls. Applications (kint) are free to add structlog, JSON formatters, or any handler on top. Follows the Python library best practice: libraries log, applications configure.

## Decision

Option B. Surrox uses only `logging.getLogger(__name__)` per module and passes structured data via `extra` dicts. No logging configuration, no handlers, no formatters — that is the application's responsibility.

## Rationale

- Python library best practice: libraries should never configure logging or add dependencies for it. The Twelve-Factor App methodology and structlog's own documentation recommend this separation.
- Zero additional dependencies. Surrox already has a lean dependency tree; adding structlog for a library that will always run inside an application (kint) is unnecessary coupling.
- kint can adopt structlog, standard JSON formatters, or OpenTelemetry log bridges without any change to surrox.
- The `extra` dict pattern is forward-compatible: structlog can extract these fields if kint configures it as a stdlib wrapper.

## Consequences

- Every surrox module uses `logging.getLogger(__name__)` — no custom logger factory.
- Structured context (column name, duration, r2 scores, etc.) is passed via `extra={}` on every log call.
- Log events use a `{module}.{event}` naming convention (e.g., `surrogate training complete`).
- A `log_duration` context manager provides consistent timing for long-running operations.
- kint is responsible for configuring handlers, formatters, and log levels. Surrox never calls `logging.basicConfig()` or adds handlers.
