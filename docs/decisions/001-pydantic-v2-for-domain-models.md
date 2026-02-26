# ADR-001: Pydantic v2 for Domain Models

## Status

Accepted

## Date

2026-02-26

## Context

The Problem layer needs immutable, validated domain objects that flow unchanged through all framework layers. Options considered:

1. **Python dataclasses** — Standard library, `frozen=True` for immutability. Manual validation in `__post_init__`. No built-in serialization.
2. **attrs** — Third-party, mature. Validators as decorators. No built-in JSON serialization.
3. **Pydantic v2** — Third-party, widely adopted. `frozen=True`, `model_validator`, built-in JSON serialization/deserialization, discriminated unions, type coercion.

## Decision

Use Pydantic v2 with `ConfigDict(frozen=True)` for all domain models.

## Rationale

- **Validation at construction**: `model_validator(mode="after")` enforces cross-field invariants at creation time. No separate validation step needed.
- **Immutability**: `frozen=True` prevents mutation after construction. Deep immutability is achieved through the combination of three layers: `frozen=True` on every model (Variable, Objective, etc.), `tuple` instead of `list` for collections (prevents structural mutation), and the fact that all contained objects are themselves frozen Pydantic models. No single mechanism is sufficient — `tuple` alone would not prevent mutation of mutable elements, and `frozen=True` alone would not prevent replacing list contents.
- **Serialization**: `model_dump()` / `model_validate()` provide JSON round-tripping for free. Required for API communication between kint and the framework.
- **Discriminated unions**: Bounds types (`ContinuousBounds | IntegerBounds | ...`) are naturally expressed as Pydantic discriminated unions with a `type` literal field.
- **Ecosystem**: Pydantic v2 is the de facto standard for validated data models in Python. FastAPI, LangChain, and most modern Python frameworks use it.

## Consequences

- Pydantic v2 becomes a core dependency (not optional)
- All domain models inherit from `BaseModel`, not `dataclass`
- Validation errors are wrapped in `pydantic.ValidationError` (consumers catch this, not raw `ValueError`)
- Slight runtime overhead compared to plain dataclasses (negligible for domain objects that are created once)
