# Skill: Hypothesis Property-Based Tests

Write property-based tests using Hypothesis for Pydantic v2 frozen models in the surrox framework.

## When to Use

When writing tests that verify invariants, roundtrips, or validation properties across many random inputs — instead of manually crafting individual test cases.

## Strategy Patterns for Pydantic v2

Pydantic v2 has NO built-in Hypothesis plugin. Use `st.builds()` and `@st.composite` exclusively.

### Basic: `st.builds` for simple models

```python
from hypothesis import strategies as st

bounds_strategy = st.builds(
    ContinuousBounds,
    lower=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    upper=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
).filter(lambda b: b.lower < b.upper)
```

### Preferred: `@st.composite` for cross-dependent fields

```python
@st.composite
def continuous_bounds(draw):
    lower = draw(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
    upper = draw(st.floats(min_value=lower + 1e-9, max_value=1e6, allow_nan=False, allow_infinity=False))
    return ContinuousBounds(lower=lower, upper=upper)
```

Prefer `@st.composite` over `.filter()` when rejection rate would exceed ~50%. Constrain at generation time, not after.

### StrEnum fields

```python
st.sampled_from(DType)
st.sampled_from(Role)
st.sampled_from(Direction)
```

### Discriminated unions (Bounds)

```python
bounds_strategy = st.one_of(
    continuous_bounds(),
    integer_bounds(),
    categorical_bounds(),w
    ordinal_bounds(),
)
```

Place simpler strategies first in `st.one_of` — shrinking prefers earlier options.

## Property Categories to Test

| Category | What It Proves | Template |
|---|---|---|
| **Roundtrip** | Serialization fidelity | `Model.model_validate_json(m.model_dump_json()) == m` |
| **Invariant** | Construction guarantees hold | `len(problem.objectives) >= 1` always |
| **Rejection** | Invalid inputs always rejected | `ContinuousBounds(lower=x, upper=x)` always raises |
| **Idempotent** | Repeated operation is stable | `sorted(sorted(xs)) == sorted(xs)` |
| **Metamorphic** | Related inputs → related outputs | Adding a variable doesn't remove existing ones |

Focus on **Roundtrip** and **Invariant** properties for immutable domain models. Focus on **Rejection** properties for validation logic.

## Strategy Organization

Place strategies in `tests/strategies.py` (central) or per-layer `tests/<layer>/strategies.py`.

```python
# tests/strategies.py
from hypothesis import strategies as st

@st.composite
def variable_names(draw):
    return draw(st.from_regex(r"[a-z][a-z0-9_]{1,19}", fullmatch=True))
```

Optionally register for `st.from_type()` resolution:

```python
# tests/conftest.py
from hypothesis import register_type_strategy
register_type_strategy(Variable, variable_strategy)
```

## Hypothesis Profiles

Configure in `tests/conftest.py`:

```python
from hypothesis import settings, HealthCheck

settings.register_profile("dev", max_examples=20, deadline=400)
settings.register_profile("ci", max_examples=500, derandomize=True, deadline=None)
settings.register_profile("full", max_examples=10_000, deadline=None)

import os
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
```

Run with: `HYPOTHESIS_PROFILE=ci uv run pytest tests/ -v`

## Rules

1. **Never use `assume()` when you can constrain the strategy.** `assume()` discards generated examples — wasteful and triggers `filter_too_much` health checks.

2. **Never use `.filter()` with >50% rejection rate.** Restructure the strategy to only produce valid inputs.

3. **Never suppress all health checks.** Only suppress specific checks you understand and accept.

4. **Never use `st.just(x)` as the primary strategy.** That defeats the purpose — use real strategies.

5. **Always constrain `st.text()` and `st.floats()`:**
   - `st.text(min_size=1, max_size=50)` — unbounded text generates huge strings.
   - `st.floats(allow_nan=False, allow_infinity=False, min_value=..., max_value=...)` — NaN/Inf break comparisons.

6. **Always keep Hypothesis database enabled** (default `DirectoryBasedExampleDatabase`). It replays previously failing examples.

7. **Test the public API, not implementation details.** Properties must survive refactoring.

8. **One property per test function.** Mixing properties makes failures hard to diagnose.

9. **Use `@example(...)` for known edge cases** alongside `@given` for random generation. Explicit examples run in `Phase.explicit`.

10. **Mutable pytest fixtures are shared across all examples in one test invocation.** Create fresh state inside the test body, not via fixtures.

## Test File Template

```python
from hypothesis import given, settings, example
from hypothesis import strategies as st
from pydantic import ValidationError
import pytest

from surrox.problem import ContinuousBounds, Variable, ProblemDefinition


class TestBoundsProperties:

    @given(
        lower=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        upper=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    def test_equal_bounds_always_rejected(self, lower, upper):
        from hypothesis import assume
        assume(lower >= upper)
        with pytest.raises(ValidationError):
            ContinuousBounds(lower=lower, upper=upper)

    @given(bounds=continuous_bounds())
    def test_roundtrip_serialization(self, bounds):
        restored = ContinuousBounds.model_validate_json(bounds.model_dump_json())
        assert restored == bounds

    @given(bounds=continuous_bounds())
    def test_lower_always_less_than_upper(self, bounds):
        assert bounds.lower < bounds.upper
```

## Shrinking

Hypothesis automatically reduces failing inputs to the minimal reproducing case:
- Integers shrink toward 0
- Text shrinks toward shorter, simpler strings
- Lists shrink toward fewer, simpler elements
- `st.one_of(a, b)` shrinks toward earlier strategies
- `st.builds()` shrinks each argument independently

No manual intervention needed. If shrinking is slow, simplify the strategy composition.
