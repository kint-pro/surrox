from __future__ import annotations

from typing import Any

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from pydantic import ValidationError

from surrox.exceptions import ConfigurationError
from surrox.optimizer.runner import _select_diverse
from surrox.suggest import ObjectivePrediction, Suggestion, SuggestionResult

safe_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False,
)

positive_floats = st.floats(
    min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False,
)


@st.composite
def objective_predictions(draw: st.DrawFn) -> ObjectivePrediction:
    lower = draw(safe_floats)
    mean = draw(st.floats(
        min_value=lower, max_value=1e6, allow_nan=False, allow_infinity=False,
    ))
    upper = draw(st.floats(
        min_value=mean, max_value=1e6, allow_nan=False, allow_infinity=False,
    ))
    std = draw(positive_floats)
    return ObjectivePrediction(mean=mean, std=std, lower=lower, upper=upper)


@st.composite
def suggestions(draw: st.DrawFn) -> Suggestion:
    n_vars = draw(st.integers(min_value=1, max_value=5))
    variables: dict[str, Any] = {}
    for i in range(n_vars):
        variables[f"x{i}"] = draw(safe_floats)

    n_objs = draw(st.integers(min_value=1, max_value=3))
    objectives: dict[str, ObjectivePrediction] = {}
    for i in range(n_objs):
        objectives[f"obj_{i}"] = draw(objective_predictions())

    return Suggestion(
        variables=variables,
        objectives=objectives,
        extrapolation_distance=draw(st.floats(
            min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False,
        )),
        is_extrapolating=draw(st.booleans()),
    )


@st.composite
def suggestion_results(draw: st.DrawFn) -> SuggestionResult:
    n_suggestions = draw(st.integers(min_value=0, max_value=5))
    suggs = tuple(draw(suggestions()) for _ in range(n_suggestions))

    n_cols = draw(st.integers(min_value=1, max_value=3))
    quality = {
        f"col_{i}": draw(st.floats(
            min_value=-1.0, max_value=1.0,
            allow_nan=False, allow_infinity=False,
        ))
        for i in range(n_cols)
    }

    return SuggestionResult(suggestions=suggs, surrogate_quality=quality)


class TestObjectivePredictionProperties:
    @given(pred=objective_predictions())
    def test_roundtrip_serialization(self, pred: ObjectivePrediction) -> None:
        restored = ObjectivePrediction.model_validate_json(
            pred.model_dump_json()
        )
        assert restored == pred

    @given(pred=objective_predictions())
    def test_frozen(self, pred: ObjectivePrediction) -> None:
        with pytest.raises(ValidationError):
            pred.mean = 0.0  # type: ignore[misc]


class TestSuggestionProperties:
    @given(s=suggestions())
    def test_roundtrip_serialization(self, s: Suggestion) -> None:
        restored = Suggestion.model_validate_json(s.model_dump_json())
        assert restored == s

    @given(s=suggestions())
    def test_always_has_variables(self, s: Suggestion) -> None:
        assert len(s.variables) >= 1

    @given(s=suggestions())
    def test_always_has_objectives(self, s: Suggestion) -> None:
        assert len(s.objectives) >= 1

    @given(s=suggestions())
    def test_extrapolation_distance_non_negative(self, s: Suggestion) -> None:
        assert s.extrapolation_distance >= 0.0

    @given(s=suggestions())
    def test_frozen(self, s: Suggestion) -> None:
        with pytest.raises(ValidationError):
            s.is_extrapolating = False  # type: ignore[misc]


class TestSuggestionResultProperties:
    @given(r=suggestion_results())
    def test_roundtrip_serialization(self, r: SuggestionResult) -> None:
        restored = SuggestionResult.model_validate_json(r.model_dump_json())
        assert restored == r

    @given(r=suggestion_results())
    def test_surrogate_quality_non_empty(self, r: SuggestionResult) -> None:
        assert len(r.surrogate_quality) >= 1

    @given(r=suggestion_results())
    def test_frozen(self, r: SuggestionResult) -> None:
        with pytest.raises(ValidationError):
            r.suggestions = ()  # type: ignore[misc]


@st.composite
def diverse_selection_inputs(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, int]:
    n_pop = draw(st.integers(min_value=1, max_value=50))
    n_dims = draw(st.integers(min_value=1, max_value=10))

    X = np.array(draw(st.lists(
        st.lists(
            st.floats(
                min_value=-100, max_value=100,
                allow_nan=False, allow_infinity=False,
            ),
            min_size=n_dims, max_size=n_dims,
        ),
        min_size=n_pop, max_size=n_pop,
    )), dtype=np.float64)

    n_sorted = draw(st.integers(min_value=0, max_value=n_pop))
    indices = draw(st.permutations(range(n_pop)))
    sorted_indices = np.array(indices[:n_sorted], dtype=int)

    n_select = draw(st.integers(min_value=1, max_value=n_pop + 5))

    return X, sorted_indices, n_select


class TestSelectDiverseProperties:
    @given(inputs=diverse_selection_inputs())
    def test_result_length_bounded(
        self, inputs: tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        X, sorted_indices, n = inputs
        result = _select_diverse(X, sorted_indices, n)
        assert len(result) <= n
        assert len(result) <= len(sorted_indices)

    @given(inputs=diverse_selection_inputs())
    def test_all_indices_from_sorted(
        self, inputs: tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        X, sorted_indices, n = inputs
        result = _select_diverse(X, sorted_indices, n)
        valid = set(sorted_indices.tolist())
        for idx in result:
            assert idx in valid

    @given(inputs=diverse_selection_inputs())
    def test_no_duplicates(
        self, inputs: tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        X, sorted_indices, n = inputs
        result = _select_diverse(X, sorted_indices, n)
        assert len(result) == len(set(result))

    @given(inputs=diverse_selection_inputs())
    def test_first_element_is_best_scored(
        self, inputs: tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        X, sorted_indices, n = inputs
        result = _select_diverse(X, sorted_indices, n)
        if len(result) > 0:
            assert result[0] == int(sorted_indices[0])

    @given(inputs=diverse_selection_inputs())
    def test_returns_all_when_n_exceeds_population(
        self, inputs: tuple[np.ndarray, np.ndarray, int],
    ) -> None:
        X, sorted_indices, n = inputs
        if n >= len(sorted_indices):
            result = _select_diverse(X, sorted_indices, n)
            assert set(result) == set(sorted_indices.tolist())


class TestSuggestInputValidation:
    @given(n=st.integers(max_value=0))
    def test_non_positive_n_suggestions_always_rejected(self, n: int) -> None:
        from surrox.suggest import suggest

        with pytest.raises(ConfigurationError, match="n_suggestions"):
            suggest(
                None, None,  # type: ignore[arg-type]
                n_suggestions=n,
            )

    @given(
        coverage=st.one_of(
            st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=1.0, allow_nan=False, allow_infinity=False),
        ),
    )
    def test_invalid_coverage_always_rejected(self, coverage: float) -> None:
        from surrox.suggest import suggest

        with pytest.raises(ConfigurationError, match="coverage"):
            suggest(
                None, None,  # type: ignore[arg-type]
                coverage=coverage,
            )
