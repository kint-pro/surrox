from __future__ import annotations

import pytest
from hypothesis import given, example
from hypothesis import strategies as st
from pydantic import ValidationError

from surrox.analysis.config import AnalysisConfig
from surrox.analysis.summary import (
    BaselineComparison,
    ConstraintStatus,
    ExtrapolationWarning,
    MonotonicityViolation,
    SolutionSummary,
    SurrogateQuality,
)
from surrox.analysis.what_if import WhatIfPrediction
from surrox.exceptions import AnalysisError

from tests.analysis.strategies import (
    analysis_configs,
    baseline_comparisons,
    constraint_statuses,
    extrapolation_warnings,
    monotonicity_violations,
    solution_summaries,
    surrogate_qualities,
    what_if_predictions,
)


class TestAnalysisConfigProperties:
    @given(config=analysis_configs())
    def test_roundtrip_serialization(self, config: AnalysisConfig) -> None:
        restored = AnalysisConfig.model_validate_json(config.model_dump_json())
        assert restored == config

    @given(config=analysis_configs())
    def test_invariant_percentiles_ordered(self, config: AnalysisConfig) -> None:
        lo, hi = config.pdp_percentiles
        assert 0 < lo < hi < 1

    @given(config=analysis_configs())
    def test_invariant_sizes_at_least_10(self, config: AnalysisConfig) -> None:
        assert config.shap_background_size >= 10
        assert config.pdp_grid_resolution >= 10
        assert config.monotonicity_check_resolution >= 10

    @given(size=st.integers(min_value=-1000, max_value=9))
    def test_rejection_shap_background_too_small(self, size: int) -> None:
        with pytest.raises((AnalysisError, ValidationError)):
            AnalysisConfig(shap_background_size=size)

    @given(size=st.integers(min_value=-1000, max_value=9))
    def test_rejection_pdp_resolution_too_small(self, size: int) -> None:
        with pytest.raises((AnalysisError, ValidationError)):
            AnalysisConfig(pdp_grid_resolution=size)

    @given(size=st.integers(min_value=-1000, max_value=9))
    def test_rejection_monotonicity_resolution_too_small(self, size: int) -> None:
        with pytest.raises((AnalysisError, ValidationError)):
            AnalysisConfig(monotonicity_check_resolution=size)

    @given(
        lo=st.floats(min_value=0.5, max_value=0.99, allow_nan=False, allow_infinity=False),
        hi=st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False),
    )
    def test_rejection_percentiles_inverted(self, lo: float, hi: float) -> None:
        if lo >= hi:
            with pytest.raises((AnalysisError, ValidationError)):
                AnalysisConfig(pdp_percentiles=(lo, hi))

    @example(lo=0.0, hi=0.5)
    @example(lo=0.5, hi=1.0)
    @given(
        lo=st.floats(allow_nan=False, allow_infinity=False),
        hi=st.floats(allow_nan=False, allow_infinity=False),
    )
    def test_rejection_percentiles_at_boundary(self, lo: float, hi: float) -> None:
        if lo <= 0 or hi >= 1 or lo >= hi:
            with pytest.raises((AnalysisError, ValidationError)):
                AnalysisConfig(pdp_percentiles=(lo, hi))


class TestSolutionSummaryProperties:
    @given(summary=solution_summaries())
    def test_roundtrip_serialization(self, summary: SolutionSummary) -> None:
        restored = SolutionSummary.model_validate_json(summary.model_dump_json())
        assert restored == summary

    @given(summary=solution_summaries())
    def test_frozen(self, summary: SolutionSummary) -> None:
        with pytest.raises(ValidationError):
            summary.n_feasible = 999  # type: ignore[misc]


class TestBaselineComparisonProperties:
    @given(comparison=baseline_comparisons())
    def test_roundtrip_serialization(self, comparison: BaselineComparison) -> None:
        restored = BaselineComparison.model_validate_json(comparison.model_dump_json())
        assert restored == comparison

    @given(comparison=baseline_comparisons())
    def test_keys_consistent(self, comparison: BaselineComparison) -> None:
        assert set(comparison.recommended_objectives.keys()) == set(
            comparison.historical_best_per_objective.keys()
        )
        assert set(comparison.recommended_objectives.keys()) == set(
            comparison.improvement.keys()
        )


class TestConstraintStatusProperties:
    @given(status=constraint_statuses())
    def test_roundtrip_json(self, status: ConstraintStatus) -> None:
        data = status.model_dump(mode="python")
        restored = ConstraintStatus.model_validate(data)
        assert restored.status == status.status
        assert restored.margin == status.margin
        assert restored.evaluation.name == status.evaluation.name


class TestSurrogateQualityProperties:
    @given(quality=surrogate_qualities())
    def test_roundtrip_serialization(self, quality: SurrogateQuality) -> None:
        restored = SurrogateQuality.model_validate_json(quality.model_dump_json())
        assert restored == quality


class TestExtrapolationWarningProperties:
    @given(warning=extrapolation_warnings())
    def test_roundtrip_serialization(self, warning: ExtrapolationWarning) -> None:
        restored = ExtrapolationWarning.model_validate_json(warning.model_dump_json())
        assert restored == warning


class TestMonotonicityViolationProperties:
    @given(violation=monotonicity_violations())
    def test_roundtrip_serialization(self, violation: MonotonicityViolation) -> None:
        restored = MonotonicityViolation.model_validate_json(
            violation.model_dump_json()
        )
        assert restored == violation

    @given(violation=monotonicity_violations())
    def test_violation_fraction_in_range(self, violation: MonotonicityViolation) -> None:
        assert 0.0 <= violation.violation_fraction <= 1.0


class TestWhatIfPredictionProperties:
    @given(prediction=what_if_predictions())
    def test_roundtrip_serialization(self, prediction: WhatIfPrediction) -> None:
        restored = WhatIfPrediction.model_validate_json(prediction.model_dump_json())
        assert restored == prediction
