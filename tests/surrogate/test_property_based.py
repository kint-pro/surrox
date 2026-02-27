from __future__ import annotations

from math import ceil

import hypothesis.strategies as st
import numpy as np
from hypothesis import given

from surrox.surrogate.config import TrainingConfig
from surrox.surrogate.pipeline import _softmax, _validate_minimum_data

valid_config_values = st.fixed_dictionaries({
    "n_trials": st.integers(min_value=1, max_value=1000),
    "cv_folds": st.integers(min_value=2, max_value=20),
    "calibration_fraction": st.floats(
        min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
    ),
    "ensemble_size": st.integers(min_value=1, max_value=50),
    "diversity_threshold": st.floats(
        min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    "softmax_temperature": st.floats(
        min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False
    ),
    "default_coverage": st.floats(
        min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
    ),
    "study_timeout_s": st.integers(min_value=1, max_value=3600),
    "random_seed": st.integers(min_value=0, max_value=2**31),
})


class TestTrainingConfigProperties:
    @given(values=valid_config_values)
    def test_valid_values_always_construct(self, values: dict) -> None:
        TrainingConfig(**values)

    @given(values=valid_config_values)
    def test_roundtrip_via_model_dump(self, values: dict) -> None:
        config = TrainingConfig(**values)
        data = config.model_dump()
        data.pop("estimator_families")
        data.pop("n_threads")
        data.pop("min_r2")
        restored = TrainingConfig(**data)
        assert restored.n_trials == config.n_trials
        assert restored.cv_folds == config.cv_folds
        assert restored.calibration_fraction == config.calibration_fraction


class TestSoftmaxProperties:
    @given(
        values=st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20,
        ),
        temperature=st.floats(
            min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    )
    def test_sums_to_one(self, values: list[float], temperature: float) -> None:
        result = _softmax(np.array(values), temperature)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-10)

    @given(
        values=st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20,
        ),
        temperature=st.floats(
            min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    )
    def test_all_non_negative(self, values: list[float], temperature: float) -> None:
        result = _softmax(np.array(values), temperature)
        assert (result >= 0).all()

    @given(
        values=st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=20,
            unique=True,
        ),
        t1=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
        t2=st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    def test_argmax_invariant_across_temperatures(
        self, values: list[float], t1: float, t2: float
    ) -> None:
        arr = np.array(values)
        result1 = _softmax(arr, t1)
        result2 = _softmax(arr, t2)
        assert np.argmax(result1) == np.argmax(result2)


class TestValidateMinimumDataProperties:
    @given(
        cv_folds=st.integers(min_value=2, max_value=20),
        calibration_fraction=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
    )
    def test_at_minimum_passes(self, cv_folds: int, calibration_fraction: float) -> None:
        config = TrainingConfig(cv_folds=cv_folds, calibration_fraction=calibration_fraction)
        min_train = ceil(50 * cv_folds / (1 - calibration_fraction))
        min_calib = ceil(100 / calibration_fraction)
        minimum = max(min_train, min_calib)
        _validate_minimum_data(minimum, config)

    @given(
        cv_folds=st.integers(min_value=2, max_value=20),
        calibration_fraction=st.floats(
            min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False
        ),
    )
    def test_below_minimum_raises(self, cv_folds: int, calibration_fraction: float) -> None:
        from surrox.exceptions import SurrogateTrainingError

        config = TrainingConfig(cv_folds=cv_folds, calibration_fraction=calibration_fraction)
        min_train = ceil(50 * cv_folds / (1 - calibration_fraction))
        min_calib = ceil(100 / calibration_fraction)
        minimum = max(min_train, min_calib)
        if minimum > 0:
            import pytest

            with pytest.raises(SurrogateTrainingError):
                _validate_minimum_data(minimum - 1, config)
