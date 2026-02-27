import pytest
from pydantic import ValidationError

from surrox.exceptions import ConfigurationError
from surrox.surrogate.config import TrainingConfig
from surrox.surrogate.families import XGBoostFamily


class TestTrainingConfig:
    def test_default_construction(self) -> None:
        config = TrainingConfig()
        assert config.n_trials == 50
        assert config.cv_folds == 5
        assert config.study_timeout_s == 300
        assert config.min_r2 == 0.7
        assert len(config.estimator_families) == 2

    def test_n_trials_must_be_at_least_1(self) -> None:
        with pytest.raises(ConfigurationError, match="n_trials"):
            TrainingConfig(n_trials=0)

    def test_cv_folds_must_be_at_least_2(self) -> None:
        with pytest.raises(ConfigurationError, match="cv_folds"):
            TrainingConfig(cv_folds=1)

    def test_calibration_fraction_must_be_between_0_and_1(self) -> None:
        with pytest.raises(ConfigurationError, match="calibration_fraction"):
            TrainingConfig(calibration_fraction=0.0)
        with pytest.raises(ConfigurationError, match="calibration_fraction"):
            TrainingConfig(calibration_fraction=1.0)

    def test_ensemble_size_must_be_at_least_1(self) -> None:
        with pytest.raises(ConfigurationError, match="ensemble_size"):
            TrainingConfig(ensemble_size=0)

    def test_diversity_threshold_bounds(self) -> None:
        with pytest.raises(ConfigurationError, match="diversity_threshold"):
            TrainingConfig(diversity_threshold=0.0)
        TrainingConfig(diversity_threshold=1.0)

    def test_softmax_temperature_must_be_positive(self) -> None:
        with pytest.raises(ConfigurationError, match="softmax_temperature"):
            TrainingConfig(softmax_temperature=0.0)

    def test_default_coverage_must_be_between_0_and_1(self) -> None:
        with pytest.raises(ConfigurationError, match="default_coverage"):
            TrainingConfig(default_coverage=0.0)
        with pytest.raises(ConfigurationError, match="default_coverage"):
            TrainingConfig(default_coverage=1.0)

    def test_estimator_families_must_not_be_empty(self) -> None:
        with pytest.raises(ConfigurationError, match="estimator_families"):
            TrainingConfig(estimator_families=())

    def test_estimator_family_names_must_be_unique(self) -> None:
        with pytest.raises(ConfigurationError, match="unique"):
            TrainingConfig(estimator_families=(XGBoostFamily(), XGBoostFamily()))

    def test_n_threads_must_be_at_least_1_if_set(self) -> None:
        with pytest.raises(ConfigurationError, match="n_threads"):
            TrainingConfig(n_threads=0)
        TrainingConfig(n_threads=None)

    def test_study_timeout_must_be_at_least_1(self) -> None:
        with pytest.raises(ConfigurationError, match="study_timeout_s"):
            TrainingConfig(study_timeout_s=0)

    def test_min_r2_bounds(self) -> None:
        with pytest.raises(ConfigurationError, match="min_r2"):
            TrainingConfig(min_r2=0.0)
        with pytest.raises(ConfigurationError, match="min_r2"):
            TrainingConfig(min_r2=1.0)
        TrainingConfig(min_r2=None)

    def test_frozen(self) -> None:
        config = TrainingConfig()
        with pytest.raises(ValidationError):
            config.n_trials = 100
