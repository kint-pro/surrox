import pytest
from pydantic import ValidationError

from surrox.exceptions import ConfigurationError
from surrox.optimizer.config import OptimizerConfig


class TestOptimizerConfig:
    def test_defaults(self) -> None:
        config = OptimizerConfig()
        assert config.population_size == 100
        assert config.n_generations == 200
        assert config.seed == 42
        assert config.extrapolation_k == 5
        assert config.extrapolation_threshold == 2.0
        assert config.constraint_confidence == 0.95

    def test_custom_values(self) -> None:
        config = OptimizerConfig(
            population_size=50, n_generations=100, seed=123,
            extrapolation_k=3, extrapolation_threshold=1.5,
            constraint_confidence=0.95,
        )
        assert config.population_size == 50

    def test_population_size_too_small(self) -> None:
        with pytest.raises(ConfigurationError, match="population_size must be >= 10"):
            OptimizerConfig(population_size=9)

    def test_n_generations_zero(self) -> None:
        with pytest.raises(ConfigurationError, match="n_generations must be >= 1"):
            OptimizerConfig(n_generations=0)

    def test_extrapolation_k_zero(self) -> None:
        with pytest.raises(ConfigurationError, match="extrapolation_k must be >= 1"):
            OptimizerConfig(extrapolation_k=0)

    def test_extrapolation_threshold_zero(self) -> None:
        with pytest.raises(ConfigurationError, match="extrapolation_threshold must be > 0"):
            OptimizerConfig(extrapolation_threshold=0.0)

    def test_extrapolation_threshold_negative(self) -> None:
        with pytest.raises(ConfigurationError, match="extrapolation_threshold must be > 0"):
            OptimizerConfig(extrapolation_threshold=-1.0)

    def test_constraint_confidence_zero(self) -> None:
        with pytest.raises(ConfigurationError, match="constraint_confidence"):
            OptimizerConfig(constraint_confidence=0.0)

    def test_constraint_confidence_one(self) -> None:
        with pytest.raises(ConfigurationError, match="constraint_confidence"):
            OptimizerConfig(constraint_confidence=1.0)

    def test_is_frozen(self) -> None:
        config = OptimizerConfig()
        with pytest.raises(ValidationError):
            config.population_size = 50  # type: ignore[misc]
