from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from surrox.exceptions import ConfigurationError


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    population_size: int = 100
    n_generations: int = 200
    seed: int = 42
    extrapolation_k: int = 5
    extrapolation_threshold: float = 2.0
    constraint_confidence: float = 0.95
    acquisition: Literal["direct", "pessimistic"] = "pessimistic"
    pessimistic_beta: float = 1.0
    min_beta_fraction: float = 0.1
    trust_region_margin: float | None = None

    @model_validator(mode="after")
    def _validate_config(self) -> "OptimizerConfig":
        if self.population_size < 10:
            raise ConfigurationError("population_size must be >= 10")
        if self.n_generations < 1:
            raise ConfigurationError("n_generations must be >= 1")
        if self.extrapolation_k < 1:
            raise ConfigurationError("extrapolation_k must be >= 1")
        if self.extrapolation_threshold <= 0:
            raise ConfigurationError("extrapolation_threshold must be > 0")
        if not (0 < self.constraint_confidence < 1):
            raise ConfigurationError(
                "constraint_confidence must be between 0 and 1 exclusive"
            )
        if self.pessimistic_beta <= 0:
            raise ConfigurationError("pessimistic_beta must be > 0")
        if not (0 <= self.min_beta_fraction <= 1):
            raise ConfigurationError("min_beta_fraction must be between 0 and 1")
        if self.trust_region_margin is not None and self.trust_region_margin < 0:
            raise ConfigurationError("trust_region_margin must be >= 0 if set")
        return self
