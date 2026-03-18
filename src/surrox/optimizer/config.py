from enum import StrEnum
from math import ceil
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from surrox.exceptions import ConfigurationError


class Strategy(StrEnum):
    GLOBAL_SURROGATE = "global_surrogate"
    TRUST_REGION = "trust_region"


class TuRBOConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_initial: int | None = None
    max_evaluations: int = 500
    batch_size: int = 1
    length_init: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    success_tolerance: int = 3
    failure_tolerance: int | None = None
    n_restarts: int = 3

    @model_validator(mode="after")
    def _validate_turbo(self) -> "TuRBOConfig":
        if self.n_initial is not None and self.n_initial < 1:
            raise ConfigurationError("n_initial must be >= 1 if set")
        if self.max_evaluations < 10:
            raise ConfigurationError("max_evaluations must be >= 10")
        if self.batch_size < 1:
            raise ConfigurationError("batch_size must be >= 1")
        if not (0 < self.length_min < self.length_init < self.length_max):
            raise ConfigurationError(
                "must satisfy 0 < length_min < length_init < length_max"
            )
        if self.success_tolerance < 1:
            raise ConfigurationError("success_tolerance must be >= 1")
        if self.failure_tolerance is not None and self.failure_tolerance < 1:
            raise ConfigurationError("failure_tolerance must be >= 1 if set")
        if self.n_restarts < 0:
            raise ConfigurationError("n_restarts must be >= 0")
        return self

    def resolve_n_initial(self, n_dims: int) -> int:
        return self.n_initial if self.n_initial is not None else 2 * n_dims

    def resolve_failure_tolerance(self, n_dims: int) -> int:
        if self.failure_tolerance is not None:
            return self.failure_tolerance
        return ceil(n_dims / self.batch_size)


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: Strategy | None = None
    dim_threshold: int = 15
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
    trust_region_center: dict[str, float] | None = None
    turbo: TuRBOConfig = TuRBOConfig()

    @model_validator(mode="after")
    def _validate_config(self) -> "OptimizerConfig":
        if self.dim_threshold < 1:
            raise ConfigurationError("dim_threshold must be >= 1")
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
        if self.trust_region_center is not None and self.trust_region_margin is None:
            raise ConfigurationError(
                "trust_region_center requires trust_region_margin to be set"
            )
        return self

    def resolve_strategy(self, n_decision_variables: int) -> Strategy:
        if self.strategy is not None:
            return self.strategy
        if n_decision_variables > self.dim_threshold:
            return Strategy.TRUST_REGION
        return Strategy.GLOBAL_SURROGATE
