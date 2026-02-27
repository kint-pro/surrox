from pydantic import BaseModel, ConfigDict, model_validator

from surrox.exceptions import ConfigurationError


class OptimizerConfig(BaseModel):
    """Configuration for the pymoo-based optimizer.

    Attributes:
        population_size: Population size for the evolutionary algorithm.
        n_generations: Number of generations to evolve.
        seed: Random seed for reproducibility.
        extrapolation_k: Number of nearest neighbors for extrapolation detection.
        extrapolation_threshold: Distance threshold beyond which a point is flagged as extrapolating.
        constraint_confidence: Conformal prediction confidence level for constraint evaluation.
    """

    model_config = ConfigDict(frozen=True)

    population_size: int = 100
    n_generations: int = 200
    seed: int = 42
    extrapolation_k: int = 5
    extrapolation_threshold: float = 2.0
    constraint_confidence: float = 0.95

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
        return self
