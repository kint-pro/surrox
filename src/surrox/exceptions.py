class SurroxError(Exception):
    """Base exception for all surrox errors."""


class ProblemDefinitionError(SurroxError):
    """Raised when a problem definition is invalid."""


class ConfigurationError(SurroxError):
    """Raised when a configuration value is invalid."""


class SurrogateTrainingError(SurroxError):
    """Raised when surrogate training fails."""


class OptimizationError(SurroxError):
    """Raised when optimization fails."""


class AnalysisError(SurroxError):
    """Raised when an analysis operation fails."""
