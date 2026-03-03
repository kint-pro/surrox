from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from surrox._logging import log_duration
from surrox.exceptions import ConfigurationError
from surrox.optimizer.config import OptimizerConfig
from surrox.optimizer.result import EvaluatedPoint
from surrox.optimizer.runner import suggest_candidates
from surrox.problem.dataset import BoundDataset
from surrox.problem.scenarios import Scenario
from surrox.surrogate.config import TrainingConfig
from surrox.surrogate.manager import SurrogateManager
from surrox.surrogate.models import EnsembleMemberConfig

if TYPE_CHECKING:
    import pandas as pd

    from surrox.problem.definition import ProblemDefinition

_logger = logging.getLogger(__name__)


class ObjectivePrediction(BaseModel):
    model_config = ConfigDict(frozen=True)

    mean: float
    std: float
    lower: float
    upper: float


class Suggestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    variables: dict[str, Any]
    objectives: dict[str, ObjectivePrediction]
    extrapolation_distance: float
    is_extrapolating: bool


class SuggestionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    suggestions: tuple[Suggestion, ...]
    surrogate_quality: dict[str, float]
    ensemble_config: dict[str, tuple[EnsembleMemberConfig, ...]] = {}


def suggest(
    problem: ProblemDefinition,
    dataframe: pd.DataFrame,
    n_suggestions: int = 5,
    surrogate_config: TrainingConfig = TrainingConfig(),  # noqa: B008
    optimizer_config: OptimizerConfig = OptimizerConfig(),  # noqa: B008
    coverage: float = 0.9,
    scenario: Scenario | None = None,
    seed_points: list[dict[str, float]] | None = None,
) -> SuggestionResult:
    if n_suggestions < 1:
        raise ConfigurationError(
            f"n_suggestions must be >= 1, got {n_suggestions}"
        )
    if not (0.0 < coverage < 1.0):
        raise ConfigurationError(
            f"coverage must be between 0 and 1 exclusive, got {coverage}"
        )

    with log_duration(
        _logger, "surrox.suggest",
        n_suggestions=n_suggestions,
        n_rows=len(dataframe),
    ):
        bound_dataset = BoundDataset(problem=problem, dataframe=dataframe)

        surrogate_manager = SurrogateManager.train(
            problem=problem,
            dataset=bound_dataset,
            config=surrogate_config,
        )

        candidates = suggest_candidates(
            bound_dataset=bound_dataset,
            surrogate_manager=surrogate_manager,
            n_candidates=n_suggestions,
            config=optimizer_config,
            scenario=scenario,
            seed_points=seed_points,
        )

        suggestions = _enrich_with_uncertainty(
            candidates, surrogate_manager, problem, coverage,
        )

        surrogate_quality = {
            col: surrogate_manager.get_ensemble_r2(col)
            for col in problem.surrogate_columns
        }

        ensemble_config = {
            col: surrogate_manager.get_ensemble_member_configs(col)
            for col in problem.surrogate_columns
        }

    return SuggestionResult(
        suggestions=suggestions,
        surrogate_quality=surrogate_quality,
        ensemble_config=ensemble_config,
    )


def _enrich_with_uncertainty(
    candidates: tuple[EvaluatedPoint, ...],
    surrogate_manager: SurrogateManager,
    problem: ProblemDefinition,
    coverage: float,
) -> tuple[Suggestion, ...]:
    import pandas as pd

    if not candidates:
        return ()

    feature_names = [v.name for v in problem.variables]
    rows = []
    for candidate in candidates:
        rows.append({name: candidate.variables.get(name) for name in feature_names})

    df = pd.DataFrame(rows)
    predictions = surrogate_manager.evaluate_with_uncertainty(df, coverage=coverage)

    suggestions: list[Suggestion] = []
    for i, candidate in enumerate(candidates):
        objective_preds: dict[str, ObjectivePrediction] = {}
        for obj in problem.objectives:
            pred = predictions[obj.column]
            objective_preds[obj.name] = ObjectivePrediction(
                mean=float(pred.mean[i]),
                std=float(pred.std[i]),
                lower=float(pred.lower[i]),
                upper=float(pred.upper[i]),
            )

        suggestions.append(Suggestion(
            variables=candidate.variables,
            objectives=objective_preds,
            extrapolation_distance=candidate.extrapolation_distance,
            is_extrapolating=candidate.is_extrapolating,
        ))

    return tuple(suggestions)
