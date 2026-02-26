from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from surrox.surrogate.conformal import ConformalCalibration
from surrox.surrogate.ensemble import Ensemble
from surrox.surrogate.models import SurrogatePrediction, TrialRecord

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

    from surrox.problem.dataset import BoundDataset
    from surrox.problem.definition import ProblemDefinition
    from surrox.surrogate.config import TrainingConfig


class SurrogateResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    column: str
    ensemble: Ensemble
    conformal: ConformalCalibration
    trial_history: tuple[TrialRecord, ...]


class SurrogateManager:
    def __init__(
        self,
        problem: ProblemDefinition,
        config: TrainingConfig,
        surrogates: dict[str, SurrogateResult],
    ) -> None:
        self._problem = problem
        self._config = config
        self._surrogates = surrogates

    @classmethod
    def train(
        cls,
        problem: ProblemDefinition,
        dataset: BoundDataset,
        config: TrainingConfig,
    ) -> SurrogateManager:
        from surrox.surrogate.pipeline import train_surrogate

        surrogates: dict[str, SurrogateResult] = {}
        for column in problem.surrogate_columns:
            surrogates[column] = train_surrogate(
                problem=problem,
                dataset_df=dataset.dataframe,
                config=config,
                column=column,
            )

        return cls(problem=problem, config=config, surrogates=surrogates)

    @property
    def problem(self) -> ProblemDefinition:
        return self._problem

    @property
    def config(self) -> TrainingConfig:
        return self._config

    def evaluate(self, X: pd.DataFrame) -> dict[str, NDArray]:
        return {
            column: result.ensemble.predict(X)
            for column, result in self._surrogates.items()
        }

    def evaluate_with_uncertainty(
        self, X: pd.DataFrame, coverage: float | None = None
    ) -> dict[str, SurrogatePrediction]:
        if coverage is None:
            coverage = self._config.default_coverage

        results: dict[str, SurrogatePrediction] = {}
        for column, result in self._surrogates.items():
            mean, std = result.ensemble.predict_with_std(X)
            pred_mean, lower, upper = result.conformal.prediction_interval(X, coverage)
            results[column] = SurrogatePrediction(
                mean=mean, std=std, lower=lower, upper=upper
            )
        return results

    def get_ensemble(self, column: str) -> Ensemble:
        return self._surrogates[column].ensemble

    def get_trial_history(self, column: str) -> tuple[TrialRecord, ...]:
        return self._surrogates[column].trial_history
