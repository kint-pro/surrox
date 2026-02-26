from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from sklearn.base import BaseEstimator, RegressorMixin

from surrox.problem.types import MonotonicDirection
from surrox.surrogate.models import EnsembleMember


class Ensemble(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    column: str
    members: tuple[EnsembleMember, ...]
    feature_names: tuple[str, ...]
    monotonic_constraints: dict[str, MonotonicDirection]

    def predict(self, X: pd.DataFrame) -> NDArray[np.floating]:
        features = list(self.feature_names)
        predictions = np.stack([m.model.predict(X[features]) for m in self.members])
        weights = np.array([m.weight for m in self.members])
        return predictions.T @ weights

    def predict_with_std(
        self, X: pd.DataFrame
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        features = list(self.feature_names)
        predictions = np.stack([m.model.predict(X[features]) for m in self.members])
        weights = np.array([m.weight for m in self.members])
        mean = predictions.T @ weights
        std = predictions.std(axis=0)
        return mean, std


class EnsembleAdapter(RegressorMixin, BaseEstimator):
    def __init__(self, ensemble: Ensemble) -> None:
        self.ensemble = ensemble

    def fit(self, X: object, y: object) -> EnsembleAdapter:
        return self

    def predict(self, X: pd.DataFrame | NDArray) -> NDArray[np.floating]:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=list(self.ensemble.feature_names))
        return self.ensemble.predict(X)

    def __sklearn_is_fitted__(self) -> bool:
        return True
