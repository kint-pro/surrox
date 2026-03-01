from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from surrox.problem.types import MonotonicDirection
from surrox.surrogate.models import EnsembleMember


class Ensemble(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    column: str
    members: tuple[EnsembleMember, ...]
    feature_names: tuple[str, ...]
    monotonic_constraints: dict[str, MonotonicDirection]
    category_mappings: dict[str, list[str]] = {}
    y_min: float = -np.inf
    y_max: float = np.inf

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X[list(self.feature_names)]
        if self.category_mappings:
            df = df.copy()
            for col, categories in self.category_mappings.items():
                if col in df.columns and not isinstance(df[col].dtype, pd.CategoricalDtype):
                    df[col] = pd.Categorical(df[col], categories=categories)
        return df

    def predict(self, X: pd.DataFrame) -> NDArray[np.floating]:
        df = self._prepare_features(X)
        predictions = np.stack([m.model.predict(df) for m in self.members])
        weights = np.array([m.weight for m in self.members])
        mean = predictions.T @ weights
        return np.clip(mean, self.y_min, self.y_max)

    def predict_with_std(
        self, X: pd.DataFrame,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        df = self._prepare_features(X)
        predictions = np.stack([m.model.predict(df) for m in self.members])
        weights = np.array([m.weight for m in self.members])
        mean = predictions.T @ weights
        std = predictions.std(axis=0)
        return np.clip(mean, self.y_min, self.y_max), std
