from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from sklearn.base import BaseEstimator


class FoldMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    fold: int
    r2: float
    rmse: float
    mae: float
    training_time_s: float
    inference_time_ms: float


class TrialRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    trial_number: int
    estimator_family: str
    hyperparameters: dict[str, object]
    fold_metrics: tuple[FoldMetrics, ...]
    mean_r2: float
    mean_rmse: float
    mean_mae: float
    mean_training_time_s: float
    mean_inference_time_ms: float
    status: str


class EnsembleMember(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    trial_number: int
    estimator_family: str
    model: BaseEstimator
    weight: float
    cv_rmse: float


class SurrogatePrediction(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    mean: NDArray[np.floating]
    std: NDArray[np.floating]
    lower: NDArray[np.floating]
    upper: NDArray[np.floating]
