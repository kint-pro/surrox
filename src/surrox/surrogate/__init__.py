from surrox.surrogate.config import TrainingConfig
from surrox.surrogate.conformal import ConformalCalibration
from surrox.surrogate.ensemble import Ensemble
from surrox.surrogate.families import LightGBMFamily, XGBoostFamily
from surrox.surrogate.manager import SurrogateManager, SurrogateResult
from surrox.surrogate.models import (
    EnsembleMember,
    EnsembleMemberConfig,
    FoldMetrics,
    SurrogatePrediction,
    TrialRecord,
)
from surrox.surrogate.pipeline import fast_train_surrogate
from surrox.surrogate.protocol import EstimatorFamily

__all__ = [
    "ConformalCalibration",
    "Ensemble",
    "EnsembleMember",
    "EnsembleMemberConfig",
    "EstimatorFamily",
    "FoldMetrics",
    "LightGBMFamily",
    "SurrogateManager",
    "SurrogatePrediction",
    "SurrogateResult",
    "TrainingConfig",
    "TrialRecord",
    "XGBoostFamily",
    "fast_train_surrogate",
]


def __dir__() -> list[str]:
    return __all__
