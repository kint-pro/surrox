from surrox.surrogate.families.tabicl import TabICLFamily  # noqa: I001, E402 — must load torch before lightgbm/xgboost (macOS OpenMP conflict)
from surrox.surrogate.families.gaussian_process import GaussianProcessFamily
from surrox.surrogate.families.lightgbm import LightGBMFamily
from surrox.surrogate.families.xgboost import XGBoostFamily

__all__ = [
    "GaussianProcessFamily",
    "LightGBMFamily",
    "TabICLFamily",
    "XGBoostFamily",
]


def __dir__() -> list[str]:
    return __all__
