from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from surrox.exceptions import ConfigurationError

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

    from surrox.surrogate.ensemble import Ensemble


class ConformalCalibration:
    def __init__(
        self,
        column: str,
        ensemble: Ensemble,
        conformity_scores: NDArray[np.floating],
        default_coverage: float,
    ) -> None:
        self.column = column
        self.ensemble = ensemble
        self.conformity_scores = conformity_scores
        self._default_coverage = default_coverage

    @classmethod
    def from_calibration_data(
        cls,
        column: str,
        ensemble: Ensemble,
        X_calib: pd.DataFrame,
        y_calib: NDArray[np.floating],
        default_coverage: float,
    ) -> ConformalCalibration:
        predictions = ensemble.predict(X_calib)
        conformity_scores = np.abs(y_calib - predictions)
        return cls(
            column=column,
            ensemble=ensemble,
            conformity_scores=conformity_scores,
            default_coverage=default_coverage,
        )

    def prediction_interval(
        self, X: pd.DataFrame, coverage: float,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        if not (0 < coverage < 1):
            raise ConfigurationError("coverage must be between 0 and 1 exclusive")

        n = len(self.conformity_scores)
        quantile_level = min(coverage * (1 + 1 / n), 1.0)
        q = float(np.quantile(self.conformity_scores, quantile_level))

        predictions = self.ensemble.predict(
            X[list(self.ensemble.feature_names)],  # type: ignore[arg-type]
        )
        return predictions, predictions - q, predictions + q
