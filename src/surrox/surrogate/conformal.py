from __future__ import annotations

from typing import TYPE_CHECKING

from mapie.regression import SplitConformalRegressor

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from numpy.typing import NDArray

    from surrox.surrogate.ensemble import EnsembleAdapter


class ConformalCalibration:
    def __init__(
        self,
        column: str,
        adapter: EnsembleAdapter,
        X_calib: NDArray[np.floating],
        y_calib: NDArray[np.floating],
        default_coverage: float,
    ) -> None:
        self.column = column
        self.adapter = adapter
        self.X_calib = X_calib
        self.y_calib = y_calib
        self._default_coverage = default_coverage
        self._default_scr = self._build_scr(default_coverage)

    def prediction_interval(
        self, X: pd.DataFrame, coverage: float
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        if not (0 < coverage < 1):
            raise ValueError("coverage must be between 0 and 1 exclusive")

        if coverage == self._default_coverage:
            scr = self._default_scr
        else:
            scr = self._build_scr(coverage)
        X_features = X[list(self.adapter.ensemble.feature_names)]
        y_pred, y_intervals = scr.predict_interval(X_features)
        return y_pred, y_intervals[:, 0, 0], y_intervals[:, 1, 0]

    def _build_scr(self, coverage: float) -> SplitConformalRegressor:
        scr = SplitConformalRegressor(
            estimator=self.adapter,
            confidence_level=coverage,
            prefit=True,
        )
        scr.conformalize(self.X_calib, self.y_calib)
        return scr
