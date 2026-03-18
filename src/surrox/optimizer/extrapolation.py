from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

from surrox.problem.types import DType
from surrox.problem.variables import Variable


class ExtrapolationGate:
    def __init__(
        self,
        training_data: pd.DataFrame,
        decision_variables: tuple[Variable, ...],
        k: int,
        threshold: float,
    ) -> None:
        self._k = k
        self._threshold = threshold
        self._ordinal_maps: dict[str, dict[str, int]] = {}
        self._categorical_one_hot: dict[str, tuple[str, ...]] = {}
        self._numeric_columns: list[str] = []
        self._encoded_column_order: list[str] = []

        for var in decision_variables:
            if var.dtype in (DType.CONTINUOUS, DType.INTEGER):
                self._numeric_columns.append(var.name)
                self._encoded_column_order.append(var.name)
            elif var.dtype == DType.ORDINAL:
                categories = var.bounds.categories  # type: ignore[union-attr]
                self._ordinal_maps[var.name] = {
                    cat: i for i, cat in enumerate(categories)
                }
                self._numeric_columns.append(var.name)
                self._encoded_column_order.append(var.name)
            elif var.dtype == DType.CATEGORICAL:
                categories = var.bounds.categories  # type: ignore[union-attr]
                self._categorical_one_hot[var.name] = categories
                for cat in categories:
                    self._encoded_column_order.append(f"{var.name}_{cat}")

        encoded_training = self._encode(training_data)

        self._feature_min = encoded_training.min(axis=0)
        self._feature_max = encoded_training.max(axis=0)
        feature_range = self._feature_max - self._feature_min
        feature_range[feature_range == 0] = 1.0
        self._feature_range = feature_range

        normalized_training = (
            encoded_training - self._feature_min
        ) / self._feature_range

        n_features = normalized_training.shape[1]
        nn_algorithm = "ball_tree" if n_features > 20 else "kd_tree"
        self._nn = NearestNeighbors(n_neighbors=k, algorithm=nn_algorithm)
        self._nn.fit(normalized_training)

        distances, _ = self._nn.kneighbors(normalized_training)
        mean_distances = distances.mean(axis=1)
        self._median_knn_distance = float(np.median(mean_distances))

    def _encode(self, df: pd.DataFrame) -> NDArray[np.float64]:
        n_rows = len(df)
        n_cols = len(self._encoded_column_order)

        if n_cols == 0:
            return np.zeros((n_rows, 1), dtype=np.float64)

        result = np.zeros((n_rows, n_cols), dtype=np.float64)
        col_idx = 0

        for col in self._numeric_columns:
            if col in self._ordinal_maps:
                result[:, col_idx] = (
                    df[col].map(self._ordinal_maps[col]).to_numpy(dtype=np.float64)  # pyright: ignore[reportArgumentType]
                )
            else:
                result[:, col_idx] = df[col].to_numpy(dtype=np.float64)
            col_idx += 1

        for var_name, categories in self._categorical_one_hot.items():
            values = df[var_name]
            for cat in categories:
                result[:, col_idx] = (values == cat).astype(np.float64)
                col_idx += 1

        return result

    def _encode_and_normalize(self, df: pd.DataFrame) -> NDArray[np.float64]:
        encoded = self._encode(df)
        return (encoded - self._feature_min) / self._feature_range

    def evaluate(
        self, candidates: pd.DataFrame
    ) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
        normalized = self._encode_and_normalize(candidates)
        distances, _ = self._nn.kneighbors(normalized)
        mean_distances = distances.mean(axis=1)

        normalized_distances = mean_distances / max(self._median_knn_distance, 1e-10)
        mask = normalized_distances > self._threshold
        return mask, normalized_distances
