from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from math import factorial
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd

PredictFn = Callable[["pd.DataFrame"], NDArray[np.floating]]


@dataclass(frozen=True)
class ShapleyResult:
    shap_values: NDArray[np.floating]
    base_value: float
    standard_error: NDArray[np.floating] | None


def shapley_values(
    predict: PredictFn,
    instances: pd.DataFrame,
    background: pd.DataFrame,
    feature_names: tuple[str, ...],
    exact_threshold: int,
    sampling_permutations: int,
    rng: np.random.Generator,
) -> ShapleyResult:
    base_value = float(np.mean(predict(background)))
    n_features = len(feature_names)

    if n_features <= exact_threshold:
        shap_values = np.stack(
            [
                _exact_instance(predict, instances.iloc[[i]], background, feature_names)
                for i in range(len(instances))
            ]
        )
        return ShapleyResult(
            shap_values=shap_values, base_value=base_value, standard_error=None
        )

    values = np.empty((len(instances), n_features))
    errors = np.empty((len(instances), n_features))
    for i in range(len(instances)):
        phi, stderr = _sampled_instance(
            predict,
            instances.iloc[[i]],
            background,
            feature_names,
            sampling_permutations,
            rng,
        )
        values[i] = phi
        errors[i] = stderr
    return ShapleyResult(
        shap_values=values, base_value=base_value, standard_error=errors
    )


def _coalition_value(
    predict: PredictFn,
    instance: pd.DataFrame,
    background: pd.DataFrame,
    feature_names: tuple[str, ...],
    present: NDArray[np.bool_],
) -> float:
    hybrid = background.copy()
    for index, name in enumerate(feature_names):
        if present[index]:
            hybrid[name] = instance.iloc[0][name]
    return float(np.mean(predict(hybrid)))


def _exact_instance(
    predict: PredictFn,
    instance: pd.DataFrame,
    background: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> NDArray[np.floating]:
    n_features = len(feature_names)
    coalition_cache: dict[int, float] = {}

    def value(mask: int) -> float:
        if mask not in coalition_cache:
            present = np.array([bool(mask >> bit & 1) for bit in range(n_features)])
            coalition_cache[mask] = _coalition_value(
                predict, instance, background, feature_names, present
            )
        return coalition_cache[mask]

    others = list(range(n_features))
    phi = np.zeros(n_features)
    for feature in range(n_features):
        remaining = [f for f in others if f != feature]
        for size in range(len(remaining) + 1):
            weight = (
                factorial(size)
                * factorial(n_features - size - 1)
                / factorial(n_features)
            )
            for subset in combinations(remaining, size):
                mask = 0
                for bit in subset:
                    mask |= 1 << bit
                phi[feature] += weight * (value(mask | 1 << feature) - value(mask))
    return phi


def _sampled_instance(
    predict: PredictFn,
    instance: pd.DataFrame,
    background: pd.DataFrame,
    feature_names: tuple[str, ...],
    sampling_permutations: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    n_features = len(feature_names)
    contributions = np.empty((sampling_permutations * 2, n_features))
    row = 0
    for _ in range(sampling_permutations):
        permutation = rng.permutation(n_features)
        for order in (permutation, permutation[::-1]):
            present = np.zeros(n_features, dtype=bool)
            previous = _coalition_value(
                predict, instance, background, feature_names, present
            )
            for feature in order:
                present[feature] = True
                current = _coalition_value(
                    predict, instance, background, feature_names, present
                )
                contributions[row, feature] = current - previous
                previous = current
            row += 1
    phi = contributions.mean(axis=0)
    standard_error = contributions.std(axis=0, ddof=1) / np.sqrt(contributions.shape[0])
    return phi, standard_error
