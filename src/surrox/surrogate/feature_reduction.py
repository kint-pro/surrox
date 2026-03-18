from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.decomposition import PCA

_logger = logging.getLogger(__name__)

_DEFAULT_IMPORTANCE_THRESHOLD = 0.01
_DEFAULT_CORRELATION_THRESHOLD = 0.9
_MIN_FEATURES_FOR_SCREENING = 10
_MIN_SAMPLES_FOR_SCREENING = 100


@dataclass(frozen=True)
class FeatureGroup:
    original_names: tuple[str, ...]
    combined_name: str
    center: NDArray[np.float64]
    scale: NDArray[np.float64]
    components: NDArray[np.float64]


@dataclass(frozen=True)
class FeatureReduction:
    selected: tuple[str, ...]
    dropped: tuple[str, ...]
    groups: tuple[FeatureGroup, ...]

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.selected + tuple(g.combined_name for g in self.groups)

    @property
    def is_identity(self) -> bool:
        return len(self.dropped) == 0 and len(self.groups) == 0

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.is_identity:
            return X[list(self.selected)]

        parts: dict[str, NDArray] = {}
        for name in self.selected:
            parts[name] = X[name].to_numpy(dtype=np.float64)

        for group in self.groups:
            cols = X[list(group.original_names)].to_numpy(dtype=np.float64)
            centered = (cols - group.center) / group.scale
            pc = centered @ group.components.T
            parts[group.combined_name] = pc[:, 0]

        return pd.DataFrame(parts)


def analyze_features(
    X: pd.DataFrame,
    y: NDArray[np.float64],
    feature_names: list[str],
    protected_features: set[str],
    importance_threshold: float = _DEFAULT_IMPORTANCE_THRESHOLD,
    correlation_threshold: float = _DEFAULT_CORRELATION_THRESHOLD,
) -> FeatureReduction:
    n_samples, n_features = len(X), len(feature_names)

    if n_features < _MIN_FEATURES_FOR_SCREENING:
        return _identity_reduction(feature_names)

    if n_samples < _MIN_SAMPLES_FOR_SCREENING:
        return _identity_reduction(feature_names)

    numeric_features = _get_numeric_features(X, feature_names)
    if len(numeric_features) < _MIN_FEATURES_FOR_SCREENING:
        return _identity_reduction(feature_names)

    importances = _compute_importances(X[numeric_features], y)
    kept, dropped = _screen_by_importance(
        numeric_features, importances, protected_features, importance_threshold,
    )

    non_numeric = [f for f in feature_names if f not in numeric_features]
    kept = list(kept) + non_numeric

    groups = _find_correlated_groups(
        X, kept, protected_features, correlation_threshold,
    )
    feature_groups = _build_pca_groups(X, groups)

    grouped_features = set()
    for group in feature_groups:
        grouped_features.update(group.original_names)

    selected = tuple(name for name in kept if name not in grouped_features)

    reduction = FeatureReduction(
        selected=selected,
        dropped=tuple(dropped),
        groups=tuple(feature_groups),
    )

    if not reduction.is_identity:
        _logger.info(
            "feature reduction applied",
            extra={
                "original_features": n_features,
                "selected": len(reduction.selected),
                "dropped": len(reduction.dropped),
                "groups": len(reduction.groups),
                "output_features": len(reduction.output_names),
            },
        )

    return reduction


def _identity_reduction(feature_names: list[str]) -> FeatureReduction:
    return FeatureReduction(
        selected=tuple(feature_names),
        dropped=(),
        groups=(),
    )


def _get_numeric_features(
    X: pd.DataFrame, feature_names: list[str],
) -> list[str]:
    return [
        f for f in feature_names
        if pd.api.types.is_numeric_dtype(X[f])
    ]


def _compute_importances(
    X: pd.DataFrame, y: NDArray[np.float64],
) -> dict[str, float]:
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)

    importances = model.feature_importances_
    return {
        name: float(imp)
        for name, imp in zip(X.columns, importances)
    }


def _screen_by_importance(
    feature_names: list[str],
    importances: dict[str, float],
    protected_features: set[str],
    threshold: float,
) -> tuple[list[str], list[str]]:
    max_imp = max(importances.values())
    cutoff = max_imp * threshold

    kept: list[str] = []
    dropped: list[str] = []

    for name in feature_names:
        if name in protected_features:
            kept.append(name)
        elif importances.get(name, 0.0) >= cutoff:
            kept.append(name)
        else:
            dropped.append(name)

    return kept, dropped


def _find_correlated_groups(
    X: pd.DataFrame,
    feature_names: list[str],
    protected_features: set[str],
    threshold: float,
) -> list[list[str]]:
    numeric = [
        f for f in feature_names
        if f not in protected_features and pd.api.types.is_numeric_dtype(X[f])
    ]

    if len(numeric) < 2:
        return []

    corr = X[numeric].corr().abs()
    visited: set[str] = set()
    groups: list[list[str]] = []

    for feat in numeric:
        if feat in visited:
            continue

        correlated = [
            other for other in numeric
            if other != feat
            and other not in visited
            and corr.loc[feat, other] >= threshold
        ]

        if correlated:
            group = [feat] + correlated
            groups.append(group)
            visited.update(group)

    return groups


def _build_pca_groups(
    X: pd.DataFrame, groups: list[list[str]],
) -> list[FeatureGroup]:
    result: list[FeatureGroup] = []

    for group_names in groups:
        data = X[group_names].to_numpy(dtype=np.float64)
        center = data.mean(axis=0)
        scale = data.std(axis=0)
        scale[scale == 0] = 1.0
        standardized = (data - center) / scale

        pca = PCA(n_components=1)
        pca.fit(standardized)

        combined_name = f"pc_{'_'.join(sorted(group_names))}"
        result.append(FeatureGroup(
            original_names=tuple(group_names),
            combined_name=combined_name,
            center=center,
            scale=scale,
            components=pca.components_,
        ))

    return result
