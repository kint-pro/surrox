from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import optuna
    from sklearn.base import BaseEstimator

    from surrox.problem.types import MonotonicDirection


@runtime_checkable
class EstimatorFamily(Protocol):
    @property
    def name(self) -> str: ...

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, Any]: ...

    def build_model(
        self,
        hyperparameters: dict[str, Any],
        monotonic_constraints: Any,
        random_seed: int,
        n_threads: int | None,
    ) -> BaseEstimator: ...

    def map_monotonic_constraints(
        self,
        constraints: dict[str, MonotonicDirection],
        feature_names: list[str],
        categorical_features: set[str],
    ) -> Any: ...
