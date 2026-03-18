from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.stats.qmc import Sobol
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from surrox.optimizer.config import TuRBOConfig

_logger = logging.getLogger(__name__)

_IMPROVEMENT_THRESHOLD = 1e-3


class ObjectiveFunction(Protocol):
    def __call__(self, X: NDArray[np.float64]) -> NDArray[np.float64]: ...


@dataclass
class TrustRegionState:
    center: NDArray[np.float64]
    length: float
    best_value: float = np.inf
    success_counter: int = 0
    failure_counter: int = 0


@dataclass
class TuRBOResult:
    best_x: NDArray[np.float64]
    best_value: float
    X_evaluated: NDArray[np.float64]
    y_evaluated: NDArray[np.float64]
    n_evaluations: int


def turbo_minimize(
    objective_fn: ObjectiveFunction,
    n_dims: int,
    config: TuRBOConfig,
    seed: int,
) -> TuRBOResult:
    n_initial = config.resolve_n_initial(n_dims)
    failure_tolerance = config.resolve_failure_tolerance(n_dims)
    rng = np.random.default_rng(seed)

    X_all = np.empty((0, n_dims), dtype=np.float64)
    y_all = np.empty(0, dtype=np.float64)
    best_x_global: NDArray[np.float64] | None = None
    best_value_global = np.inf

    for restart in range(config.n_restarts + 1):
        remaining = config.max_evaluations - len(y_all)
        if remaining < 1:
            break

        n_init = min(n_initial, remaining)
        X_init = _sobol_sample(n_init, n_dims, rng)
        y_init = objective_fn(X_init)

        X_local = X_init.copy()
        y_local = y_init.copy()
        X_all = np.vstack([X_all, X_init])
        y_all = np.concatenate([y_all, y_init])

        best_idx = int(np.argmin(y_local))
        state = TrustRegionState(
            center=X_local[best_idx].copy(),
            length=config.length_init,
            best_value=float(y_local[best_idx]),
        )

        while len(y_all) < config.max_evaluations:
            if state.length < config.length_min:
                _logger.info(
                    "trust region length below minimum, restarting",
                    extra={"restart": restart, "length": state.length},
                )
                break

            gp = _fit_local_gp(X_local, y_local, n_dims, rng)
            lengthscales = _extract_lengthscales(gp, n_dims)

            n_candidates = min(config.batch_size, config.max_evaluations - len(y_all))
            if n_candidates < 1:
                break

            X_cand = _thompson_sample(
                gp, state, lengthscales, n_dims, n_candidates, rng,
            )
            y_cand = objective_fn(X_cand)

            X_local = np.vstack([X_local, X_cand])
            y_local = np.concatenate([y_local, y_cand])
            X_all = np.vstack([X_all, X_cand])
            y_all = np.concatenate([y_all, y_cand])

            best_new = float(np.min(y_cand))
            _update_state(state, best_new, config, failure_tolerance)

        local_best_idx = int(np.argmin(y_local))
        if y_local[local_best_idx] < best_value_global:
            best_value_global = float(y_local[local_best_idx])
            best_x_global = X_local[local_best_idx].copy()

    if best_x_global is None:
        best_idx = int(np.argmin(y_all))
        best_x_global = X_all[best_idx].copy()
        best_value_global = float(y_all[best_idx])

    return TuRBOResult(
        best_x=best_x_global,
        best_value=best_value_global,
        X_evaluated=X_all,
        y_evaluated=y_all,
        n_evaluations=len(y_all),
    )


def _sobol_sample(
    n: int, d: int, rng: np.random.Generator,
) -> NDArray[np.float64]:
    sampler = Sobol(d, scramble=True, seed=int(rng.integers(0, 2**31)))
    raw = sampler.random(n)
    return np.clip(raw, 0.0, 1.0)


def _fit_local_gp(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    n_dims: int,
    rng: np.random.Generator,
) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0) * Matern(
        nu=2.5,
        length_scale=np.ones(n_dims),
        length_scale_bounds=(1e-3, 1e3),
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=int(rng.integers(0, 2**31)),
    )
    gp.fit(X, y)
    return gp


def _extract_lengthscales(
    gp: GaussianProcessRegressor, n_dims: int,
) -> NDArray[np.float64]:
    kernel = gp.kernel_
    for part in [kernel.k2 if hasattr(kernel, "k2") else kernel]:
        if hasattr(part, "length_scale"):
            ls = np.atleast_1d(part.length_scale)
            if len(ls) == n_dims:
                return ls.astype(np.float64)
    return np.ones(n_dims, dtype=np.float64)


def _compute_tr_bounds(
    state: TrustRegionState,
    lengthscales: NDArray[np.float64],
    n_dims: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    weights = lengthscales / np.mean(lengthscales)
    weights = weights / np.prod(weights) ** (1.0 / n_dims)

    half_widths = weights * state.length / 2.0
    tr_lb = np.clip(state.center - half_widths, 0.0, 1.0)
    tr_ub = np.clip(state.center + half_widths, 0.0, 1.0)

    return tr_lb, tr_ub


def _thompson_sample(
    gp: GaussianProcessRegressor,
    state: TrustRegionState,
    lengthscales: NDArray[np.float64],
    n_dims: int,
    n_candidates: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    tr_lb, tr_ub = _compute_tr_bounds(state, lengthscales, n_dims)

    n_sobol = max(min(5000, 100 * n_dims), 200)
    X_sobol = _sobol_sample(n_sobol, n_dims, rng)
    X_cand = tr_lb + (tr_ub - tr_lb) * X_sobol

    perturb_prob = min(20.0 / n_dims, 1.0)
    if perturb_prob < 1.0:
        mask = rng.random((n_sobol, n_dims)) < perturb_prob
        mask |= np.eye(n_sobol, n_dims, dtype=bool)[:n_sobol, :]
        center_tiled = np.tile(state.center, (n_sobol, 1))
        X_cand = np.where(mask, X_cand, center_tiled)

    rs = int(rng.integers(0, 2**31))
    y_samples = gp.sample_y(X_cand, n_samples=1, random_state=rs)
    y_samples = y_samples.ravel()

    best_indices = np.argsort(y_samples)[:n_candidates]
    return X_cand[best_indices]


def _update_state(
    state: TrustRegionState,
    best_new_value: float,
    config: TuRBOConfig,
    failure_tolerance: int,
) -> None:
    threshold = state.best_value - _IMPROVEMENT_THRESHOLD * abs(state.best_value)

    if best_new_value < threshold:
        state.success_counter += 1
        state.failure_counter = 0
        state.best_value = best_new_value
        state.center = state.center

        if state.success_counter >= config.success_tolerance:
            state.length = min(2.0 * state.length, config.length_max)
            state.success_counter = 0
            _logger.debug(
                "trust region expanded", extra={"length": state.length},
            )
    else:
        state.failure_counter += 1
        state.success_counter = 0

        if state.failure_counter >= failure_tolerance:
            state.length = state.length / 2.0
            state.failure_counter = 0
            _logger.debug(
                "trust region shrunk", extra={"length": state.length},
            )

    if best_new_value < state.best_value:
        state.best_value = best_new_value
