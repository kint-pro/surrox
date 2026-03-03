from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3, ReferenceDirectionSurvival
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.algorithm import Algorithm
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.population import Population
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import FloatRandomSampling, IntegerRandomSampling
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.util.ref_dirs import get_reference_directions

from surrox.optimizer.config import OptimizerConfig
from surrox.problem.definition import ProblemDefinition
from surrox.problem.types import DType


class SeedAugmentedSampling(Sampling):
    def __init__(self, base_sampling: Sampling, seed_X: NDArray[np.float64]) -> None:
        super().__init__()
        self._base = base_sampling
        self._seed_X = seed_X

    def _do(self, problem: object, n_samples: int, **kwargs: object) -> np.ndarray:
        n_seed = min(len(self._seed_X), n_samples)
        n_random = n_samples - n_seed

        if n_random > 0:
            random_pop = self._base(problem, n_random, **kwargs)
            seed_pop = Population.new("X", self._seed_X[:n_seed])
            merged = Population.merge(seed_pop, random_pop)
            return merged.get("X")

        return self._seed_X[:n_seed]


def _has_categorical(problem: ProblemDefinition) -> bool:
    return any(
        v.dtype in (DType.CATEGORICAL, DType.ORDINAL)
        for v in problem.decision_variables
    )


def _has_integer(problem: ProblemDefinition) -> bool:
    return any(v.dtype == DType.INTEGER for v in problem.decision_variables)


def _integer_operators(
    seed_X: NDArray[np.float64] | None = None,
) -> dict[str, object]:
    sampling: Sampling = IntegerRandomSampling()
    if seed_X is not None:
        sampling = SeedAugmentedSampling(sampling, seed_X)
    return {
        "sampling": sampling,
        "crossover": SBX(
            prob=1.0,
            eta=3.0,
            vtype=float,
            repair=RoundingRepair(),  # pyright: ignore[reportArgumentType]
        ),
        "mutation": PM(
            prob=1.0,
            eta=3.0,
            vtype=float,
            repair=RoundingRepair(),  # pyright: ignore[reportArgumentType]
        ),
    }


def _float_sampling(
    seed_X: NDArray[np.float64] | None = None,
) -> Sampling | None:
    if seed_X is None:
        return None
    return SeedAugmentedSampling(FloatRandomSampling(), seed_X)


def _compute_ref_dirs(n_obj: int) -> tuple[object, int]:
    n_partitions = max(4, 16 - 2 * n_obj)
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
    return ref_dirs, len(ref_dirs)


def select_algorithm(
    problem: ProblemDefinition,
    config: OptimizerConfig,
    seed_X: NDArray[np.float64] | None = None,
) -> Algorithm:
    n_obj = len(problem.objectives)
    has_cat = _has_categorical(problem)
    has_int = _has_integer(problem)
    pop = config.population_size

    if has_cat:
        return _select_mixed_variable(n_obj, pop)

    sampling = _float_sampling(seed_X)

    if n_obj == 1:
        if has_int:
            return GA(pop_size=pop, **_integer_operators(seed_X))  # type: ignore[arg-type]
        kwargs: dict[str, object] = {"pop_size": pop}
        if sampling is not None:
            kwargs["sampling"] = sampling
        return DE(**kwargs)  # type: ignore[arg-type]

    if n_obj <= 3:
        if has_int:
            return NSGA2(pop_size=pop, **_integer_operators(seed_X))  # type: ignore[arg-type]
        kwargs = {"pop_size": pop}
        if sampling is not None:
            kwargs["sampling"] = sampling
        return NSGA2(**kwargs)  # type: ignore[arg-type]

    ref_dirs, n_ref = _compute_ref_dirs(n_obj)
    effective_pop = max(pop, n_ref)
    if has_int:
        return NSGA3(
            pop_size=effective_pop,
            ref_dirs=ref_dirs,
            **_integer_operators(seed_X),  # type: ignore[arg-type]
        )
    kwargs = {"pop_size": effective_pop, "ref_dirs": ref_dirs}
    if sampling is not None:
        kwargs["sampling"] = sampling
    return NSGA3(**kwargs)  # type: ignore[arg-type]


def _select_mixed_variable(n_obj: int, pop_size: int) -> Algorithm:
    if n_obj == 1:
        return MixedVariableGA(pop_size=pop_size)

    if n_obj <= 3:
        return MixedVariableGA(
            pop_size=pop_size,
            survival=RankAndCrowding(),  # pyright: ignore[reportArgumentType]
        )

    ref_dirs, n_ref = _compute_ref_dirs(n_obj)
    effective_pop = max(pop_size, n_ref)
    return MixedVariableGA(
        pop_size=effective_pop,
        survival=ReferenceDirectionSurvival(ref_dirs),  # pyright: ignore[reportArgumentType]
    )
