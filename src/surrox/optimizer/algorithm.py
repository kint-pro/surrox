from __future__ import annotations

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3, ReferenceDirectionSurvival
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.algorithm import Algorithm
from pymoo.core.mixed import MixedVariableGA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.util.ref_dirs import get_reference_directions

from surrox.optimizer.config import OptimizerConfig
from surrox.problem.definition import ProblemDefinition
from surrox.problem.types import DType


def _has_categorical(problem: ProblemDefinition) -> bool:
    return any(
        v.dtype in (DType.CATEGORICAL, DType.ORDINAL)
        for v in problem.decision_variables
    )


def _has_integer(problem: ProblemDefinition) -> bool:
    return any(
        v.dtype == DType.INTEGER for v in problem.decision_variables
    )


def _integer_operators() -> dict[str, object]:
    return {
        "sampling": IntegerRandomSampling(),
        "crossover": SBX(
            prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()  # pyright: ignore[reportArgumentType]
        ),
        "mutation": PM(
            prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()  # pyright: ignore[reportArgumentType]
        ),
    }


def _compute_ref_dirs(n_obj: int) -> tuple[object, int]:
    n_partitions = max(4, 16 - 2 * n_obj)
    ref_dirs = get_reference_directions(
        "das-dennis", n_obj, n_partitions=n_partitions
    )
    return ref_dirs, len(ref_dirs)


def select_algorithm(
    problem: ProblemDefinition, config: OptimizerConfig
) -> Algorithm:
    n_obj = len(problem.objectives)
    has_cat = _has_categorical(problem)
    has_int = _has_integer(problem)
    pop = config.population_size

    if has_cat:
        return _select_mixed_variable(n_obj, pop)

    if n_obj == 1:
        if has_int:
            return GA(pop_size=pop, **_integer_operators())  # type: ignore[arg-type]
        return DE(pop_size=pop)

    if n_obj <= 3:
        if has_int:
            return NSGA2(pop_size=pop, **_integer_operators())  # type: ignore[arg-type]
        return NSGA2(pop_size=pop)

    ref_dirs, n_ref = _compute_ref_dirs(n_obj)
    effective_pop = max(pop, n_ref)
    if has_int:
        return NSGA3(
            pop_size=effective_pop,
            ref_dirs=ref_dirs,
            **_integer_operators(),  # type: ignore[arg-type]
        )
    return NSGA3(pop_size=effective_pop, ref_dirs=ref_dirs)


def _select_mixed_variable(
    n_obj: int, pop_size: int
) -> Algorithm:
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
