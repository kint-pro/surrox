from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, PlainSerializer, PlainValidator

from surrox.types import NumpyArray


def _validate_marginal_rates(
    v: Any,
) -> dict[tuple[str, str], Any]:
    if isinstance(v, dict) and v and isinstance(next(iter(v.keys())), str):
        return {
            tuple(k.split("|")): val  # type: ignore[misc]
            for k, val in v.items()
        }
    return v  # type: ignore[no-any-return]


def _serialize_marginal_rates(
    v: dict[tuple[str, str], Any],
) -> dict[str, list[Any]]:
    return {
        f"{k[0]}|{k[1]}": val.tolist()
        for k, val in v.items()
    }


MarginalRates = Annotated[
    dict[tuple[str, str], NumpyArray],
    PlainValidator(_validate_marginal_rates),
    PlainSerializer(_serialize_marginal_rates, return_type=dict),
]


class TradeOffResult(BaseModel):
    """Trade-off analysis between objective pairs on the Pareto front.

    Attributes:
        objective_pairs: All pairwise combinations of objectives.
        marginal_rates: Marginal rates of substitution per objective pair, each an array of rates between adjacent Pareto points.
        pareto_objectives: Objective values on the Pareto front, shape (n_points, n_objectives), sorted by first objective.
    """

    model_config = ConfigDict(frozen=True)

    objective_pairs: tuple[tuple[str, str], ...]
    marginal_rates: MarginalRates
    pareto_objectives: NumpyArray
