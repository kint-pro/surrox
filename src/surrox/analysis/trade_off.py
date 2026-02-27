from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class TradeOffResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    objective_pairs: tuple[tuple[str, str], ...]
    marginal_rates: dict[tuple[str, str], NDArray]
    pareto_objectives: NDArray
