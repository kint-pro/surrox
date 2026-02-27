from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class PDPICEResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    variable_name: str
    column: str
    grid_values: NDArray
    pdp_values: NDArray
    ice_values: NDArray
