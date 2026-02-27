from pydantic import BaseModel, ConfigDict

from surrox.types import NumpyArray


class PDPICEResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    variable_name: str
    column: str
    grid_values: NumpyArray
    pdp_values: NumpyArray
    ice_values: NumpyArray
