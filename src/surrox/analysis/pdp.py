from pydantic import BaseModel, ConfigDict

from surrox.types import NumpyArray


class PDPICEResult(BaseModel):
    """Partial Dependence Plot and Individual Conditional Expectation curves.

    Attributes:
        variable_name: Decision variable being varied.
        column: Target column being predicted.
        grid_values: Grid points along the variable range, shape (n_grid,).
        pdp_values: Ensemble-weighted PDP values, shape (n_grid,).
        ice_values: Ensemble-weighted ICE curves, shape (n_samples, n_grid).
    """

    model_config = ConfigDict(frozen=True)

    variable_name: str
    column: str
    grid_values: NumpyArray
    pdp_values: NumpyArray
    ice_values: NumpyArray
