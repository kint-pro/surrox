from __future__ import annotations

from typing import Annotated, Any

import numpy as np
from numpy.typing import NDArray
from pydantic import PlainSerializer, PlainValidator


def _validate_numpy_array(v: Any) -> NDArray[np.floating[Any]]:
    return np.asarray(v, dtype=np.float64)


def _serialize_numpy_array(v: NDArray[np.floating[Any]]) -> list[Any]:
    return v.tolist()


NumpyArray = Annotated[
    NDArray[np.floating[Any]],
    PlainValidator(_validate_numpy_array),
    PlainSerializer(_serialize_numpy_array, return_type=list),
]
