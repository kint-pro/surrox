import numpy as np
import pandas as pd
import pytest

from surrox.optimizer.extrapolation import ExtrapolationGate
from surrox.problem.types import DType, Role
from surrox.problem.variables import (
    CategoricalBounds,
    ContinuousBounds,
    IntegerBounds,
    OrdinalBounds,
    Variable,
)


@pytest.fixture
def continuous_vars() -> tuple[Variable, ...]:
    return (
        Variable(name="x1", dtype=DType.CONTINUOUS, role=Role.DECISION,
                 bounds=ContinuousBounds(lower=0.0, upper=10.0)),
        Variable(name="x2", dtype=DType.CONTINUOUS, role=Role.DECISION,
                 bounds=ContinuousBounds(lower=0.0, upper=10.0)),
    )


@pytest.fixture
def training_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({"x1": rng.uniform(0, 10, 100), "x2": rng.uniform(0, 10, 100)})


class TestExtrapolationGateContinuous:
    def test_points_inside_are_not_extrapolating(
        self, continuous_vars: tuple[Variable, ...], training_data: pd.DataFrame
    ) -> None:
        gate = ExtrapolationGate(training_data, continuous_vars, k=5, threshold=2.0)
        candidates = pd.DataFrame({"x1": [5.0, 5.0], "x2": [5.0, 5.0]})
        mask, distances = gate.evaluate(candidates)
        assert not mask[0]
        assert not mask[1]

    def test_points_far_outside_are_extrapolating(
        self, continuous_vars: tuple[Variable, ...], training_data: pd.DataFrame
    ) -> None:
        gate = ExtrapolationGate(training_data, continuous_vars, k=5, threshold=2.0)
        candidates = pd.DataFrame({"x1": [1000.0], "x2": [1000.0]})
        mask, distances = gate.evaluate(candidates)
        assert mask[0]
        assert distances[0] > 2.0

    def test_distances_are_positive(
        self, continuous_vars: tuple[Variable, ...], training_data: pd.DataFrame
    ) -> None:
        gate = ExtrapolationGate(training_data, continuous_vars, k=5, threshold=2.0)
        candidates = pd.DataFrame({"x1": [5.0], "x2": [5.0]})
        _, distances = gate.evaluate(candidates)
        assert distances[0] > 0


class TestExtrapolationGateCategorical:
    def test_unseen_combination_has_higher_distance(self) -> None:
        cat_var = Variable(
            name="mode", dtype=DType.CATEGORICAL, role=Role.DECISION,
            bounds=CategoricalBounds(categories=("auto", "manual", "hybrid")),
        )
        cont_var = Variable(
            name="x1", dtype=DType.CONTINUOUS, role=Role.DECISION,
            bounds=ContinuousBounds(lower=0.0, upper=10.0),
        )

        training = pd.DataFrame({
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 10,
            "mode": ["auto", "auto", "auto", "manual", "manual", "manual"] * 10,
        })

        gate = ExtrapolationGate(training, (cont_var, cat_var), k=3, threshold=2.0)

        seen = pd.DataFrame({"x1": [3.0], "mode": ["auto"]})
        unseen = pd.DataFrame({"x1": [3.0], "mode": ["hybrid"]})

        _, dist_seen = gate.evaluate(seen)
        _, dist_unseen = gate.evaluate(unseen)
        assert dist_unseen[0] > dist_seen[0]


class TestExtrapolationGateOrdinal:
    def test_ordinal_encoded_as_positional(self) -> None:
        ord_var = Variable(
            name="level", dtype=DType.ORDINAL, role=Role.DECISION,
            bounds=OrdinalBounds(categories=("low", "medium", "high")),
        )

        training = pd.DataFrame({"level": ["low", "medium", "high"] * 20})
        gate = ExtrapolationGate(training, (ord_var,), k=3, threshold=2.0)

        candidates = pd.DataFrame({"level": ["medium"]})
        mask, _ = gate.evaluate(candidates)
        assert not mask[0]


class TestExtrapolationGateInteger:
    def test_integer_vars_treated_as_numeric(self) -> None:
        int_var = Variable(
            name="n", dtype=DType.INTEGER, role=Role.DECISION,
            bounds=IntegerBounds(lower=1, upper=10),
        )

        training = pd.DataFrame({"n": list(range(1, 11)) * 10})
        gate = ExtrapolationGate(training, (int_var,), k=3, threshold=2.0)

        inside = pd.DataFrame({"n": [5]})
        mask, _ = gate.evaluate(inside)
        assert not mask[0]
