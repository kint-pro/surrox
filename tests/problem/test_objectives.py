import pytest
from pydantic import ValidationError

from surrox.problem.objectives import Objective
from surrox.problem.types import Direction


class TestObjective:
    def test_create_minimize_objective(self) -> None:
        obj = Objective(name="cost", direction=Direction.MINIMIZE, column="cost_col")
        assert obj.name == "cost"
        assert obj.direction == Direction.MINIMIZE
        assert obj.column == "cost_col"
        assert obj.reference_value is None

    def test_create_maximize_objective(self) -> None:
        obj = Objective(
            name="efficiency", direction=Direction.MAXIMIZE, column="eff_col"
        )
        assert obj.direction == Direction.MAXIMIZE

    def test_create_with_reference_value(self) -> None:
        obj = Objective(
            name="cost",
            direction=Direction.MINIMIZE,
            column="cost_col",
            reference_value=500.0,
        )
        assert obj.reference_value == 500.0

    def test_reference_value_defaults_to_none(self) -> None:
        obj = Objective(name="x", direction=Direction.MINIMIZE, column="col")
        assert obj.reference_value is None

    def test_reference_value_can_be_negative(self) -> None:
        obj = Objective(
            name="x",
            direction=Direction.MAXIMIZE,
            column="col",
            reference_value=-10.0,
        )
        assert obj.reference_value == -10.0

    def test_reference_value_can_be_zero(self) -> None:
        obj = Objective(
            name="x",
            direction=Direction.MINIMIZE,
            column="col",
            reference_value=0.0,
        )
        assert obj.reference_value == 0.0

    def test_name_required(self) -> None:
        with pytest.raises(ValidationError):
            Objective(direction=Direction.MINIMIZE, column="col")  # type: ignore[call-arg]

    def test_direction_required(self) -> None:
        with pytest.raises(ValidationError):
            Objective(name="x", column="col")  # type: ignore[call-arg]

    def test_column_required(self) -> None:
        with pytest.raises(ValidationError):
            Objective(name="x", direction=Direction.MINIMIZE)  # type: ignore[call-arg]

    def test_invalid_direction_raises(self) -> None:
        with pytest.raises(ValidationError):
            Objective(name="x", direction="invalid", column="col")  # type: ignore[arg-type]

    def test_is_frozen(self) -> None:
        obj = Objective(name="cost", direction=Direction.MINIMIZE, column="cost_col")
        with pytest.raises(ValidationError):
            obj.name = "other"  # type: ignore[misc]

    def test_serialization_round_trip(self) -> None:
        obj = Objective(
            name="profit",
            direction=Direction.MAXIMIZE,
            column="profit_col",
            reference_value=1000.0,
        )
        data = obj.model_dump()
        reconstructed = Objective(**data)
        assert reconstructed == obj

    def test_serialization_round_trip_without_reference_value(self) -> None:
        obj = Objective(name="x", direction=Direction.MINIMIZE, column="col")
        data = obj.model_dump()
        reconstructed = Objective(**data)
        assert reconstructed == obj
