import pytest
from pydantic import ValidationError

from surrox.problem.domain_knowledge import MonotonicRelation
from surrox.problem.types import MonotonicDirection


class TestMonotonicRelation:
    def test_create_increasing_relation(self) -> None:
        relation = MonotonicRelation(
            decision_variable="temperature",
            objective_or_constraint="cost",
            direction=MonotonicDirection.INCREASING,
        )
        assert relation.decision_variable == "temperature"
        assert relation.objective_or_constraint == "cost"
        assert relation.direction == MonotonicDirection.INCREASING

    def test_create_decreasing_relation(self) -> None:
        relation = MonotonicRelation(
            decision_variable="efficiency",
            objective_or_constraint="emissions",
            direction=MonotonicDirection.DECREASING,
        )
        assert relation.direction == MonotonicDirection.DECREASING

    def test_decision_variable_required(self) -> None:
        with pytest.raises(ValidationError):
            MonotonicRelation(objective_or_constraint="cost", direction=MonotonicDirection.INCREASING)  # type: ignore[call-arg]

    def test_objective_or_constraint_required(self) -> None:
        with pytest.raises(ValidationError):
            MonotonicRelation(decision_variable="x", direction=MonotonicDirection.INCREASING)  # type: ignore[call-arg]

    def test_direction_required(self) -> None:
        with pytest.raises(ValidationError):
            MonotonicRelation(decision_variable="x", objective_or_constraint="y")  # type: ignore[call-arg]

    def test_invalid_direction_raises(self) -> None:
        with pytest.raises(ValidationError):
            MonotonicRelation(decision_variable="x", objective_or_constraint="y", direction="invalid")  # type: ignore[arg-type]

    def test_is_frozen(self) -> None:
        relation = MonotonicRelation(
            decision_variable="x",
            objective_or_constraint="y",
            direction=MonotonicDirection.INCREASING,
        )
        with pytest.raises(ValidationError):
            relation.decision_variable = "z"  # type: ignore[misc]

    def test_serialization_round_trip(self) -> None:
        relation = MonotonicRelation(
            decision_variable="pressure",
            objective_or_constraint="yield",
            direction=MonotonicDirection.DECREASING,
        )
        data = relation.model_dump()
        reconstructed = MonotonicRelation(**data)
        assert reconstructed == relation
