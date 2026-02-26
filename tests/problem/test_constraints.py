import pytest
from pydantic import ValidationError

from surrox.problem.constraints import DataConstraint, LinearConstraint
from surrox.problem.types import ConstraintOperator


class TestLinearConstraint:
    def test_create_with_single_coefficient(self) -> None:
        lc = LinearConstraint(
            name="upper_limit",
            coefficients={"x": 1.0},
            operator=ConstraintOperator.LE,
            rhs=10.0,
        )
        assert lc.name == "upper_limit"
        assert lc.coefficients == {"x": 1.0}
        assert lc.operator == ConstraintOperator.LE
        assert lc.rhs == 10.0

    def test_create_with_multiple_coefficients(self) -> None:
        lc = LinearConstraint(
            name="sum_limit",
            coefficients={"x": 1.0, "y": 1.0},
            operator=ConstraintOperator.LE,
            rhs=100.0,
        )
        assert lc.coefficients == {"x": 1.0, "y": 1.0}

    def test_create_with_ge_operator(self) -> None:
        lc = LinearConstraint(
            name="min_budget",
            coefficients={"budget": 1.0},
            operator=ConstraintOperator.GE,
            rhs=50.0,
        )
        assert lc.operator == ConstraintOperator.GE

    def test_create_with_eq_operator(self) -> None:
        lc = LinearConstraint(
            name="exact_sum",
            coefficients={"a": 1.0, "b": 1.0},
            operator=ConstraintOperator.EQ,
            rhs=20.0,
        )
        assert lc.operator == ConstraintOperator.EQ

    def test_create_with_negative_coefficients(self) -> None:
        lc = LinearConstraint(
            name="diff",
            coefficients={"x": 1.0, "y": -1.0},
            operator=ConstraintOperator.GE,
            rhs=0.0,
        )
        assert lc.coefficients["y"] == -1.0

    def test_empty_coefficients_raises(self) -> None:
        with pytest.raises(ValidationError, match="coefficients must not be empty"):
            LinearConstraint(
                name="empty",
                coefficients={},
                operator=ConstraintOperator.LE,
                rhs=1.0,
            )

    def test_zero_coefficient_raises(self) -> None:
        with pytest.raises(ValidationError, match="coefficients must not be zero"):
            LinearConstraint(
                name="zero",
                coefficients={"x": 1.0, "y": 0.0},
                operator=ConstraintOperator.LE,
                rhs=10.0,
            )

    def test_name_required(self) -> None:
        with pytest.raises(ValidationError):
            LinearConstraint(
                coefficients={"x": 1.0},  # type: ignore[call-arg]
                operator=ConstraintOperator.LE,
                rhs=1.0,
            )

    def test_is_frozen(self) -> None:
        lc = LinearConstraint(
            name="c",
            coefficients={"x": 1.0},
            operator=ConstraintOperator.LE,
            rhs=1.0,
        )
        with pytest.raises(ValidationError):
            lc.rhs = 2.0  # type: ignore[misc]

    def test_serialization_round_trip(self) -> None:
        lc = LinearConstraint(
            name="budget",
            coefficients={"price": 2.0, "qty": 3.0},
            operator=ConstraintOperator.LE,
            rhs=500.0,
        )
        data = lc.model_dump()
        reconstructed = LinearConstraint(**data)
        assert reconstructed == lc


class TestDataConstraint:
    def test_create_with_le_operator(self) -> None:
        dc = DataConstraint(
            name="pressure_limit",
            column="pressure",
            operator=ConstraintOperator.LE,
            limit=100.0,
        )
        assert dc.name == "pressure_limit"
        assert dc.column == "pressure"
        assert dc.operator == ConstraintOperator.LE
        assert dc.limit == 100.0

    def test_create_with_ge_operator(self) -> None:
        dc = DataConstraint(
            name="min_flow",
            column="flow_rate",
            operator=ConstraintOperator.GE,
            limit=5.0,
        )
        assert dc.operator == ConstraintOperator.GE

    def test_create_with_eq_operator(self) -> None:
        dc = DataConstraint(
            name="exact_temp",
            column="temperature",
            operator=ConstraintOperator.EQ,
            limit=25.0,
        )
        assert dc.operator == ConstraintOperator.EQ

    def test_create_with_negative_limit(self) -> None:
        dc = DataConstraint(
            name="neg_limit",
            column="delta",
            operator=ConstraintOperator.GE,
            limit=-50.0,
        )
        assert dc.limit == -50.0

    def test_create_with_zero_limit(self) -> None:
        dc = DataConstraint(
            name="zero_limit",
            column="residual",
            operator=ConstraintOperator.LE,
            limit=0.0,
        )
        assert dc.limit == 0.0

    def test_name_required(self) -> None:
        with pytest.raises(ValidationError):
            DataConstraint(column="x", operator=ConstraintOperator.LE, limit=1.0)  # type: ignore[call-arg]

    def test_column_required(self) -> None:
        with pytest.raises(ValidationError):
            DataConstraint(name="c", operator=ConstraintOperator.LE, limit=1.0)  # type: ignore[call-arg]

    def test_operator_required(self) -> None:
        with pytest.raises(ValidationError):
            DataConstraint(name="c", column="x", limit=1.0)  # type: ignore[call-arg]

    def test_limit_required(self) -> None:
        with pytest.raises(ValidationError):
            DataConstraint(name="c", column="x", operator=ConstraintOperator.LE)  # type: ignore[call-arg]

    def test_invalid_operator_raises(self) -> None:
        with pytest.raises(ValidationError):
            DataConstraint(name="c", column="x", operator="invalid", limit=1.0)  # type: ignore[arg-type]

    def test_is_frozen(self) -> None:
        dc = DataConstraint(
            name="c", column="x", operator=ConstraintOperator.LE, limit=1.0
        )
        with pytest.raises(ValidationError):
            dc.limit = 2.0  # type: ignore[misc]

    def test_serialization_round_trip(self) -> None:
        dc = DataConstraint(
            name="emission_cap",
            column="co2",
            operator=ConstraintOperator.LE,
            limit=500.0,
        )
        data = dc.model_dump()
        reconstructed = DataConstraint(**data)
        assert reconstructed == dc
