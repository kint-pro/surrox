import pytest
from pydantic import ValidationError

from surrox.problem.types import DType, Role
from surrox.problem.variables import (
    CategoricalBounds,
    ContinuousBounds,
    IntegerBounds,
    OrdinalBounds,
    Variable,
)


class TestContinuousBounds:
    def test_create_valid_bounds(self) -> None:
        """Test that valid continuous bounds are created successfully."""
        bounds = ContinuousBounds(lower=0.0, upper=10.0)
        assert bounds.lower == 0.0
        assert bounds.upper == 10.0
        assert bounds.type == "continuous"

    def test_create_with_negative_lower(self) -> None:
        """Test that negative lower bound is valid when upper is greater."""
        bounds = ContinuousBounds(lower=-100.0, upper=-1.0)
        assert bounds.lower == -100.0
        assert bounds.upper == -1.0

    def test_create_with_fractional_values(self) -> None:
        """Test that fractional float bounds are accepted."""
        bounds = ContinuousBounds(lower=0.001, upper=0.999)
        assert bounds.lower == pytest.approx(0.001)
        assert bounds.upper == pytest.approx(0.999)

    def test_type_field_is_always_continuous(self) -> None:
        """Test that the type discriminator is always 'continuous'."""
        bounds = ContinuousBounds(lower=1.0, upper=2.0)
        assert bounds.type == "continuous"

    def test_lower_equal_to_upper_raises(self) -> None:
        """Test that lower == upper raises a ValidationError."""
        with pytest.raises(ValidationError, match="lower .* must be less than upper"):
            ContinuousBounds(lower=5.0, upper=5.0)

    def test_lower_greater_than_upper_raises(self) -> None:
        """Test that lower > upper raises a ValidationError."""
        with pytest.raises(ValidationError, match="lower .* must be less than upper"):
            ContinuousBounds(lower=10.0, upper=1.0)

    def test_is_frozen(self) -> None:
        """Test that ContinuousBounds is immutable."""
        bounds = ContinuousBounds(lower=0.0, upper=1.0)
        with pytest.raises(ValidationError):
            bounds.lower = 5.0  # type: ignore[misc]

    def test_serialization_round_trip(self) -> None:
        """Test that model_dump and reconstruction produce equal objects."""
        bounds = ContinuousBounds(lower=-5.0, upper=5.0)
        data = bounds.model_dump()
        reconstructed = ContinuousBounds(**data)
        assert reconstructed == bounds


class TestIntegerBounds:
    def test_create_valid_bounds(self) -> None:
        """Test that valid integer bounds are created successfully."""
        bounds = IntegerBounds(lower=0, upper=100)
        assert bounds.lower == 0
        assert bounds.upper == 100
        assert bounds.type == "integer"

    def test_create_with_negative_values(self) -> None:
        """Test that negative integer bounds are valid."""
        bounds = IntegerBounds(lower=-50, upper=-1)
        assert bounds.lower == -50
        assert bounds.upper == -1

    def test_lower_equal_to_upper_raises(self) -> None:
        """Test that lower == upper raises a ValidationError."""
        with pytest.raises(ValidationError, match="lower .* must be less than upper"):
            IntegerBounds(lower=5, upper=5)

    def test_lower_greater_than_upper_raises(self) -> None:
        """Test that lower > upper raises a ValidationError."""
        with pytest.raises(ValidationError, match="lower .* must be less than upper"):
            IntegerBounds(lower=10, upper=1)

    def test_is_frozen(self) -> None:
        """Test that IntegerBounds is immutable."""
        bounds = IntegerBounds(lower=0, upper=10)
        with pytest.raises(ValidationError):
            bounds.upper = 20  # type: ignore[misc]

    def test_serialization_round_trip(self) -> None:
        """Test that model_dump and reconstruction produce equal objects."""
        bounds = IntegerBounds(lower=1, upper=99)
        data = bounds.model_dump()
        reconstructed = IntegerBounds(**data)
        assert reconstructed == bounds


class TestCategoricalBounds:
    def test_create_with_two_categories(self) -> None:
        """Test that two categories is the minimum valid count."""
        bounds = CategoricalBounds(categories=("a", "b"))
        assert bounds.categories == ("a", "b")
        assert bounds.type == "categorical"

    def test_create_with_many_categories(self) -> None:
        """Test that many categories are accepted."""
        cats = tuple(f"cat_{i}" for i in range(50))
        bounds = CategoricalBounds(categories=cats)
        assert len(bounds.categories) == 50

    def test_single_category_raises(self) -> None:
        """Test that fewer than 2 categories raises a ValidationError."""
        with pytest.raises(ValidationError, match="at least 2 categories"):
            CategoricalBounds(categories=("only_one",))

    def test_zero_categories_raises(self) -> None:
        """Test that empty categories raises a ValidationError."""
        with pytest.raises(ValidationError, match="at least 2 categories"):
            CategoricalBounds(categories=())

    def test_duplicate_categories_raises(self) -> None:
        """Test that duplicate categories raises a ValidationError."""
        with pytest.raises(ValidationError, match="categories must be unique"):
            CategoricalBounds(categories=("a", "b", "a"))

    def test_is_frozen(self) -> None:
        """Test that CategoricalBounds is immutable."""
        bounds = CategoricalBounds(categories=("x", "y"))
        with pytest.raises(ValidationError):
            bounds.categories = ("x", "y", "z")  # type: ignore[misc]

    def test_serialization_round_trip(self) -> None:
        """Test that model_dump and reconstruction produce equal objects."""
        bounds = CategoricalBounds(categories=("low", "medium", "high"))
        data = bounds.model_dump()
        reconstructed = CategoricalBounds(**data)
        assert reconstructed == bounds


class TestOrdinalBounds:
    def test_create_with_two_categories(self) -> None:
        """Test that two ordinal categories is the minimum valid count."""
        bounds = OrdinalBounds(categories=("small", "large"))
        assert bounds.categories == ("small", "large")
        assert bounds.type == "ordinal"

    def test_create_with_many_categories(self) -> None:
        """Test that many ordinal categories are accepted."""
        cats = tuple(f"level_{i}" for i in range(10))
        bounds = OrdinalBounds(categories=cats)
        assert len(bounds.categories) == 10

    def test_single_category_raises(self) -> None:
        """Test that fewer than 2 ordinal categories raises a ValidationError."""
        with pytest.raises(ValidationError, match="at least 2 categories"):
            OrdinalBounds(categories=("only_one",))

    def test_zero_categories_raises(self) -> None:
        """Test that empty ordinal categories raises a ValidationError."""
        with pytest.raises(ValidationError, match="at least 2 categories"):
            OrdinalBounds(categories=())

    def test_duplicate_categories_raises(self) -> None:
        """Test that duplicate ordinal categories raises a ValidationError."""
        with pytest.raises(ValidationError, match="categories must be unique"):
            OrdinalBounds(categories=("a", "b", "a"))

    def test_is_frozen(self) -> None:
        """Test that OrdinalBounds is immutable."""
        bounds = OrdinalBounds(categories=("x", "y"))
        with pytest.raises(ValidationError):
            bounds.categories = ("x", "y", "z")  # type: ignore[misc]

    def test_serialization_round_trip(self) -> None:
        """Test that model_dump and reconstruction produce equal objects."""
        bounds = OrdinalBounds(categories=("cold", "warm", "hot"))
        data = bounds.model_dump()
        reconstructed = OrdinalBounds(**data)
        assert reconstructed == bounds


class TestVariable:
    def test_create_continuous_decision_variable(self) -> None:
        """Test creating a continuous decision variable."""
        var = Variable(
            name="temperature",
            dtype=DType.CONTINUOUS,
            role=Role.DECISION,
            bounds=ContinuousBounds(lower=0.0, upper=100.0),
        )
        assert var.name == "temperature"
        assert var.dtype == DType.CONTINUOUS
        assert var.role == Role.DECISION
        assert var.bounds.type == "continuous"

    def test_create_integer_decision_variable(self) -> None:
        """Test creating an integer decision variable."""
        var = Variable(
            name="count",
            dtype=DType.INTEGER,
            role=Role.DECISION,
            bounds=IntegerBounds(lower=1, upper=10),
        )
        assert var.dtype == DType.INTEGER
        assert var.bounds.type == "integer"

    def test_create_categorical_context_variable(self) -> None:
        """Test creating a categorical context variable."""
        var = Variable(
            name="mode",
            dtype=DType.CATEGORICAL,
            role=Role.CONTEXT,
            bounds=CategoricalBounds(categories=("on", "off")),
        )
        assert var.role == Role.CONTEXT
        assert var.dtype == DType.CATEGORICAL

    def test_create_ordinal_context_variable(self) -> None:
        """Test creating an ordinal context variable."""
        var = Variable(
            name="quality",
            dtype=DType.ORDINAL,
            role=Role.CONTEXT,
            bounds=OrdinalBounds(categories=("low", "medium", "high")),
        )
        assert var.dtype == DType.ORDINAL
        assert var.role == Role.CONTEXT

    def test_continuous_dtype_with_integer_bounds_raises(self) -> None:
        """Test that continuous dtype with integer bounds raises ValidationError."""
        with pytest.raises(ValidationError, match="requires continuous bounds"):
            Variable(
                name="x",
                dtype=DType.CONTINUOUS,
                role=Role.DECISION,
                bounds=IntegerBounds(lower=0, upper=10),
            )

    def test_integer_dtype_with_continuous_bounds_raises(self) -> None:
        """Test that integer dtype with continuous bounds raises ValidationError."""
        with pytest.raises(ValidationError, match="requires integer bounds"):
            Variable(
                name="x",
                dtype=DType.INTEGER,
                role=Role.DECISION,
                bounds=ContinuousBounds(lower=0.0, upper=10.0),
            )

    def test_categorical_dtype_with_ordinal_bounds_raises(self) -> None:
        """Test that categorical dtype with ordinal bounds raises ValidationError."""
        with pytest.raises(ValidationError, match="requires categorical bounds"):
            Variable(
                name="x",
                dtype=DType.CATEGORICAL,
                role=Role.CONTEXT,
                bounds=OrdinalBounds(categories=("a", "b")),
            )

    def test_ordinal_dtype_with_categorical_bounds_raises(self) -> None:
        """Test that ordinal dtype with categorical bounds raises ValidationError."""
        with pytest.raises(ValidationError, match="requires ordinal bounds"):
            Variable(
                name="x",
                dtype=DType.ORDINAL,
                role=Role.CONTEXT,
                bounds=CategoricalBounds(categories=("a", "b")),
            )

    def test_is_frozen(self) -> None:
        """Test that Variable is immutable."""
        var = Variable(
            name="x",
            dtype=DType.CONTINUOUS,
            role=Role.DECISION,
            bounds=ContinuousBounds(lower=0.0, upper=1.0),
        )
        with pytest.raises(ValidationError):
            var.name = "y"  # type: ignore[misc]

    def test_serialization_round_trip(self) -> None:
        """Test that model_dump and reconstruction produce equal objects."""
        var = Variable(
            name="pressure",
            dtype=DType.CONTINUOUS,
            role=Role.DECISION,
            bounds=ContinuousBounds(lower=1.0, upper=200.0),
        )
        data = var.model_dump()
        reconstructed = Variable(**data)
        assert reconstructed == var

    def test_name_required(self) -> None:
        """Test that name is required."""
        with pytest.raises(ValidationError):
            Variable(
                dtype=DType.CONTINUOUS,  # type: ignore[call-arg]
                role=Role.DECISION,
                bounds=ContinuousBounds(lower=0.0, upper=1.0),
            )

    def test_bounds_required(self) -> None:
        """Test that bounds is required."""
        with pytest.raises(ValidationError):
            Variable(
                name="x",  # type: ignore[call-arg]
                dtype=DType.CONTINUOUS,
                role=Role.DECISION,
            )
