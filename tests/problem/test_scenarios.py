import pytest
from pydantic import ValidationError

from surrox.problem.scenarios import Scenario


class TestScenario:
    def test_create_with_single_context_value(self) -> None:
        """Test creating a scenario with one context variable value."""
        scenario = Scenario(name="baseline", context_values={"mode": "low"})
        assert scenario.name == "baseline"
        assert scenario.context_values == {"mode": "low"}

    def test_create_with_multiple_context_values(self) -> None:
        """Test creating a scenario with multiple context variable values."""
        scenario = Scenario(
            name="full_scenario",
            context_values={"mode": "high", "region": "north", "season": "summer"},
        )
        assert len(scenario.context_values) == 3

    def test_create_with_numeric_context_value(self) -> None:
        """Test creating a scenario where context values can be numeric."""
        scenario = Scenario(name="numeric", context_values={"load": 42.5})
        assert scenario.context_values["load"] == 42.5

    def test_empty_context_values_raises(self) -> None:
        """Test that an empty context_values dict raises a ValidationError."""
        with pytest.raises(
            ValidationError, match="at least one context variable value"
        ):
            Scenario(name="empty", context_values={})

    def test_name_required(self) -> None:
        """Test that name is required."""
        with pytest.raises(ValidationError):
            Scenario(context_values={"x": 1})  # type: ignore[call-arg]

    def test_context_values_required(self) -> None:
        """Test that context_values is required."""
        with pytest.raises(ValidationError):
            Scenario(name="s")  # type: ignore[call-arg]

    def test_is_frozen(self) -> None:
        """Test that Scenario is immutable."""
        scenario = Scenario(name="s", context_values={"x": 1})
        with pytest.raises(ValidationError):
            scenario.name = "other"  # type: ignore[misc]

    def test_serialization_round_trip(self) -> None:
        """Test that model_dump and reconstruction produce equal objects."""
        scenario = Scenario(
            name="winter_peak",
            context_values={"season": "winter", "demand": "high"},
        )
        data = scenario.model_dump()
        reconstructed = Scenario(**data)
        assert reconstructed == scenario
