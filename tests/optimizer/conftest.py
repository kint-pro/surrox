from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from surrox.optimizer.config import OptimizerConfig
from surrox.problem.constraints import DataConstraint, LinearConstraint
from surrox.problem.definition import ProblemDefinition
from surrox.problem.dataset import BoundDataset
from surrox.problem.objectives import Objective
from surrox.problem.types import ConstraintOperator, Direction, DType, Role
from surrox.problem.variables import (
    CategoricalBounds,
    ContinuousBounds,
    IntegerBounds,
    Variable,
)
from surrox.surrogate.models import SurrogatePrediction


def _make_surrogate_manager(
    columns: list[str],
    predict_fn: dict[str, float] | None = None,
    lower_fn: dict[str, float] | None = None,
    upper_fn: dict[str, float] | None = None,
) -> MagicMock:
    predict_values = predict_fn or {c: 50.0 for c in columns}
    lower_values = lower_fn or {c: 40.0 for c in columns}
    upper_values = upper_fn or {c: 60.0 for c in columns}

    manager = MagicMock()
    manager.evaluate.side_effect = lambda df: {
        col: np.full(len(df), predict_values[col]) for col in columns
    }
    manager.evaluate_with_uncertainty.side_effect = lambda df, coverage=None: {
        col: SurrogatePrediction(
            mean=np.full(len(df), predict_values[col]),
            std=np.full(len(df), 5.0),
            lower=np.full(len(df), lower_values[col]),
            upper=np.full(len(df), upper_values[col]),
        )
        for col in columns
    }
    return manager


@pytest.fixture
def continuous_problem() -> ProblemDefinition:
    return ProblemDefinition(
        variables=(
            Variable(
                name="x1", dtype=DType.CONTINUOUS, role=Role.DECISION,
                bounds=ContinuousBounds(lower=0.0, upper=10.0),
            ),
            Variable(
                name="x2", dtype=DType.CONTINUOUS, role=Role.DECISION,
                bounds=ContinuousBounds(lower=0.0, upper=10.0),
            ),
        ),
        objectives=(
            Objective(name="obj1", direction=Direction.MINIMIZE, column="y1"),
        ),
    )


@pytest.fixture
def integer_problem() -> ProblemDefinition:
    return ProblemDefinition(
        variables=(
            Variable(
                name="x1", dtype=DType.CONTINUOUS, role=Role.DECISION,
                bounds=ContinuousBounds(lower=0.0, upper=10.0),
            ),
            Variable(
                name="n_cranes", dtype=DType.INTEGER, role=Role.DECISION,
                bounds=IntegerBounds(lower=1, upper=12),
            ),
        ),
        objectives=(
            Objective(name="obj1", direction=Direction.MINIMIZE, column="y1"),
        ),
    )


@pytest.fixture
def categorical_problem() -> ProblemDefinition:
    return ProblemDefinition(
        variables=(
            Variable(
                name="x1", dtype=DType.CONTINUOUS, role=Role.DECISION,
                bounds=ContinuousBounds(lower=0.0, upper=10.0),
            ),
            Variable(
                name="mode", dtype=DType.CATEGORICAL, role=Role.DECISION,
                bounds=CategoricalBounds(categories=("auto", "manual")),
            ),
        ),
        objectives=(
            Objective(name="obj1", direction=Direction.MINIMIZE, column="y1"),
        ),
    )


@pytest.fixture
def multi_objective_problem() -> ProblemDefinition:
    return ProblemDefinition(
        variables=(
            Variable(
                name="x1", dtype=DType.CONTINUOUS, role=Role.DECISION,
                bounds=ContinuousBounds(lower=0.0, upper=10.0),
            ),
            Variable(
                name="x2", dtype=DType.CONTINUOUS, role=Role.DECISION,
                bounds=ContinuousBounds(lower=0.0, upper=10.0),
            ),
        ),
        objectives=(
            Objective(name="cost", direction=Direction.MINIMIZE, column="cost"),
            Objective(name="quality", direction=Direction.MAXIMIZE, column="quality"),
        ),
    )


@pytest.fixture
def small_config() -> OptimizerConfig:
    return OptimizerConfig(
        population_size=10,
        n_generations=5,
        seed=42,
    )


@pytest.fixture
def training_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "x1": rng.uniform(0, 10, n),
        "x2": rng.uniform(0, 10, n),
        "y1": rng.uniform(0, 100, n),
    })


@pytest.fixture
def training_df_with_categorical() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "x1": rng.uniform(0, 10, n),
        "mode": rng.choice(["auto", "manual"], n),
        "y1": rng.uniform(0, 100, n),
    })


@pytest.fixture
def training_df_with_integer() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "x1": rng.uniform(0, 10, n),
        "n_cranes": rng.integers(1, 13, n),
        "y1": rng.uniform(0, 100, n),
    })


@pytest.fixture
def mock_surrogate() -> MagicMock:
    return _make_surrogate_manager(["y1"])


@pytest.fixture
def mock_surrogate_multi() -> MagicMock:
    return _make_surrogate_manager(
        ["cost", "quality"],
        predict_fn={"cost": 50.0, "quality": 80.0},
        lower_fn={"cost": 40.0, "quality": 70.0},
        upper_fn={"cost": 60.0, "quality": 90.0},
    )
