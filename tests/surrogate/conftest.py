import numpy as np
import pandas as pd
import pytest

from surrox.problem import (
    BoundDataset,
    ContinuousBounds,
    DataConstraint,
    Direction,
    DType,
    IntegerBounds,
    MonotonicDirection,
    MonotonicRelation,
    Objective,
    ProblemDefinition,
    Role,
    Variable,
)
from surrox.surrogate.config import TrainingConfig
from surrox.surrogate.families import (
    GaussianProcessFamily,
    LightGBMFamily,
    TabICLFamily,
    XGBoostFamily,
)


@pytest.fixture
def xgboost_family() -> XGBoostFamily:
    return XGBoostFamily()


@pytest.fixture
def lightgbm_family() -> LightGBMFamily:
    return LightGBMFamily()


@pytest.fixture
def gaussian_process_family() -> GaussianProcessFamily:
    return GaussianProcessFamily()


@pytest.fixture
def tabicl_family() -> TabICLFamily:
    return TabICLFamily()


@pytest.fixture
def training_config() -> TrainingConfig:
    return TrainingConfig(
        n_trials=10,
        cv_folds=3,
        study_timeout_s=120,
        ensemble_size=3,
        min_r2=None,
    )


@pytest.fixture
def problem_definition() -> ProblemDefinition:
    return ProblemDefinition(
        variables=(
            Variable(
                name="temperature",
                dtype=DType.CONTINUOUS,
                role=Role.DECISION,
                bounds=ContinuousBounds(lower=0.0, upper=100.0),
            ),
            Variable(
                name="pressure",
                dtype=DType.CONTINUOUS,
                role=Role.DECISION,
                bounds=ContinuousBounds(lower=0.0, upper=200.0),
            ),
            Variable(
                name="duration",
                dtype=DType.INTEGER,
                role=Role.DECISION,
                bounds=IntegerBounds(lower=1, upper=50),
            ),
        ),
        objectives=(
            Objective(name="yield", direction=Direction.MAXIMIZE, column="yield_pct"),
        ),
        data_constraints=(
            DataConstraint(
                name="max_emissions",
                column="emissions",
                operator="le",
                limit=100.0,
            ),
        ),
        monotonic_relations=(
            MonotonicRelation(
                decision_variable="temperature",
                objective_or_constraint="yield",
                direction=MonotonicDirection.INCREASING,
            ),
        ),
    )


@pytest.fixture
def synthetic_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 600

    temperature = rng.uniform(0, 100, n)
    pressure = rng.uniform(0, 200, n)
    duration = rng.integers(1, 51, n)

    yield_pct = 0.3 * temperature + 0.1 * pressure + 0.5 * duration + rng.normal(0, 5, n)
    emissions = 0.2 * temperature + 0.05 * pressure + rng.normal(0, 3, n)

    return pd.DataFrame({
        "temperature": temperature,
        "pressure": pressure,
        "duration": duration,
        "yield_pct": yield_pct,
        "emissions": emissions,
    })


@pytest.fixture
def bound_dataset(
    problem_definition: ProblemDefinition, synthetic_dataframe: pd.DataFrame
) -> BoundDataset:
    return BoundDataset(problem=problem_definition, dataframe=synthetic_dataframe)
