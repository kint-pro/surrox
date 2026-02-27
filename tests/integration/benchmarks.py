from __future__ import annotations

import numpy as np
import pandas as pd

from surrox.problem.constraints import DataConstraint
from surrox.problem.definition import ProblemDefinition
from surrox.problem.objectives import Objective
from surrox.problem.types import (
    ConstraintOperator,
    Direction,
    DType,
    Role,
)
from surrox.problem.variables import ContinuousBounds, Variable


def branin_value(x1: float, x2: float) -> float:
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8 * np.pi)
    return float(
        a * (x2 - b * x1**2 + c * x1 - r) ** 2
        + s * (1 - t) * np.cos(x1)
        + s
    )


BRANIN_KNOWN_MINIMUM = 0.397887


def generate_branin(
    n_samples: int = 600, seed: int = 42,
) -> tuple[ProblemDefinition, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    x1 = rng.uniform(-5.0, 10.0, n_samples)
    x2 = rng.uniform(0.0, 15.0, n_samples)

    y = np.array([
        branin_value(a, b)
        for a, b in zip(x1, x2, strict=True)
    ])
    noise = rng.normal(0, 0.1, n_samples)
    constraint_col = x1 + x2 + noise

    problem = ProblemDefinition(
        variables=(
            Variable(
                name="x1", dtype=DType.CONTINUOUS,
                role=Role.DECISION,
                bounds=ContinuousBounds(lower=-5.0, upper=10.0),
            ),
            Variable(
                name="x2", dtype=DType.CONTINUOUS,
                role=Role.DECISION,
                bounds=ContinuousBounds(lower=0.0, upper=15.0),
            ),
        ),
        objectives=(
            Objective(
                name="branin",
                direction=Direction.MINIMIZE,
                column="branin",
            ),
        ),
        data_constraints=(
            DataConstraint(
                name="sum_limit", column="sum_x",
                operator=ConstraintOperator.LE, limit=20.0,
            ),
        ),
    )

    df = pd.DataFrame({
        "x1": x1, "x2": x2,
        "branin": y, "sum_x": constraint_col,
    })
    return problem, df


def rosenbrock_value(x1: float, x2: float) -> float:
    return float(100.0 * (x2 - x1**2) ** 2 + (1 - x1) ** 2)


ROSENBROCK_KNOWN_MINIMUM = 0.0
ROSENBROCK_KNOWN_OPTIMUM = (1.0, 1.0)


def generate_rosenbrock(
    n_samples: int = 600, seed: int = 42,
) -> tuple[ProblemDefinition, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    x1 = rng.uniform(-2.0, 2.0, n_samples)
    x2 = rng.uniform(-2.0, 2.0, n_samples)

    y = np.array([
        rosenbrock_value(a, b)
        for a, b in zip(x1, x2, strict=True)
    ])

    problem = ProblemDefinition(
        variables=(
            Variable(
                name="x1", dtype=DType.CONTINUOUS,
                role=Role.DECISION,
                bounds=ContinuousBounds(lower=-2.0, upper=2.0),
            ),
            Variable(
                name="x2", dtype=DType.CONTINUOUS,
                role=Role.DECISION,
                bounds=ContinuousBounds(lower=-2.0, upper=2.0),
            ),
        ),
        objectives=(
            Objective(
                name="rosenbrock",
                direction=Direction.MINIMIZE,
                column="rosenbrock",
            ),
        ),
    )

    df = pd.DataFrame({"x1": x1, "x2": x2, "rosenbrock": y})
    return problem, df


def zdt1_values(x: np.ndarray) -> tuple[float, float]:
    n = len(x)
    f1 = float(x[0])
    g = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)
    f2 = float(g * (1.0 - np.sqrt(f1 / g)))
    return f1, f2


def generate_zdt1(
    n_variables: int = 5,
    n_samples: int = 600,
    seed: int = 42,
) -> tuple[ProblemDefinition, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    X = rng.uniform(0.0, 1.0, (n_samples, n_variables))

    f1_vals = []
    f2_vals = []
    for row in X:
        f1, f2 = zdt1_values(row)
        f1_vals.append(f1)
        f2_vals.append(f2)

    var_names = [f"x{i}" for i in range(n_variables)]
    variables = tuple(
        Variable(
            name=name, dtype=DType.CONTINUOUS,
            role=Role.DECISION,
            bounds=ContinuousBounds(lower=0.0, upper=1.0),
        )
        for name in var_names
    )

    problem = ProblemDefinition(
        variables=variables,
        objectives=(
            Objective(
                name="f1", direction=Direction.MINIMIZE,
                column="f1",
            ),
            Objective(
                name="f2", direction=Direction.MINIMIZE,
                column="f2",
            ),
        ),
    )

    data = {name: X[:, i] for i, name in enumerate(var_names)}
    data["f1"] = f1_vals
    data["f2"] = f2_vals
    df = pd.DataFrame(data)
    return problem, df
