from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from surrox.exceptions import OptimizationError
from surrox.optimizer.config import OptimizerConfig
from surrox.optimizer.extrapolation import ExtrapolationGate
from surrox.optimizer.problem_adapter import SurroxProblem, _build_pymoo_variables
from surrox.problem.constraints import DataConstraint, LinearConstraint
from surrox.problem.definition import ProblemDefinition
from surrox.problem.objectives import Objective
from surrox.problem.scenarios import Scenario
from surrox.problem.types import ConstraintOperator, ConstraintSeverity, Direction, DType, Role
from surrox.problem.variables import (
    CategoricalBounds,
    ContinuousBounds,
    IntegerBounds,
    Variable,
)
from surrox.surrogate.models import SurrogatePrediction


class TestBuildPymooVariables:
    def test_continuous_maps_to_real(self) -> None:
        problem = ProblemDefinition(
            variables=(
                Variable(name="x", dtype=DType.CONTINUOUS, role=Role.DECISION,
                         bounds=ContinuousBounds(lower=0.0, upper=10.0)),
            ),
            objectives=(Objective(name="o", direction=Direction.MINIMIZE, column="y"),),
        )
        pymoo_vars = _build_pymoo_variables(problem)
        assert "x" in pymoo_vars

    def test_integer_maps_to_integer(self) -> None:
        problem = ProblemDefinition(
            variables=(
                Variable(name="n", dtype=DType.INTEGER, role=Role.DECISION,
                         bounds=IntegerBounds(lower=1, upper=10)),
            ),
            objectives=(Objective(name="o", direction=Direction.MINIMIZE, column="y"),),
        )
        pymoo_vars = _build_pymoo_variables(problem)
        assert "n" in pymoo_vars

    def test_categorical_maps_to_choice(self) -> None:
        problem = ProblemDefinition(
            variables=(
                Variable(name="m", dtype=DType.CATEGORICAL, role=Role.DECISION,
                         bounds=CategoricalBounds(categories=("a", "b"))),
            ),
            objectives=(Objective(name="o", direction=Direction.MINIMIZE, column="y"),),
        )
        pymoo_vars = _build_pymoo_variables(problem)
        assert "m" in pymoo_vars


class TestSurroxProblemFailFast:
    def test_context_variables_without_scenario_raises(self) -> None:
        problem = ProblemDefinition(
            variables=(
                Variable(name="x", dtype=DType.CONTINUOUS, role=Role.DECISION,
                         bounds=ContinuousBounds(lower=0.0, upper=10.0)),
                Variable(name="ctx", dtype=DType.CATEGORICAL, role=Role.CONTEXT,
                         bounds=CategoricalBounds(categories=("a", "b"))),
            ),
            objectives=(Objective(name="o", direction=Direction.MINIMIZE, column="y"),),
        )
        gate = MagicMock(spec=ExtrapolationGate)
        with pytest.raises(OptimizationError, match="context variables"):
            SurroxProblem(
                problem=problem,
                surrogate_manager=MagicMock(),
                extrapolation_gate=gate,
                config=OptimizerConfig(),
                extrapolation_penalty=1e6,
                scenario=None,
            )

    def test_no_context_variables_without_scenario_ok(self) -> None:
        problem = ProblemDefinition(
            variables=(
                Variable(name="x", dtype=DType.CONTINUOUS, role=Role.DECISION,
                         bounds=ContinuousBounds(lower=0.0, upper=10.0)),
            ),
            objectives=(Objective(name="o", direction=Direction.MINIMIZE, column="y"),),
        )
        gate = MagicMock(spec=ExtrapolationGate)
        surrogate = MagicMock()
        surrogate.get_ensemble_r2.return_value = 0.0
        sp = SurroxProblem(
            problem=problem,
            surrogate_manager=surrogate,
            extrapolation_gate=gate,
            config=OptimizerConfig(),
            extrapolation_penalty=1e6,
        )
        assert sp is not None


class TestSurroxProblemEvaluation:
    def test_minimize_direction_direct(self) -> None:
        problem = ProblemDefinition(
            variables=(
                Variable(name="x", dtype=DType.CONTINUOUS, role=Role.DECISION,
                         bounds=ContinuousBounds(lower=0.0, upper=10.0)),
            ),
            objectives=(
                Objective(name="obj", direction=Direction.MINIMIZE, column="y"),
            ),
        )

        surrogate = MagicMock()
        surrogate.evaluate.return_value = {"y": np.array([42.0])}
        surrogate.get_ensemble_r2.return_value = 0.0

        training = pd.DataFrame({"x": np.linspace(0, 10, 50)})
        gate = ExtrapolationGate(
            training, problem.decision_variables, k=3, threshold=100.0,
        )

        config = OptimizerConfig(acquisition="direct")
        sp = SurroxProblem(problem, surrogate, gate, config, 1e6)

        out: dict = {}
        sp._evaluate({"x": 5.0}, out)
        assert out["F"][0] == 42.0

    def test_maximize_direction_direct_negates(self) -> None:
        problem = ProblemDefinition(
            variables=(
                Variable(name="x", dtype=DType.CONTINUOUS, role=Role.DECISION,
                         bounds=ContinuousBounds(lower=0.0, upper=10.0)),
            ),
            objectives=(
                Objective(name="obj", direction=Direction.MAXIMIZE, column="y"),
            ),
        )

        surrogate = MagicMock()
        surrogate.evaluate.return_value = {"y": np.array([42.0])}
        surrogate.get_ensemble_r2.return_value = 0.0

        training = pd.DataFrame({"x": np.linspace(0, 10, 50)})
        gate = ExtrapolationGate(
            training, problem.decision_variables, k=3, threshold=100.0,
        )

        config = OptimizerConfig(acquisition="direct")
        sp = SurroxProblem(problem, surrogate, gate, config, 1e6)

        out: dict = {}
        sp._evaluate({"x": 5.0}, out)
        assert out["F"][0] == -42.0

    def test_pessimistic_minimize_uses_mean_plus_beta_std(self) -> None:
        problem = ProblemDefinition(
            variables=(
                Variable(name="x", dtype=DType.CONTINUOUS, role=Role.DECISION,
                         bounds=ContinuousBounds(lower=0.0, upper=10.0)),
            ),
            objectives=(
                Objective(name="obj", direction=Direction.MINIMIZE, column="y"),
            ),
        )

        surrogate = MagicMock()
        surrogate.evaluate_with_uncertainty.return_value = {
            "y": SurrogatePrediction(
                mean=np.array([50.0]),
                std=np.array([5.0]),
                lower=np.array([40.0]),
                upper=np.array([60.0]),
            )
        }
        surrogate.get_ensemble_r2.return_value = 0.0

        training = pd.DataFrame({"x": np.linspace(0, 10, 50)})
        gate = ExtrapolationGate(
            training, problem.decision_variables, k=3, threshold=100.0,
        )

        config = OptimizerConfig(acquisition="pessimistic", pessimistic_beta=2.0)
        sp = SurroxProblem(problem, surrogate, gate, config, 1e6)

        out: dict = {}
        sp._evaluate({"x": 5.0}, out)
        assert out["F"][0] == pytest.approx(50.0 + 2.0 * 5.0)

    def test_pessimistic_maximize_uses_negated_mean_minus_beta_std(self) -> None:
        problem = ProblemDefinition(
            variables=(
                Variable(name="x", dtype=DType.CONTINUOUS, role=Role.DECISION,
                         bounds=ContinuousBounds(lower=0.0, upper=10.0)),
            ),
            objectives=(
                Objective(name="obj", direction=Direction.MAXIMIZE, column="y"),
            ),
        )

        surrogate = MagicMock()
        surrogate.evaluate_with_uncertainty.return_value = {
            "y": SurrogatePrediction(
                mean=np.array([50.0]),
                std=np.array([5.0]),
                lower=np.array([40.0]),
                upper=np.array([60.0]),
            )
        }
        surrogate.get_ensemble_r2.return_value = 0.0

        training = pd.DataFrame({"x": np.linspace(0, 10, 50)})
        gate = ExtrapolationGate(
            training, problem.decision_variables, k=3, threshold=100.0,
        )

        config = OptimizerConfig(acquisition="pessimistic", pessimistic_beta=2.0)
        sp = SurroxProblem(problem, surrogate, gate, config, 1e6)

        out: dict = {}
        sp._evaluate({"x": 5.0}, out)
        assert out["F"][0] == pytest.approx(-(50.0 - 2.0 * 5.0))

    def test_pessimistic_zero_std_equals_direct(self) -> None:
        problem = ProblemDefinition(
            variables=(
                Variable(name="x", dtype=DType.CONTINUOUS, role=Role.DECISION,
                         bounds=ContinuousBounds(lower=0.0, upper=10.0)),
            ),
            objectives=(
                Objective(name="obj", direction=Direction.MINIMIZE, column="y"),
            ),
        )

        surrogate = MagicMock()
        surrogate.evaluate.return_value = {"y": np.array([42.0])}
        surrogate.evaluate_with_uncertainty.return_value = {
            "y": SurrogatePrediction(
                mean=np.array([42.0]),
                std=np.array([0.0]),
                lower=np.array([42.0]),
                upper=np.array([42.0]),
            )
        }
        surrogate.get_ensemble_r2.return_value = 0.0

        training = pd.DataFrame({"x": np.linspace(0, 10, 50)})
        gate = ExtrapolationGate(
            training, problem.decision_variables, k=3, threshold=100.0,
        )

        config_pessimistic = OptimizerConfig(acquisition="pessimistic", pessimistic_beta=2.0)
        sp_pessimistic = SurroxProblem(problem, surrogate, gate, config_pessimistic, 1e6)
        out_pessimistic: dict = {}
        sp_pessimistic._evaluate({"x": 5.0}, out_pessimistic)

        config_direct = OptimizerConfig(acquisition="direct")
        sp_direct = SurroxProblem(problem, surrogate, gate, config_direct, 1e6)
        out_direct: dict = {}
        sp_direct._evaluate({"x": 5.0}, out_direct)

        assert out_pessimistic["F"][0] == pytest.approx(out_direct["F"][0])

    def test_le_constraint_uses_upper_bound(self) -> None:
        problem = ProblemDefinition(
            variables=(
                Variable(name="x", dtype=DType.CONTINUOUS, role=Role.DECISION,
                         bounds=ContinuousBounds(lower=0.0, upper=10.0)),
            ),
            objectives=(
                Objective(name="obj", direction=Direction.MINIMIZE, column="y"),
            ),
            data_constraints=(
                DataConstraint(name="dc", column="y", operator=ConstraintOperator.LE, limit=100.0),
            ),
        )

        surrogate = MagicMock()
        surrogate.evaluate.return_value = {"y": np.array([90.0])}
        surrogate.evaluate_with_uncertainty.return_value = {
            "y": SurrogatePrediction(
                mean=np.array([90.0]),
                std=np.array([5.0]),
                lower=np.array([80.0]),
                upper=np.array([105.0]),
            )
        }
        surrogate.get_ensemble_r2.return_value = 0.0

        training = pd.DataFrame({"x": np.linspace(0, 10, 50)})
        gate = ExtrapolationGate(
            training, problem.decision_variables, k=3, threshold=100.0,
        )

        sp = SurroxProblem(problem, surrogate, gate, OptimizerConfig(), 1e6)
        out: dict = {}
        sp._evaluate({"x": 5.0}, out)

        assert out["G"][0] == 105.0 - 100.0

    def test_soft_constraint_no_g_value(self) -> None:
        problem = ProblemDefinition(
            variables=(
                Variable(name="x", dtype=DType.CONTINUOUS, role=Role.DECISION,
                         bounds=ContinuousBounds(lower=0.0, upper=10.0)),
            ),
            objectives=(
                Objective(name="obj", direction=Direction.MINIMIZE, column="y"),
            ),
            data_constraints=(
                DataConstraint(
                    name="dc", column="y", operator=ConstraintOperator.LE,
                    limit=100.0, severity=ConstraintSeverity.SOFT, penalty_weight=10.0,
                ),
            ),
        )

        surrogate = MagicMock()
        surrogate.evaluate.return_value = {"y": np.array([90.0])}
        surrogate.evaluate_with_uncertainty.return_value = {
            "y": SurrogatePrediction(
                mean=np.array([90.0]),
                std=np.array([5.0]),
                lower=np.array([80.0]),
                upper=np.array([105.0]),
            )
        }
        surrogate.get_ensemble_r2.return_value = 0.0

        training = pd.DataFrame({"x": np.linspace(0, 10, 50)})
        gate = ExtrapolationGate(
            training, problem.decision_variables, k=3, threshold=100.0,
        )

        config = OptimizerConfig(acquisition="direct")
        sp = SurroxProblem(problem, surrogate, gate, config, 1e6)
        out: dict = {}
        sp._evaluate({"x": 5.0}, out)

        assert "G" not in out
        assert out["F"][0] == 90.0 + 10.0 * (105.0 - 100.0)

    def test_soft_constraint_no_violation_no_penalty(self) -> None:
        problem = ProblemDefinition(
            variables=(
                Variable(name="x", dtype=DType.CONTINUOUS, role=Role.DECISION,
                         bounds=ContinuousBounds(lower=0.0, upper=10.0)),
            ),
            objectives=(
                Objective(name="obj", direction=Direction.MINIMIZE, column="y"),
            ),
            data_constraints=(
                DataConstraint(
                    name="dc", column="y", operator=ConstraintOperator.LE,
                    limit=100.0, severity=ConstraintSeverity.SOFT, penalty_weight=10.0,
                ),
            ),
        )

        surrogate = MagicMock()
        surrogate.evaluate.return_value = {"y": np.array([50.0])}
        surrogate.evaluate_with_uncertainty.return_value = {
            "y": SurrogatePrediction(
                mean=np.array([50.0]),
                std=np.array([5.0]),
                lower=np.array([40.0]),
                upper=np.array([60.0]),
            )
        }
        surrogate.get_ensemble_r2.return_value = 0.0

        training = pd.DataFrame({"x": np.linspace(0, 10, 50)})
        gate = ExtrapolationGate(
            training, problem.decision_variables, k=3, threshold=100.0,
        )

        config = OptimizerConfig(acquisition="direct")
        sp = SurroxProblem(problem, surrogate, gate, config, 1e6)
        out: dict = {}
        sp._evaluate({"x": 5.0}, out)

        assert "G" not in out
        assert out["F"][0] == 50.0

    def test_mixed_hard_soft_constraints(self) -> None:
        problem = ProblemDefinition(
            variables=(
                Variable(name="x", dtype=DType.CONTINUOUS, role=Role.DECISION,
                         bounds=ContinuousBounds(lower=0.0, upper=10.0)),
            ),
            objectives=(
                Objective(name="obj", direction=Direction.MINIMIZE, column="y"),
            ),
            data_constraints=(
                DataConstraint(
                    name="hard_dc", column="y", operator=ConstraintOperator.LE,
                    limit=100.0,
                ),
                DataConstraint(
                    name="soft_dc", column="y", operator=ConstraintOperator.LE,
                    limit=80.0, severity=ConstraintSeverity.SOFT, penalty_weight=5.0,
                ),
            ),
        )

        surrogate = MagicMock()
        surrogate.evaluate.return_value = {"y": np.array([90.0])}
        surrogate.evaluate_with_uncertainty.return_value = {
            "y": SurrogatePrediction(
                mean=np.array([90.0]),
                std=np.array([5.0]),
                lower=np.array([80.0]),
                upper=np.array([105.0]),
            )
        }
        surrogate.get_ensemble_r2.return_value = 0.0

        training = pd.DataFrame({"x": np.linspace(0, 10, 50)})
        gate = ExtrapolationGate(
            training, problem.decision_variables, k=3, threshold=100.0,
        )

        config = OptimizerConfig(acquisition="direct")
        sp = SurroxProblem(problem, surrogate, gate, config, 1e6)
        out: dict = {}
        sp._evaluate({"x": 5.0}, out)

        assert len(out["G"]) == 1
        assert out["G"][0] == 105.0 - 100.0
        soft_violation = 105.0 - 80.0
        assert out["F"][0] == 90.0 + 5.0 * soft_violation

    def test_soft_linear_constraint_penalty(self) -> None:
        problem = ProblemDefinition(
            variables=(
                Variable(name="x", dtype=DType.CONTINUOUS, role=Role.DECISION,
                         bounds=ContinuousBounds(lower=0.0, upper=10.0)),
            ),
            objectives=(
                Objective(name="obj", direction=Direction.MINIMIZE, column="y"),
            ),
            linear_constraints=(
                LinearConstraint(
                    name="soft_lc", coefficients={"x": 1.0},
                    operator=ConstraintOperator.LE, rhs=5.0,
                    severity=ConstraintSeverity.SOFT, penalty_weight=3.0,
                ),
            ),
        )

        surrogate = MagicMock()
        surrogate.evaluate.return_value = {"y": np.array([42.0])}
        surrogate.get_ensemble_r2.return_value = 0.0

        training = pd.DataFrame({"x": np.linspace(0, 10, 50)})
        gate = ExtrapolationGate(
            training, problem.decision_variables, k=3, threshold=100.0,
        )

        config = OptimizerConfig(acquisition="direct")
        sp = SurroxProblem(problem, surrogate, gate, config, 1e6)
        out: dict = {}
        sp._evaluate({"x": 7.0}, out)

        assert "G" not in out
        assert out["F"][0] == 42.0 + 3.0 * 2.0

    def test_adaptive_beta_scales_with_r2(self) -> None:
        problem = ProblemDefinition(
            variables=(
                Variable(name="x", dtype=DType.CONTINUOUS, role=Role.DECISION,
                         bounds=ContinuousBounds(lower=0.0, upper=10.0)),
            ),
            objectives=(
                Objective(name="obj", direction=Direction.MINIMIZE, column="y"),
            ),
        )

        surrogate = MagicMock()
        surrogate.evaluate_with_uncertainty.return_value = {
            "y": SurrogatePrediction(
                mean=np.array([50.0]),
                std=np.array([10.0]),
                lower=np.array([40.0]),
                upper=np.array([60.0]),
            )
        }
        surrogate.get_ensemble_r2.return_value = 0.8

        training = pd.DataFrame({"x": np.linspace(0, 10, 50)})
        gate = ExtrapolationGate(
            training, problem.decision_variables, k=3, threshold=100.0,
        )

        config = OptimizerConfig(acquisition="pessimistic", pessimistic_beta=1.0, min_beta_fraction=0.0)
        sp = SurroxProblem(problem, surrogate, gate, config, 1e6)

        out: dict = {}
        sp._evaluate({"x": 5.0}, out)
        assert out["F"][0] == pytest.approx(50.0 + (1.0 * 0.2) * 10.0)

    def test_adaptive_beta_high_r2_approaches_direct(self) -> None:
        problem = ProblemDefinition(
            variables=(
                Variable(name="x", dtype=DType.CONTINUOUS, role=Role.DECISION,
                         bounds=ContinuousBounds(lower=0.0, upper=10.0)),
            ),
            objectives=(
                Objective(name="obj", direction=Direction.MINIMIZE, column="y"),
            ),
        )

        surrogate = MagicMock()
        surrogate.evaluate_with_uncertainty.return_value = {
            "y": SurrogatePrediction(
                mean=np.array([50.0]),
                std=np.array([10.0]),
                lower=np.array([40.0]),
                upper=np.array([60.0]),
            )
        }
        surrogate.get_ensemble_r2.return_value = 0.99

        training = pd.DataFrame({"x": np.linspace(0, 10, 50)})
        gate = ExtrapolationGate(
            training, problem.decision_variables, k=3, threshold=100.0,
        )

        config = OptimizerConfig(acquisition="pessimistic", pessimistic_beta=1.0, min_beta_fraction=0.0)
        sp = SurroxProblem(problem, surrogate, gate, config, 1e6)

        out: dict = {}
        sp._evaluate({"x": 5.0}, out)
        assert out["F"][0] == pytest.approx(50.0 + 0.01 * 10.0)
