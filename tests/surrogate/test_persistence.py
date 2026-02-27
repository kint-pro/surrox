from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from surrox.surrogate.config import TrainingConfig
from surrox.surrogate.manager import SurrogateManager


@pytest.fixture(scope="module")
def trained_manager() -> SurrogateManager:
    from surrox.problem.constraints import DataConstraint
    from surrox.problem.dataset import BoundDataset
    from surrox.problem.definition import ProblemDefinition
    from surrox.problem.objectives import Objective
    from surrox.problem.types import ConstraintOperator, Direction, DType, Role
    from surrox.problem.variables import ContinuousBounds, Variable

    rng = np.random.default_rng(42)
    n = 600
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)
    cost = 2 * x1 + 3 * x2 + rng.normal(0, 1, n)
    emissions = x1 + x2 + rng.normal(0, 0.5, n)

    problem = ProblemDefinition(
        variables=(
            Variable(name="x1", dtype=DType.CONTINUOUS, role=Role.DECISION,
                     bounds=ContinuousBounds(lower=0.0, upper=10.0)),
            Variable(name="x2", dtype=DType.CONTINUOUS, role=Role.DECISION,
                     bounds=ContinuousBounds(lower=0.0, upper=10.0)),
        ),
        objectives=(
            Objective(name="cost", direction=Direction.MINIMIZE, column="cost"),
        ),
        data_constraints=(
            DataConstraint(name="emission_cap", column="emissions",
                           operator=ConstraintOperator.LE, limit=15.0),
        ),
    )

    df = pd.DataFrame({"x1": x1, "x2": x2, "cost": cost, "emissions": emissions})
    dataset = BoundDataset(problem=problem, dataframe=df)

    config = TrainingConfig(
        n_trials=5,
        cv_folds=3,
        ensemble_size=2,
        min_r2=None,
        random_seed=42,
    )

    return SurrogateManager.train(problem=problem, dataset=dataset, config=config)


@pytest.mark.slow
class TestSurrogateManagerPersistence:
    def test_save_creates_expected_structure(
        self, trained_manager: SurrogateManager, tmp_path: pytest.TempPathFactory,
    ) -> None:
        save_dir = tmp_path / "surrogates"  # type: ignore[operator]
        trained_manager.save(save_dir)

        assert (save_dir / "metadata.json").exists()
        assert (save_dir / "models").is_dir()
        assert (save_dir / "conformal").is_dir()

        assert (save_dir / "conformal" / "cost.npz").exists()
        assert (save_dir / "conformal" / "emissions.npz").exists()

    def test_metadata_contains_versions_and_fingerprint(
        self, trained_manager: SurrogateManager, tmp_path: pytest.TempPathFactory,
    ) -> None:
        import json

        save_dir = tmp_path / "surrogates"  # type: ignore[operator]
        trained_manager.save(save_dir)

        metadata = json.loads((save_dir / "metadata.json").read_text())
        assert "versions" in metadata
        assert "surrox" in metadata["versions"]
        assert "xgboost" in metadata["versions"]
        assert "lightgbm" in metadata["versions"]
        assert "dataset_fingerprint" in metadata
        assert len(metadata["dataset_fingerprint"]) == 64
        assert "training_config" in metadata

    def test_load_reconstructs_manager(
        self, trained_manager: SurrogateManager, tmp_path: pytest.TempPathFactory,
    ) -> None:
        save_dir = tmp_path / "surrogates"  # type: ignore[operator]
        trained_manager.save(save_dir)
        loaded = SurrogateManager.load(save_dir)

        assert loaded.problem == trained_manager.problem
        assert loaded.config.n_trials == trained_manager.config.n_trials
        assert loaded.config.cv_folds == trained_manager.config.cv_folds
        assert loaded.config.ensemble_size == trained_manager.config.ensemble_size
        assert loaded.config.random_seed == trained_manager.config.random_seed
        assert loaded.config.min_r2 == trained_manager.config.min_r2
        assert loaded.dataset_fingerprint == trained_manager.dataset_fingerprint
        assert set(loaded._surrogates.keys()) == set(trained_manager._surrogates.keys())

    def test_loaded_predictions_match_original(
        self, trained_manager: SurrogateManager, tmp_path: pytest.TempPathFactory,
    ) -> None:
        save_dir = tmp_path / "surrogates"  # type: ignore[operator]
        trained_manager.save(save_dir)
        loaded = SurrogateManager.load(save_dir)

        rng = np.random.default_rng(99)
        X_test = pd.DataFrame({
            "x1": rng.uniform(0, 10, 20),
            "x2": rng.uniform(0, 10, 20),
        })

        original_preds = trained_manager.evaluate(X_test)
        loaded_preds = loaded.evaluate(X_test)

        for column in original_preds:
            np.testing.assert_array_almost_equal(
                loaded_preds[column], original_preds[column],
                decimal=5,
            )

    def test_loaded_uncertainty_works(
        self, trained_manager: SurrogateManager, tmp_path: pytest.TempPathFactory,
    ) -> None:
        save_dir = tmp_path / "surrogates"  # type: ignore[operator]
        trained_manager.save(save_dir)
        loaded = SurrogateManager.load(save_dir)

        rng = np.random.default_rng(99)
        X_test = pd.DataFrame({
            "x1": rng.uniform(0, 10, 20),
            "x2": rng.uniform(0, 10, 20),
        })

        results = loaded.evaluate_with_uncertainty(X_test)
        for column in results:
            assert len(results[column].mean) == 20
            assert len(results[column].lower) == 20
            assert len(results[column].upper) == 20

    def test_trial_history_preserved(
        self, trained_manager: SurrogateManager, tmp_path: pytest.TempPathFactory,
    ) -> None:
        save_dir = tmp_path / "surrogates"  # type: ignore[operator]
        trained_manager.save(save_dir)
        loaded = SurrogateManager.load(save_dir)

        for column in trained_manager._surrogates:
            original_history = trained_manager.get_trial_history(column)
            loaded_history = loaded.get_trial_history(column)
            assert len(loaded_history) == len(original_history)
            for orig, load in zip(original_history, loaded_history, strict=True):
                assert orig.trial_number == load.trial_number
                assert orig.estimator_family == load.estimator_family
                assert orig.mean_r2 == pytest.approx(load.mean_r2)

    def test_ensemble_metadata_preserved(
        self, trained_manager: SurrogateManager, tmp_path: pytest.TempPathFactory,
    ) -> None:
        save_dir = tmp_path / "surrogates"  # type: ignore[operator]
        trained_manager.save(save_dir)
        loaded = SurrogateManager.load(save_dir)

        for column in trained_manager._surrogates:
            orig_ensemble = trained_manager.get_ensemble(column)
            loaded_ensemble = loaded.get_ensemble(column)
            assert orig_ensemble.feature_names == loaded_ensemble.feature_names
            assert (
                orig_ensemble.monotonic_constraints
                == loaded_ensemble.monotonic_constraints
            )
            assert len(orig_ensemble.members) == len(loaded_ensemble.members)
            for orig_m, loaded_m in zip(
                orig_ensemble.members, loaded_ensemble.members, strict=True
            ):
                assert orig_m.weight == pytest.approx(loaded_m.weight)
                assert orig_m.estimator_family == loaded_m.estimator_family

    def test_load_nonexistent_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        from surrox.exceptions import SurroxError

        with pytest.raises(SurroxError, match="metadata.json not found"):
            SurrogateManager.load(tmp_path / "nonexistent")  # type: ignore[operator]
