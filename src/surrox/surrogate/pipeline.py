from __future__ import annotations

import logging
import time
from math import ceil
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from surrox.exceptions import SurrogateTrainingError
from surrox.problem.types import DType, MonotonicDirection
from surrox.surrogate.conformal import ConformalCalibration
from surrox.surrogate.ensemble import Ensemble
from surrox.surrogate.models import EnsembleMember, FoldMetrics, TrialRecord

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

    from surrox.problem.definition import ProblemDefinition
    from surrox.surrogate.config import TrainingConfig
    from surrox.surrogate.manager import SurrogateResult
    from surrox.surrogate.protocol import EstimatorFamily


def train_surrogate(
    problem: ProblemDefinition,
    dataset_df: pd.DataFrame,
    config: TrainingConfig,
    column: str,
) -> SurrogateResult:
    feature_names = [v.name for v in problem.variables]
    categorical_features = {
        v.name
        for v in problem.variables
        if v.dtype in (DType.CATEGORICAL, DType.ORDINAL)
    }
    raw_constraints = problem.monotonic_constraints_for(column)
    families_by_name = {f.name: f for f in config.estimator_families}

    _validate_minimum_data(len(dataset_df), config)

    X = dataset_df[feature_names]
    y = dataset_df[column]

    X_train, X_calib, y_train, y_calib = train_test_split(
        X,
        y,
        test_size=config.calibration_fraction,
        random_state=config.random_seed,
    )

    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()
    X_calib_np = X_calib.to_numpy()
    y_calib_np = y_calib.to_numpy()

    oof_predictions: dict[int, NDArray] = {}
    trial_records: list[TrialRecord] = []

    def objective(trial: optuna.Trial) -> float:
        family_name = trial.suggest_categorical(
            "estimator_family", list(families_by_name.keys())
        )
        family = families_by_name[family_name]

        hyperparameters = family.suggest_hyperparameters(trial)
        mapped_constraints = family.map_monotonic_constraints(
            raw_constraints, feature_names, categorical_features
        )

        kf = KFold(
            n_splits=config.cv_folds,
            shuffle=True,
            random_state=config.random_seed,
        )

        fold_metrics_list: list[FoldMetrics] = []
        oof = np.empty(len(y_train_np))

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_np)):
            X_fold_train = X_train_np[train_idx]
            y_fold_train = y_train_np[train_idx]
            X_fold_val = X_train_np[val_idx]
            y_fold_val = y_train_np[val_idx]

            model = family.build_model(
                hyperparameters,
                mapped_constraints,
                config.random_seed,
                config.n_threads,
            )

            t_start = time.perf_counter()
            model.fit(X_fold_train, y_fold_train)
            training_time_s = time.perf_counter() - t_start

            t_start = time.perf_counter()
            predictions = model.predict(X_fold_val)
            inference_time_ms = (time.perf_counter() - t_start) * 1000

            oof[val_idx] = predictions

            r2 = r2_score(y_fold_val, predictions)
            rmse = float(np.sqrt(mean_squared_error(y_fold_val, predictions)))
            mae = float(mean_absolute_error(y_fold_val, predictions))

            fold_metrics_list.append(
                FoldMetrics(
                    fold=fold_idx,
                    r2=r2,
                    rmse=rmse,
                    mae=mae,
                    training_time_s=training_time_s,
                    inference_time_ms=inference_time_ms,
                )
            )

            cumulative_rmse = float(np.mean([fm.rmse for fm in fold_metrics_list]))
            trial.report(cumulative_rmse, fold_idx)
            if trial.should_prune():
                record = _build_trial_record(
                    trial.number,
                    family_name,
                    hyperparameters,
                    fold_metrics_list,
                    "pruned",
                )
                trial_records.append(record)
                raise optuna.TrialPruned()

        oof_predictions[trial.number] = oof
        record = _build_trial_record(
            trial.number, family_name, hyperparameters, fold_metrics_list, "completed"
        )
        trial_records.append(record)
        _logger.debug(
            "trial complete",
            extra={
                "column": column,
                "trial_number": trial.number,
                "family": family_name,
                "mean_r2": round(record.mean_r2, 4),
                "mean_rmse": round(record.mean_rmse, 4),
            },
        )
        return record.mean_rmse

    optuna_logger = logging.getLogger("optuna")
    original_level = optuna_logger.level
    optuna_logger.setLevel(logging.WARNING)
    sampler = optuna.samplers.TPESampler(
        seed=config.random_seed,
        multivariate=True,
        n_startup_trials=min(20, config.n_trials),
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=min(10, config.n_trials),
        n_warmup_steps=min(2, config.cv_folds - 1),
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    try:
        study.optimize(
            objective,
            n_trials=config.n_trials,
            timeout=config.study_timeout_s,
        )
    finally:
        optuna_logger.setLevel(original_level)

    completed_records = [r for r in trial_records if r.status == "completed"]
    if not completed_records:
        n = len(trial_records)
        raise SurrogateTrainingError(
            f"surrogate '{column}': all {n} trials were "
            f"pruned or failed — no completed trials"
        )

    ensemble = _build_ensemble(
        completed_records=completed_records,
        oof_predictions=oof_predictions,
        families_by_name=families_by_name,
        raw_constraints=raw_constraints,
        feature_names=feature_names,
        categorical_features=categorical_features,
        column=column,
        config=config,
        X_train=X_train_np,
        y_train=y_train_np,
    )

    _logger.info(
        "ensemble built",
        extra={
            "column": column,
            "ensemble_size": len(ensemble.members),
            "best_cv_rmse": round(
                min(r.mean_rmse for r in completed_records), 4
            ),
        },
    )

    _validate_quality_gate(ensemble, X_calib_np, y_calib_np, column, config)

    conformal = ConformalCalibration.from_calibration_data(
        column=column,
        ensemble=ensemble,
        X_calib=X_calib_np,
        y_calib=y_calib_np,
        default_coverage=config.default_coverage,
    )
    _logger.info(
        "conformal calibrated",
        extra={
            "column": column,
            "coverage": config.default_coverage,
            "n_calib_samples": len(X_calib_np),
        },
    )

    from surrox.surrogate.manager import SurrogateResult

    return SurrogateResult(
        column=column,
        ensemble=ensemble,
        conformal=conformal,
        trial_history=tuple(trial_records),
    )


def _validate_minimum_data(n_rows: int, config: TrainingConfig) -> None:
    min_train_pool = ceil(50 * config.cv_folds / (1 - config.calibration_fraction))
    min_calib = ceil(100 / config.calibration_fraction)
    minimum = max(min_train_pool, min_calib)
    if n_rows < minimum:
        cal = config.calibration_fraction
        raise SurrogateTrainingError(
            f"dataset has {n_rows} rows, but minimum is "
            f"{minimum} (cv_folds={config.cv_folds}, "
            f"calibration_fraction={cal})"
        )


def _validate_quality_gate(
    ensemble: Ensemble,
    X_calib: NDArray,
    y_calib: NDArray,
    column: str,
    config: TrainingConfig,
) -> None:
    if config.min_r2 is None:
        return
    import pandas as pd

    X_calib_df = pd.DataFrame(X_calib, columns=list(ensemble.feature_names))
    predictions = ensemble.predict(X_calib_df)
    r2 = r2_score(y_calib, predictions)
    if r2 < config.min_r2:
        _logger.warning(
            "quality gate failed",
            extra={
                "column": column,
                "r2": round(float(r2), 4),
                "min_r2": config.min_r2,
            },
        )
        raise SurrogateTrainingError(
            f"surrogate '{column}': ensemble R² on calibration set is {r2:.4f}, "
            f"below minimum threshold {config.min_r2} — surrogate quality insufficient"
        )


def _build_trial_record(
    trial_number: int,
    family_name: str,
    hyperparameters: dict[str, Any],
    fold_metrics_list: list[FoldMetrics],
    status: str,
) -> TrialRecord:
    return TrialRecord(
        trial_number=trial_number,
        estimator_family=family_name,
        hyperparameters=hyperparameters,
        fold_metrics=tuple(fold_metrics_list),
        mean_r2=float(np.mean([fm.r2 for fm in fold_metrics_list])),
        mean_rmse=float(np.mean([fm.rmse for fm in fold_metrics_list])),
        mean_mae=float(np.mean([fm.mae for fm in fold_metrics_list])),
        mean_training_time_s=float(
            np.mean([fm.training_time_s for fm in fold_metrics_list])
        ),
        mean_inference_time_ms=float(
            np.mean([fm.inference_time_ms for fm in fold_metrics_list])
        ),
        status=status,
    )


def _build_ensemble(
    completed_records: list[TrialRecord],
    oof_predictions: dict[int, NDArray],
    families_by_name: dict[str, EstimatorFamily],
    raw_constraints: dict[str, MonotonicDirection],
    feature_names: list[str],
    categorical_features: set[str],
    column: str,
    config: TrainingConfig,
    X_train: NDArray,
    y_train: NDArray,
) -> Ensemble:
    sorted_records = sorted(completed_records, key=lambda r: r.mean_rmse)

    selected: list[TrialRecord] = []
    selected_oof: list[NDArray] = []

    for record in sorted_records:
        if len(selected) >= config.ensemble_size:
            break

        candidate_oof = oof_predictions[record.trial_number]

        if selected_oof:
            max_corr = _max_correlation(candidate_oof, selected_oof)
            if max_corr >= config.diversity_threshold:
                continue

        selected.append(record)
        selected_oof.append(candidate_oof)

    rmse_values = np.array([r.mean_rmse for r in selected])
    weights = _softmax(-rmse_values, config.softmax_temperature)

    members: list[EnsembleMember] = []
    for record, weight in zip(selected, weights, strict=False):
        family = families_by_name[record.estimator_family]
        mapped_constraints = family.map_monotonic_constraints(
            raw_constraints, feature_names, categorical_features
        )
        model = family.build_model(
            record.hyperparameters,
            mapped_constraints,
            config.random_seed,
            config.n_threads,
        )
        model.fit(X_train, y_train)

        members.append(
            EnsembleMember(
                trial_number=record.trial_number,
                estimator_family=record.estimator_family,
                model=model,
                weight=float(weight),
                cv_rmse=record.mean_rmse,
            )
        )

    return Ensemble(
        column=column,
        members=tuple(members),
        feature_names=tuple(feature_names),
        monotonic_constraints=raw_constraints,
    )


def _max_correlation(candidate: NDArray, selected: list[NDArray]) -> float:
    correlations = []
    for existing in selected:
        corr = np.corrcoef(candidate, existing)[0, 1]
        if np.isnan(corr):
            return 1.0
        correlations.append(float(corr))
    return max(correlations)


def _softmax(values: NDArray, temperature: float) -> NDArray:
    scaled = values / temperature
    scaled -= scaled.max()
    exp_values = np.exp(scaled)
    return exp_values / exp_values.sum()
