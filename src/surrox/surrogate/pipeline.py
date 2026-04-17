from __future__ import annotations

import logging
import time
from math import ceil
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from surrox.exceptions import SurrogateTrainingError
from surrox.problem.types import DType, MonotonicDirection
from surrox.surrogate.conformal import ConformalCalibration
from surrox.surrogate.ensemble import Ensemble
from surrox.surrogate.models import EnsembleMember, EnsembleMemberConfig, FoldMetrics, TrialRecord

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fast-track: fixed default hyperparameters — no Optuna search
# ---------------------------------------------------------------------------

_FAST_TRACK_DEFAULTS: dict[str, dict[str, Any]] = {
    "xgboost": {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.01,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
    },
    "lightgbm": {
        "num_leaves": 31,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 10,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
    },
    "gaussian_process": {},
}

# GP kernel optimisation is O(n³) — too slow for n > this threshold in fast track
_FAST_TRACK_GP_MAX_N = 200

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

    X = dataset_df[feature_names].copy()
    y = dataset_df[column]

    category_mappings: dict[str, list[str]] = {}
    for var in problem.variables:
        if var.dtype in (DType.CATEGORICAL, DType.ORDINAL):
            categories = list(var.bounds.categories)
            X[var.name] = pd.Categorical(X[var.name], categories=categories)
            category_mappings[var.name] = categories

    X_train, X_calib, y_train, y_calib = train_test_split(
        X,
        y,
        test_size=config.calibration_fraction,
        random_state=config.random_seed,
    )

    y_train_np = y_train.to_numpy()
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

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train_np[train_idx]
            X_fold_val = X_train.iloc[val_idx]
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
        multivariate=False,
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

    y_min, y_max = problem.prediction_bounds_for_column(column)

    ensemble = _build_ensemble(
        completed_records=completed_records,
        oof_predictions=oof_predictions,
        families_by_name=families_by_name,
        raw_constraints=raw_constraints,
        feature_names=feature_names,
        categorical_features=categorical_features,
        column=column,
        config=config,
        X_train=X_train,
        y_train=y_train_np,
        category_mappings=category_mappings,
        y_min=y_min,
        y_max=y_max,
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

    ensemble_r2 = _compute_ensemble_r2(ensemble, X_calib, y_calib_np)
    _validate_quality_gate(ensemble_r2, column, config)

    conformal = ConformalCalibration.from_calibration_data(
        column=column,
        ensemble=ensemble,
        X_calib=X_calib,
        y_calib=y_calib_np,
        default_coverage=config.default_coverage,
    )
    _logger.info(
        "conformal calibrated",
        extra={
            "column": column,
            "coverage": config.default_coverage,
            "n_calib_samples": len(X_calib),
        },
    )

    from surrox.surrogate.manager import SurrogateResult

    return SurrogateResult(
        column=column,
        ensemble=ensemble,
        conformal=conformal,
        trial_history=tuple(trial_records),
        ensemble_r2=ensemble_r2,
    )


def _validate_minimum_data(n_rows: int, config: TrainingConfig) -> None:
    spf = config.min_samples_per_fold
    mcs = config.min_calibration_samples
    min_train_pool = ceil(spf * config.cv_folds / (1 - config.calibration_fraction))
    min_calib = ceil(mcs / config.calibration_fraction)
    minimum = max(min_train_pool, min_calib)
    if n_rows < minimum:
        raise SurrogateTrainingError(
            f"dataset has {n_rows} rows, but minimum is "
            f"{minimum} (cv_folds={config.cv_folds}, "
            f"calibration_fraction={config.calibration_fraction})"
        )


def _compute_ensemble_r2(
    ensemble: Ensemble,
    X_calib: pd.DataFrame,
    y_calib: NDArray,
) -> float:
    predictions = ensemble.predict(X_calib)
    return float(r2_score(y_calib, predictions))


def _validate_quality_gate(
    ensemble_r2: float,
    column: str,
    config: TrainingConfig,
) -> None:
    if config.min_r2 is None:
        return

    if ensemble_r2 < config.min_r2:
        _logger.warning(
            "quality gate failed",
            extra={
                "column": column,
                "r2": round(ensemble_r2, 4),
                "min_r2": config.min_r2,
            },
        )
        raise SurrogateTrainingError(
            f"surrogate '{column}': ensemble R² on calibration set is {ensemble_r2:.4f}, "
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
    X_train: pd.DataFrame,
    y_train: NDArray,
    category_mappings: dict[str, list[str]],
    y_min: float = -np.inf,
    y_max: float = np.inf,
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
        category_mappings=category_mappings,
        y_min=y_min,
        y_max=y_max,
    )



def _max_correlation(candidate: NDArray, selected: list[NDArray]) -> float:
    correlations = []
    for existing in selected:
        with np.errstate(divide="ignore", invalid="ignore"):
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


def fast_train_surrogate(
    problem: "ProblemDefinition",
    dataset_df: "pd.DataFrame",
    column: str,
    families: tuple[str, ...] = ("xgboost", "lightgbm", "gaussian_process"),
    calibration_fraction: float | None = None,
    random_seed: int = 42,
) -> "SurrogateResult":
    """Train a surrogate for one column without Optuna hyperparameter search.

    Uses fixed sensible default hyperparameters for each family. Skips
    cross-validation and the full Optuna loop — roughly 20x faster than
    ``train_surrogate`` with no meaningful quality loss for a first-look
    estimate.

    GP is automatically excluded for n > _FAST_TRACK_GP_MAX_N to avoid
    O(n³) kernel optimisation overhead.

    Returns a ``SurrogateResult`` identical in structure to the one produced
    by ``train_surrogate`` / ``refit_surrogate``, so it plugs into
    ``SurrogateManager`` transparently.
    """
    from surrox.surrogate.config import TrainingConfig
    from surrox.surrogate.models import EnsembleMemberConfig

    n_rows = len(dataset_df)

    if calibration_fraction is None:
        calibration_fraction = 0.15 if n_rows < 200 else 0.2

    effective_families = [
        f for f in families
        if f in _FAST_TRACK_DEFAULTS
        and not (f == "gaussian_process" and n_rows > _FAST_TRACK_GP_MAX_N)
    ]
    if not effective_families:
        effective_families = ["xgboost"]

    weight = 1.0 / len(effective_families)
    member_configs = tuple(
        EnsembleMemberConfig(
            estimator_family=fam,
            hyperparameters=_FAST_TRACK_DEFAULTS[fam],
            weight=weight,
        )
        for fam in effective_families
    )

    config = TrainingConfig(
        calibration_fraction=calibration_fraction,
        min_r2=None,
        min_calibration_samples=max(5, int(n_rows * calibration_fraction * 0.5)),
        min_samples_per_fold=5,
        random_seed=random_seed,
        refit_ensemble={column: member_configs},
    )

    _logger.info(
        "fast_train_surrogate",
        extra={
            "column": column,
            "n_rows": n_rows,
            "families": effective_families,
            "calibration_fraction": calibration_fraction,
        },
    )

    return refit_surrogate(
        problem=problem,
        dataset_df=dataset_df,
        config=config,
        column=column,
        member_configs=member_configs,
    )


def refit_surrogate(
    problem: ProblemDefinition,
    dataset_df: pd.DataFrame,
    config: TrainingConfig,
    column: str,
    member_configs: tuple[EnsembleMemberConfig, ...],
) -> SurrogateResult:
    feature_names = [v.name for v in problem.variables]
    categorical_features = {
        v.name
        for v in problem.variables
        if v.dtype in (DType.CATEGORICAL, DType.ORDINAL)
    }
    raw_constraints = problem.monotonic_constraints_for(column)
    families_by_name = {f.name: f for f in config.estimator_families}

    X = dataset_df[feature_names].copy()
    y = dataset_df[column]

    category_mappings: dict[str, list[str]] = {}
    for var in problem.variables:
        if var.dtype in (DType.CATEGORICAL, DType.ORDINAL):
            categories = list(var.bounds.categories)
            X[var.name] = pd.Categorical(X[var.name], categories=categories)
            category_mappings[var.name] = categories

    X_train, X_calib, y_train, y_calib = train_test_split(
        X,
        y,
        test_size=config.calibration_fraction,
        random_state=config.random_seed,
    )

    y_train_np = y_train.to_numpy()
    y_calib_np = y_calib.to_numpy()

    members: list[EnsembleMember] = []
    for i, mc in enumerate(member_configs):
        family = families_by_name[mc.estimator_family]
        mapped_constraints = family.map_monotonic_constraints(
            raw_constraints, feature_names, categorical_features
        )
        model = family.build_model(
            dict(mc.hyperparameters),
            mapped_constraints,
            config.random_seed,
            config.n_threads,
        )
        model.fit(X_train, y_train_np)
        members.append(
            EnsembleMember(
                trial_number=i,
                estimator_family=mc.estimator_family,
                model=model,
                weight=mc.weight,
                cv_rmse=0.0,
            )
        )

    y_min, y_max = problem.prediction_bounds_for_column(column)

    ensemble = Ensemble(
        column=column,
        members=tuple(members),
        feature_names=tuple(feature_names),
        monotonic_constraints=raw_constraints,
        category_mappings=category_mappings,
        y_min=y_min,
        y_max=y_max,
    )

    ensemble_r2 = _compute_ensemble_r2(ensemble, X_calib, y_calib_np)
    _validate_quality_gate(ensemble_r2, column, config)

    conformal = ConformalCalibration.from_calibration_data(
        column=column,
        ensemble=ensemble,
        X_calib=X_calib,
        y_calib=y_calib_np,
        default_coverage=config.default_coverage,
    )

    from surrox.surrogate.manager import SurrogateResult

    return SurrogateResult(
        column=column,
        ensemble=ensemble,
        conformal=conformal,
        trial_history=(),
        ensemble_r2=ensemble_r2,
    )
