"""Development-only model evaluation on fixed expanding chronological folds.

Run from the repository root: .venv/bin/python -m src.evaluate
No final-validation data or production models are used by this experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .features import CATEGORICAL_FEATURE_COLUMNS, FEATURE_COLUMNS, build_features, build_model_features
from .preprocessing import build_sklearn_preprocessor

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ChronologicalFold:
    name: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str


FOLDS = (
    ChronologicalFold("fold_1", "2025-01-01", "2025-04-30", "2025-05-01", "2025-06-30"),
    ChronologicalFold("fold_2", "2025-01-01", "2025-06-30", "2025-07-01", "2025-08-31"),
    ChronologicalFold("fold_3", "2025-01-01", "2025-08-31", "2025-09-01", "2025-10-31"),
)
FEATURE_SETS = {
    "training_fold_median": [],
    "distance_only": ["distance"],
    "all_26_candidates": FEATURE_COLUMNS,
}
CATBOOST_CONFIGURATIONS = {
    "catboost_d7_lr003_l2_3": {"depth": 7, "learning_rate": 0.03, "l2_leaf_reg": 3},
    "catboost_d6_lr005_l2_5": {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5},
    "catboost_d8_lr004_l2_7": {"depth": 8, "learning_rate": 0.04, "l2_leaf_reg": 7},
}
CATBOOST_COMMON_PARAMETERS = {
    "loss_function": "MAE", "eval_metric": "MAE", "iterations": 1500,
    "random_seed": 42, "verbose": False, "thread_count": 4,
    "allow_writing_files": False, "nan_mode": "Min",
}


def chronological_splits(frame: pd.DataFrame, folds=FOLDS):
    """Yield positional indices; every date belongs wholly to a single side."""
    dates = pd.to_datetime(frame["date"], format="ISO8601", errors="raise")
    if dates.isna().any():
        raise ValueError("Chronological split dates must not be missing")
    if dates.dt.tz is not None or not dates.equals(dates.dt.normalize()):
        raise ValueError("Chronological splits require timezone-naive calendar dates")
    for fold in folds:
        if not (fold.train_start <= fold.train_end < fold.validation_start <= fold.validation_end):
            raise ValueError(f"Invalid or overlapping fold boundaries: {fold.name}")
        train_mask = dates.between(fold.train_start, fold.train_end)
        validation_mask = dates.between(fold.validation_start, fold.validation_end)
        train_idx = np.flatnonzero(train_mask.to_numpy())
        validation_idx = np.flatnonzero(validation_mask.to_numpy())
        if not len(train_idx) or not len(validation_idx):
            raise ValueError(f"Empty training or validation period: {fold.name}")
        if dates.iloc[train_idx].max() >= dates.iloc[validation_idx].min():
            raise AssertionError(f"Future observations crossed fold boundaries: {fold.name}")
        yield fold, train_idx, validation_idx


def make_baselines():
    """Fresh estimators for each fold, with fixed, untuned configurations."""
    distance_preprocessor = ColumnTransformer([
        ("distance", SimpleImputer(strategy="median", keep_empty_features=True), ["distance"]),
    ], remainder="drop")
    return {
        "global_median": (
            DummyRegressor(strategy="median"), "training_fold_median", {"strategy": "median"},
        ),
        "distance_linear": (
            Pipeline([("distance", distance_preprocessor), ("regressor", LinearRegression())]),
            "distance_only", {"fit_intercept": True},
        ),
        "ridge": (
            Pipeline([
                ("preprocessing", build_sklearn_preprocessor()),
                ("regressor", Ridge(alpha=1.0, solver="lsqr", tol=1e-6, max_iter=10000, random_state=42)),
            ]),
            "all_26_candidates",
            {"alpha": 1.0, "solver": "lsqr", "tol": 1e-6, "max_iter": 10000, "random_state": 42},
        ),
    }


def regression_metrics(target, predictions) -> dict:
    values = np.asarray(predictions, dtype=float)
    actual = np.asarray(target, dtype=float)
    if values.shape != actual.shape or not np.isfinite(values).all() or not np.isfinite(actual).all():
        raise ValueError("Metrics require matching, finite targets and predictions")
    return {
        "MAE": float(mean_absolute_error(actual, values)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, values))),
        "R2": float(r2_score(actual, values)),
    }


def development_features_and_target(frame: pd.DataFrame):
    """Validate labeled targets and build stateless, whitelisted features."""
    if "posted_rate" not in frame:
        raise ValueError("Labeled development data must contain posted_rate")
    target = pd.to_numeric(frame["posted_rate"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError("Development targets must be finite and strictly positive")
    # Only deterministic row-wise transformations precede splitting.
    return build_features(frame), target


def run_baselines(frame: pd.DataFrame, folds=FOLDS) -> pd.DataFrame:
    """Fit preprocessing/estimators on each fold's training rows only."""
    features, target = development_features_and_target(frame)
    rows = []
    for fold, train_idx, validation_idx in chronological_splits(frame, folds):
        X_train, X_validation = features.iloc[train_idx], features.iloc[validation_idx]
        y_train, y_validation = target[train_idx], target[validation_idx]
        for name, (estimator, feature_set, parameters) in make_baselines().items():
            start = perf_counter()
            if name == "global_median":
                # Constant placeholders make the absence of predictive inputs explicit.
                estimator.fit(np.zeros((len(train_idx), 1)), y_train)
                predictions = estimator.predict(np.zeros((len(validation_idx), 1)))
            else:
                estimator.fit(X_train, y_train)
                predictions = estimator.predict(X_validation)
            elapsed = perf_counter() - start
            metrics = regression_metrics(y_validation, predictions)
            rows.append({
                "model": name, "fold": fold.name,
                "train_start": fold.train_start, "train_end": fold.train_end,
                "validation_start": fold.validation_start, "validation_end": fold.validation_end,
                **metrics, "feature_set": feature_set,
                "train_rows": len(train_idx), "validation_rows": len(validation_idx),
                "training_and_prediction_seconds": elapsed,
                "nonpositive_predictions": int((predictions <= 0).sum()),
                "parameters": json.dumps(parameters, sort_keys=True),
            })
            print(f"{name:16s} {fold.name}: MAE={metrics['MAE']:.2f} RMSE={metrics['RMSE']:.2f} R2={metrics['R2']:.6f}", flush=True)
    return pd.DataFrame(rows)


def run_catboost(frame: pd.DataFrame, folds=FOLDS, configurations=None,
                 early_stopping_rounds: int = 100, feature_columns=None,
                 feature_set="all_26_candidates", target_strategy="direct",
                 fixed_iterations=None, model_callback=None) -> pd.DataFrame:
    """Evaluate three fixed candidates using native categoricals and numeric NaN.

    Each outer chronological validation window is also the early-stopping
    eval_set. Its labels choose the best iteration, so reported scores are
    selection estimates, not scores from an untouched independent test set.
    """
    features, target = development_features_and_target(frame)
    if feature_columns is not None:
        features = build_model_features(frame, feature_columns)
    if target_strategy not in ("direct", "log1p"):
        raise ValueError("Unknown target strategy")
    transformed_target = np.log1p(target) if target_strategy == "log1p" else target
    configurations = CATBOOST_CONFIGURATIONS if configurations is None else configurations
    if early_stopping_rounds < 1:
        raise ValueError("early_stopping_rounds must be positive")
    rows = []
    diagnostics = []
    for fold, train_idx, validation_idx in chronological_splits(frame, folds):
        X_train, X_validation = features.iloc[train_idx], features.iloc[validation_idx]
        y_train, y_validation = target[train_idx], target[validation_idx]
        for name, overrides in configurations.items():
            parameters = {**CATBOOST_COMMON_PARAMETERS, **overrides}
            if target_strategy == "log1p":
                parameters["eval_metric"] = DollarMAE()
            if fixed_iterations is not None:
                parameters["iterations"] = int(fixed_iterations)
            estimator = CatBoostRegressor(**parameters)
            print(f"Training {name} {fold.name}: {len(train_idx):,} training / {len(validation_idx):,} validation rows", flush=True)
            start = perf_counter()
            categorical = [column for column in CATEGORICAL_FEATURE_COLUMNS if column in features]
            fit_options = {"cat_features": categorical}
            if fixed_iterations is None:
                fit_options.update(eval_set=(X_validation, transformed_target[validation_idx]),
                                   early_stopping_rounds=early_stopping_rounds, use_best_model=True)
            else:
                fit_options["use_best_model"] = False
            estimator.fit(X_train, transformed_target[train_idx], **fit_options)
            training_seconds = perf_counter() - start
            predict_start = perf_counter()
            predictions = inverse_target(estimator.predict(X_validation), target_strategy)
            prediction_seconds = perf_counter() - predict_start
            metrics = regression_metrics(y_validation, predictions)
            best_iteration = int(estimator.get_best_iteration()) if fixed_iterations is None else None
            metric_key = "DollarMAE" if target_strategy == "log1p" else "MAE"
            iterations_executed = (len(estimator.get_evals_result()["validation"][metric_key])
                                   if fixed_iterations is None else estimator.tree_count_)
            if fixed_iterations is None and not 0 <= best_iteration < iterations_executed <= parameters["iterations"]:
                raise AssertionError("Invalid CatBoost iteration metadata")
            recorded_parameters = {
                **parameters, "eval_metric": metric_key,
                "early_stopping_rounds": early_stopping_rounds if fixed_iterations is None else None,
                "use_best_model": fixed_iterations is None, "cat_features": categorical,
            }
            rows.append({
                "model": name, "fold": fold.name,
                "train_start": fold.train_start, "train_end": fold.train_end,
                "validation_start": fold.validation_start, "validation_end": fold.validation_end,
                **metrics, "feature_set": feature_set, "target_strategy": target_strategy,
                "feature_columns": json.dumps(features.columns.tolist()),
                "train_rows": len(train_idx), "validation_rows": len(validation_idx),
                "training_and_prediction_seconds": training_seconds + prediction_seconds,
                "training_seconds": training_seconds, "prediction_seconds": prediction_seconds,
                "best_iteration": best_iteration, "tree_count": estimator.tree_count_,
                "iterations_executed": iterations_executed,
                "nonpositive_predictions": int((predictions <= 0).sum()),
                "parameters": json.dumps(recorded_parameters, sort_keys=True),
            })
            all_features = build_features(frame)
            validation_features = all_features.iloc[validation_idx]
            train_features = all_features.iloc[train_idx]
            masks = {
                "missing_weight": frame.iloc[validation_idx].weight.isna().to_numpy(),
                "negative_weight": frame.iloc[validation_idx].weight.lt(0).to_numpy(),
                "missing_market_index": validation_features.market_index.isna().to_numpy(),
                "unseen_route": (~validation_features.route.isin(train_features.route)).to_numpy(),
                "unseen_city": (~validation_features.pickup.isin(train_features.pickup)
                                | ~validation_features.delivery.isin(train_features.delivery)).to_numpy(),
            }
            for subset, mask in masks.items():
                n = int(mask.sum())
                subset_metrics = regression_metrics(y_validation[mask], predictions[mask]) if n > 1 else {"MAE": None, "RMSE": None, "R2": None}
                diagnostics.append({"model": name, "fold": fold.name, "subset": subset,
                                    "rows": n, **subset_metrics})
            if model_callback is not None:
                model_callback(estimator, fold, frame.iloc[validation_idx])
            print(f"{name} {fold.name}: MAE={metrics['MAE']:.2f} RMSE={metrics['RMSE']:.2f} R2={metrics['R2']:.6f}; best_iteration={best_iteration}, trees={estimator.tree_count_}, training={training_seconds:.1f}s", flush=True)
    result = pd.DataFrame(rows)
    # Keep attrs serializable: DataFrame-valued attrs break pandas concat equality.
    result.attrs["diagnostics"] = diagnostics
    return result


def inverse_target(predictions, strategy: str):
    if strategy not in ("direct", "log1p"):
        raise ValueError("Unknown target strategy")
    values = np.asarray(predictions, dtype=float)
    with np.errstate(over="raise", invalid="raise"):
        return np.expm1(values) if strategy == "log1p" else values


class DollarMAE:
    """CatBoost stopping metric: log-target predictions evaluated in dollars."""

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        error, total = 0.0, 0.0
        for index in range(len(target)):
            row_weight = 1.0 if weight is None else weight[index]
            error += row_weight * abs(math.expm1(approxes[0][index]) - math.expm1(target[index]))
            total += row_weight
        return error, total

    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)


def merge_results(previous: pd.DataFrame, executed: pd.DataFrame) -> pd.DataFrame:
    """Replace rerun model entries while retaining prior baseline experiments."""
    keep = previous.loc[~previous.model.isin(executed.model.unique())].copy()
    executed = executed.copy()
    for column in ("best_iteration", "tree_count", "iterations_executed"):
        if column in keep or column in executed:
            for frame in (keep, executed):
                frame[column] = (frame[column].astype("Int64") if column in frame
                                 else pd.Series(pd.NA, index=frame.index, dtype="Int64"))
    result = pd.concat([keep, executed], ignore_index=True)
    if result.duplicated(["model", "fold"]).any():
        raise ValueError("Duplicate model/fold records in comparison results")
    for column in ("best_iteration", "tree_count", "iterations_executed"):
        if column in result:
            result[column] = result[column].astype("Int64")
    return result


def markdown_table(frame: pd.DataFrame) -> str:
    formatted = frame.copy()
    for name in ("MAE", "RMSE", "R2"):
        if name in formatted:
            formatted[name] = formatted[name].map(lambda value: f"{value:.6f}")
    for name in ("training_seconds", "prediction_seconds"):
        if name in formatted:
            formatted[name] = formatted[name].map(lambda value: f"{value:.3f}" if pd.notna(value) else "—")
    return "\n".join([
        "| " + " | ".join(formatted.columns) + " |",
        "| " + " | ".join(["---"] * len(formatted.columns)) + " |",
        *("| " + " | ".join("—" if pd.isna(value) else str(value) for value in row) + " |" for row in formatted.itertuples(index=False, name=None)),
    ])


def save_report(results: pd.DataFrame, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    results.to_csv(directory / "model_comparison.csv", index=False)
    means = results.groupby("model", sort=False)[["MAE", "RMSE", "R2"]].mean().reset_index()
    required = ["model", "fold", "train_start", "train_end", "validation_start", "validation_end", "MAE", "RMSE", "R2", "feature_set"]
    counts = results.drop_duplicates("fold")[["fold", "train_rows", "validation_rows"]]
    recent = results.loc[results.fold.eq("fold_3"), ["model", "MAE", "RMSE", "R2"]]
    nonpositive = results.groupby("model", sort=False)["nonpositive_predictions"].sum().reset_index()
    best_mean = means.sort_values("MAE").iloc[0]
    best_recent = recent.sort_values("MAE").iloc[0]
    feature_sets = dict(FEATURE_SETS)
    if "feature_columns" in results:
        for row in results[["feature_set", "feature_columns"]].drop_duplicates().itertuples(index=False):
            if isinstance(row.feature_columns, str):
                feature_sets[row.feature_set] = json.loads(row.feature_columns)
    feature_text = "\n".join(f"- **{name}:** {', '.join(columns) if columns else 'no predictive inputs; median of training targets only'}." for name, columns in feature_sets.items())
    parameter_text = "\n".join(f"- **{name}:** `{parameters}`." for name, parameters in results[["model", "parameters"]].drop_duplicates().itertuples(index=False, name=None))
    catboost_rows = results.loc[results.model.str.startswith("catboost_")]
    catboost_text = ""
    if not catboost_rows.empty:
        iteration_columns = ["model", "fold", "best_iteration", "tree_count", "iterations_executed", "training_seconds", "prediction_seconds"]
        catboost_text = f"""## CatBoost iterations and training time

{markdown_table(catboost_rows[iteration_columns])}

best_iteration is zero-based; tree_count is the number of retained trees. iterations_executed includes rounds run before stopping, including rounds later discarded by use_best_model=True. Fixed-budget confirmation rows have no best_iteration because they do not receive an eval_set. Training time excludes prediction time. Baseline records from the earlier run contain combined fitting/prediction time only; missing CatBoost-specific fields are not applicable to them.

The initial hyperparameter comparison used three joint configurations: depth 7 / learning_rate 0.03 / l2_leaf_reg 3; depth 6 / learning_rate 0.05 / l2_leaf_reg 5; depth 8 / learning_rate 0.04 / l2_leaf_reg 7, each across three folds. Later feature/weight/target comparisons hold the strongest configuration (depth 6 / learning_rate 0.05 / l2_leaf_reg 5) fixed. Early-stopped comparisons use a 1,500-iteration cap and 100-round patience; fixed-budget confirmations use the frozen iteration counts without an eval_set. Full parameters are logged in every record.

**Early-stopping methodology:** for adaptive-budget comparisons, each chronological outer validation window is also CatBoost's eval_set. Gradient/tree fitting and native categorical statistics use training-fold observations; eval_set labels choose the best iteration. Fixed-budget confirmation fits do not use evaluation labels while fitting, but their features/configuration were chosen using the same earlier validation windows. Metrics remain model-selection estimates, not an untouched independent test. Direct-target stopping uses MAE; log1p-target stopping uses the custom DollarMAE metric, converting both prediction and truth back with expm1. All reported MAE/RMSE/R2 use original dollar-scale outcomes. September–October retains its role as the primary local selection holdout. No November–December final-validation observations are loaded or used.

CatBoost uses explicit string categories and native numeric NaN handling, with no sklearn one-hot encoding or full-data imputation. No external target encoding is computed. Selected feature subsets are listed below, and raw weight is distinguished from abs(weight) plus original missing/negative indicators. Coordinates are retained in eligible final models. Model serialization and production training are separate later steps.
"""
    text = f"""# Chronological model comparison

Reproduce the initial baselines/hyperparameters: `.venv/bin/python -m src.evaluate --models all`. Reproduce the controlled feature/weight/target experiments and frozen-budget confirmation: `.venv/bin/python scripts/run_model_experiments.py --phase all`. Regenerate analysis/selection documents: `.venv/bin/python scripts/report_model_selection.py`. Incremental initial comparison options are `--models catboost` and `--models baselines`.

All {len(results)} recorded model/fold results use only `data/train_test.csv` (48,000 labeled rows, January 1–October 31, 2025). The F-clean/direct ablation reuses the identical previously executed depth-6 full-feature result, rather than refitting it. No final-validation features, targets, or inference distributions are used for these experiments. Dates are never shuffled, and entire dates stay together. Fold 3 is the primary September–October local holdout; earlier folds provide additional evidence of temporal stability. See `docs/validation_plan.md`.

## Fold sizes

{markdown_table(counts)}

## Per-fold results

{markdown_table(results[required])}

MAE and RMSE are in posted-rate units (USD); R2 is dimensionless. CSV metrics retain normal floating-point precision. No row or high-rate outlier is removed. Predictions are evaluated directly without clipping; the CSV records any nonpositive local predictions. Production outputs will need their separate positivity checks.

## Unweighted mean across the three folds

{markdown_table(means)}

These are arithmetic means of per-fold metrics, not metrics recomputed on pooled predictions. Expanding training windows share historical observations; the folds are not independent samples.

## Primary September–October holdout

{markdown_table(recent)}

## Observations from these fixed experiments

Among these experiments, **{best_mean['model']}** has the lowest mean chronological MAE ({best_mean['MAE']:.2f}). **{best_recent['model']}** has the lowest primary September–October MAE ({best_recent['MAE']:.2f}). The frozen production decision, inference constraints, and fixed-budget confirmation are documented in `docs/model_selection.md`; the feature/weight/target evidence is in `reports/feature_ablation.md`.

Nonpositive local predictions across all three validation windows:

{markdown_table(nonpositive)}

These local predictions are retained in the metrics without clipping. They are not final submission outputs. A configuration that produces nonpositive predictions needs an explicitly validated treatment before production use.

## Feature sets and fixed configurations

{feature_text}

{parameter_text}

{catboost_text}

The distance baseline predicts an intercept plus a distance coefficient. Its distance median imputer is fitted only on training rows. Ridge uses training-fold numeric medians, scaling, and one-hot categoricals with `handle_unknown="ignore"`, retaining coordinates for unfamiliar city labels. The global median is calculated independently from each fold's training targets. Every fold receives fresh estimators and fresh learned preprocessing. Feature engineering is deterministic, excludes load_id/posted_rate, preserves missing/negative-weight flags, and uses a fixed reference date.

Market/quote provenance, availability, missingness, and temporal behavior are documented in `reports/feature_ablation.md`. These local results do not establish accuracy for future unseen cities or the fixed December scenario. No production model, submission predictions, or Spotter hidden metric is created by these experiment commands.

The CSV also records row counts, fixed estimator parameters, elapsed fitting/prediction time, and nonpositive prediction counts. Timing varies by machine and run.
"""
    (directory / "model_comparison.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", choices=["baselines", "catboost", "all"], default="all")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "docs/supplied_file_checksums.json").read_text())
    protected = ["data/train_test.csv", "score.py"]
    for name in protected:
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != manifest["files"][name]["sha256"]:
            raise AssertionError(f"Original supplied file changed: {name}")
    frame = pd.read_csv(ROOT / "data/train_test.csv")
    if len(frame) != 48000 or frame.date.min() != "2025-01-01" or frame.date.max() != "2025-10-31":
        raise ValueError("Unexpected labeled development rows or dates")
    comparison_path = ROOT / "reports/model_comparison.csv"
    previous = pd.read_csv(comparison_path, float_precision="round_trip") if comparison_path.exists() else pd.DataFrame(columns=["model", "fold"])
    # Ensure retained experiments use precisely the same fold boundaries/counts.
    splits = {fold.name: (fold, len(train_idx), len(val_idx))
              for fold, train_idx, val_idx in chronological_splits(frame)}
    for row in previous.itertuples(index=False):
        fold, train_rows, validation_rows = splits[row.fold]
        if (row.train_start, row.train_end, row.validation_start, row.validation_end,
            row.train_rows, row.validation_rows) != (
            fold.train_start, fold.train_end, fold.validation_start, fold.validation_end,
            train_rows, validation_rows):
            raise ValueError("Prior comparison contains different fold definitions")
    executed = []
    if args.models in ("baselines", "all"):
        executed.append(run_baselines(frame))
    if args.models in ("catboost", "all"):
        executed.append(run_catboost(frame))
    results = merge_results(previous, pd.concat(executed, ignore_index=True))
    save_report(results, ROOT / "reports")
    for name in protected:
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != manifest["files"][name]["sha256"]:
            raise AssertionError(f"Supplied file changed during experiments: {name}")
    print("Saved reports/model_comparison.csv and reports/model_comparison.md")


if __name__ == "__main__":
    main()
