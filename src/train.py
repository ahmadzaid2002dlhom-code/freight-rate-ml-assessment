"""Fit the already frozen primary and December models on all development rows.

Run from the repository root: python -m src.train
No final-validation observations, eval_set, early stopping, or tuning are used.
"""

from datetime import datetime, timezone
import importlib.metadata
import platform
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .features import build_model_features
from .model_config import load_frozen_config
from .production import ROOT, DECEMBER_COLUMNS, check_ids, check_immutable_inputs, city_coordinate_lookup, sha256, validate_december, write_json


def chronological_metrics(spec):
    """Verify saved metrics against the executed fixed-budget fold records."""
    results = pd.read_csv(ROOT / "reports/model_comparison.csv")
    selected = results[results.model.eq(spec["confirmation_model"])].sort_values("fold")
    if selected.fold.tolist() != ["fold_1", "fold_2", "fold_3"]:
        raise ValueError("Missing or duplicate frozen-model fold results")
    for metric in ("MAE", "RMSE", "R2"):
        if not np.isclose(selected[metric].mean(), spec["local_metrics_mean"][metric], rtol=1e-10):
            raise ValueError(f"Frozen mean {metric} differs from executed experiments")
        if not np.isclose(selected.iloc[-1][metric], spec["local_metrics_primary_holdout"][metric], rtol=1e-10):
            raise ValueError(f"Frozen holdout {metric} differs from executed experiments")
    columns = ["fold", "train_start", "train_end", "validation_start", "validation_end", "train_rows", "validation_rows", "MAE", "RMSE", "R2"]
    return {"mean_across_folds": spec["local_metrics_mean"],
            "sep_oct_holdout": spec["local_metrics_primary_holdout"],
            "folds": selected[columns].to_dict("records"),
            "source": "reports/model_comparison.csv", "model": spec["confirmation_model"],
            "methodology": "Three expanding chronological folds; historical folds reused for model selection; no independent hidden-test metric."}


def main():
    check_immutable_inputs()
    config = load_frozen_config()
    training = pd.read_csv(ROOT / "data/train_test.csv")
    check_ids(training, 48000, "Development")
    dates = pd.to_datetime(training.date, format="%Y-%m-%d", errors="raise")
    if dates.isna().any() or str(dates.min().date()) != "2025-01-01" or str(dates.max().date()) != "2025-10-31":
        raise ValueError("Unexpected development date range")
    target = pd.to_numeric(training.posted_rate, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(target).all() or not (target > 0).all():
        raise ValueError("Training targets must be finite and positive; no rows may be dropped")
    predictors = training.drop(columns=["load_id", "posted_rate"])
    december = pd.read_csv(ROOT / "data/december_chart_inputs.csv")
    validate_december(december, require_predictions=False)
    artifact_dir = ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    preprocessing = {
        "frozen_configuration": config,
        "numeric_missing": "CatBoost native NaN handling (nan_mode=Min); no learned imputation",
        "categorical_missing": "Explicit __MISSING__ strings in src/features.py",
        "weight": "abs(original weight), original missing and negative indicators; retain every row",
        "coordinate_lookup_source": "data/train_test.csv, consistent across pickup and delivery roles",
        "city_coordinates": city_coordinate_lookup(training),
        "december_original_inputs": december[DECEMBER_COLUMNS[:-1]].to_dict("records"),
        "source_hashes": {path: sha256(ROOT / path) for path in [
            "data/train_test.csv", "docs/final_model_config.json", "src/features.py", "src/model_config.py", "src/evaluate.py",
        ]},
    }
    versions = {name: importlib.metadata.version(name) for name in (
        "numpy", "pandas", "matplotlib", "scikit-learn", "catboost", "joblib", "pytest", "python-docx",
    )}
    versions["python"] = platform.python_version()
    models = {}
    for role in ("primary", "december"):
        spec = config[role]
        local_metrics = chronological_metrics(spec)
        features = build_model_features(predictors, spec["feature_columns"], spec["reference_date"])
        if len(features) != 48000 or {"load_id", "posted_rate"} & set(features):
            raise AssertionError("Training row or leakage contract violated")
        fit_target = np.log1p(target) if spec["target_strategy"] == "log1p" else target
        model = CatBoostRegressor(**spec["parameters"])
        started = perf_counter()
        model.fit(features, fit_target, cat_features=spec["categorical_features"], use_best_model=False)
        elapsed = perf_counter() - started
        if model.tree_count_ != spec["parameters"]["iterations"]:
            raise AssertionError("Fitted model changed the frozen tree budget")
        filename = f"{role}_model.cbm"
        temporary = artifact_dir / (filename + ".tmp")
        model.save_model(str(temporary), format="cbm")
        restored = CatBoostRegressor()
        restored.load_model(str(temporary), format="cbm")
        np.testing.assert_allclose(restored.predict(features.iloc[:128]), model.predict(features.iloc[:128]), rtol=0, atol=0)
        temporary.replace(artifact_dir / filename)
        models[role] = {
            "target": "posted_rate", "model_type": spec["model_type"], "parameters": spec["parameters"],
            "selected_features": spec["feature_columns"], "categorical_features": spec["categorical_features"],
            "target_strategy": spec["target_strategy"], "prediction_floor": spec["prediction_floor"],
            "training_rows": len(features), "training_date_range": {"start": str(dates.min().date()), "end": str(dates.max().date())},
            "random_seed": spec["parameters"]["random_seed"], "chronological_validation_metrics": local_metrics,
            "training_seconds": elapsed, "tree_count": model.tree_count_, "model_file": filename,
            "model_sha256": sha256(artifact_dir / filename), "package_versions": versions,
            "training_policy": "All 48,000 labeled rows; fixed parameters; no eval_set or early stopping",
        }
        print(f"Trained {role}: {len(features):,} rows, {model.tree_count_} trees, {elapsed:.2f} seconds; serialization verified.", flush=True)
    write_json(artifact_dir / "preprocessing.json", preprocessing)
    metadata = {**models["primary"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "production_training_complete", "preprocessing_file": "preprocessing.json",
                "preprocessing_sha256": sha256(artifact_dir / "preprocessing.json"),
                "source_hashes": preprocessing["source_hashes"], "december": models["december"],
                "december_strategy": "Separate frozen group-C direct-target model; supplied city coordinates only; no market_index or quote_signal"}
    write_json(artifact_dir / "training_metadata.json", metadata)
    check_immutable_inputs()
    print("Saved native CatBoost models, preprocessing.json, and training_metadata.json.")


if __name__ == "__main__":
    main()
