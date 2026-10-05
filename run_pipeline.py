"""Reproduce production artifacts and predictions: python run_pipeline.py.

Research/model selection and the original scorer are separate commands.
"""

from time import perf_counter

import pandas as pd

from scripts.check_inputs import INPUTS, validate_contracts
from src.features import build_model_features
from src.model_config import load_frozen_config
from src.predict import main as generate_predictions
from src.production import (
    ROOT, DECEMBER_COLUMNS, add_december_coordinates, check_immutable_inputs,
    city_coordinate_lookup, sha256, validate_december, validate_submission, write_json,
)
from src.train import chronological_metrics, main as train_final_models


def require_input_contracts(frames):
    """Fail explicitly on invalid inputs; retain missing/negative predictor rows."""
    checks = validate_contracts(frames)
    failures = [f"{name}: {detail}" for name, passed, detail in checks if not passed]
    if failures:
        raise ValueError("Input validation failed:\n" + "\n".join(failures))
    validate_december(frames["december_chart_inputs"], require_predictions=False)
    return len(checks)


def validate_inputs():
    check_immutable_inputs()
    frames = {name: pd.read_csv(ROOT / "data" / f"{name}.csv") for name in INPUTS}
    count = require_input_contracts(frames)
    config = load_frozen_config()
    # Confirm recorded metrics before fitting either model; these are not new experiments.
    for role in ("primary", "december"):
        chronological_metrics(config[role])
    train = frames["train_test"].drop(columns=["load_id", "posted_rate"])
    validation = frames["validation"].drop(columns="load_id")
    december = frames["december_chart_inputs"]
    enriched = add_december_coordinates(
        december.drop(columns="predicted_rate"), city_coordinate_lookup(train),
    )
    feature_shapes = {}
    for name, frame, role in (
        ("development_primary", train, "primary"),
        ("development_december", train, "december"),
        ("final_inference", validation, "primary"),
        ("fixed_december", enriched, "december"),
    ):
        spec = config[role]
        features = build_model_features(frame, spec["feature_columns"], spec["reference_date"])
        if len(features) != len(frame) or {"load_id", "posted_rate"} & set(features):
            raise AssertionError("Feature row-preservation or leakage contract failed")
        feature_shapes[name] = list(features.shape)
    print(f"Inputs PASS: {count} contract checks; immutable hashes and frozen features verified.", flush=True)
    return count, feature_shapes, december[DECEMBER_COLUMNS[:-1]].to_dict("records")


def verify_outputs(original_december_inputs):
    template = pd.read_csv(ROOT / "data/validation_predictions_template.csv")
    predictions = pd.read_csv(ROOT / "validation_predictions.csv")
    december = pd.read_csv(ROOT / "data/december_chart_inputs.csv")
    validate_submission(predictions, template)
    validate_december(december, original_inputs=original_december_inputs)
    check_immutable_inputs()
    return {"validation_rows": len(predictions), "december_rows": len(december),
            "validation_sha256": sha256(ROOT / "validation_predictions.csv"),
            "december_sha256": sha256(ROOT / "data/december_chart_inputs.csv")}


def main():
    started = perf_counter()
    print("1/4 Validate inputs and build the frozen features", flush=True)
    checks, feature_shapes, original_december_inputs = validate_inputs()
    print("2/4 Train both frozen models on all 48,000 labeled rows and save artifacts", flush=True)
    train_final_models()
    print("3/4 Generate validation and fixed-December predictions", flush=True)
    generate_predictions()
    print("4/4 Verify written outputs and immutable inputs", flush=True)
    output_checks = verify_outputs(original_december_inputs)
    write_json(ROOT / "artifacts/pipeline_checks.json", {
        "status": "PASS", "input_contract_checks": checks, "feature_shapes": feature_shapes,
        **output_checks, "elapsed_seconds": perf_counter() - started,
        "frozen_configuration_sha256": sha256(ROOT / "docs/final_model_config.json"),
        "research_experiments_rerun": False,
    })
    print("Production pipeline PASS: 12,000 validation predictions and 31 December predictions.", flush=True)


if __name__ == "__main__":
    main()
