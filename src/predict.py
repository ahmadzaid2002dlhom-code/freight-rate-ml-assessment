"""Generate both submission files using the saved, frozen production artifacts.

Run from the repository root: python -m src.predict
"""

import json

import pandas as pd
from catboost import CatBoostRegressor
from pandas.testing import assert_frame_equal

from .model_config import load_frozen_config, predict_with_spec
from .production import (
    ROOT, DECEMBER_COLUMNS, add_december_coordinates, align_predictions,
    check_immutable_inputs, rate_summary, sha256, validate_december,
    validate_submission, write_csv, write_json,
)


def load_artifacts():
    directory = ROOT / "artifacts"
    metadata = json.loads((directory / "training_metadata.json").read_text())
    if sha256(directory / metadata["preprocessing_file"]) != metadata["preprocessing_sha256"]:
        raise ValueError("Saved preprocessing changed after training")
    preprocessing = json.loads((directory / metadata["preprocessing_file"]).read_text())
    config = load_frozen_config()
    if preprocessing["frozen_configuration"] != config:
        raise ValueError("Frozen configuration differs from trained artifacts")
    for relative, digest in preprocessing["source_hashes"].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"Training source or feature implementation changed: {relative}")
    models = {}
    for role in ("primary", "december"):
        saved = metadata if role == "primary" else metadata["december"]
        path = directory / saved["model_file"]
        if sha256(path) != saved["model_sha256"]:
            raise ValueError(f"Saved {role} model changed")
        model = CatBoostRegressor()
        model.load_model(str(path), format="cbm")
        spec = config[role]
        if model.feature_names_ != spec["feature_columns"] or model.tree_count_ != spec["parameters"]["iterations"]:
            raise ValueError(f"Loaded {role} model violates frozen features or tree budget")
        if model.get_cat_feature_indices() != [spec["feature_columns"].index(name) for name in spec["categorical_features"]]:
            raise ValueError(f"Loaded {role} categorical preprocessing differs")
        models[role] = model
    return models, config, preprocessing


def main():
    check_immutable_inputs()
    models, config, preprocessing = load_artifacts()
    validation = pd.read_csv(ROOT / "data/validation.csv")
    template = pd.read_csv(ROOT / "data/validation_predictions_template.csv")
    rates = predict_with_spec(models["primary"], validation.drop(columns="load_id"), config["primary"])
    output = align_predictions(validation, template, rates)
    december_path = ROOT / "data/december_chart_inputs.csv"
    december = pd.read_csv(december_path)
    validate_december(december, require_predictions=False, original_inputs=preprocessing["december_original_inputs"])
    enriched = add_december_coordinates(december.drop(columns="predicted_rate"), preprocessing["city_coordinates"])
    december_output = december.copy(deep=True)
    december_output["predicted_rate"] = predict_with_spec(models["december"], enriched, config["december"])
    assert_frame_equal(december_output[DECEMBER_COLUMNS[:-1]], december[DECEMBER_COLUMNS[:-1]])
    validate_december(december_output, original_inputs=preprocessing["december_original_inputs"])
    # Both outputs are checked before either file is written; verify CSV round trips too.
    write_csv(ROOT / "validation_predictions.csv", output)
    write_csv(december_path, december_output)
    validate_submission(pd.read_csv(ROOT / "validation_predictions.csv"), template)
    validate_december(pd.read_csv(december_path), original_inputs=preprocessing["december_original_inputs"])
    check_immutable_inputs()
    summaries = {"validation": rate_summary(output.predicted_rate), "december": rate_summary(december_output.predicted_rate),
                 "validation_sha256": sha256(ROOT / "validation_predictions.csv"), "december_sha256": sha256(december_path),
                 "primary_floor_count": int(output.predicted_rate.eq(config["primary"]["prediction_floor"]).sum()),
                 "december_floor_count": int(december_output.predicted_rate.eq(config["december"]["prediction_floor"]).sum())}
    write_json(ROOT / "artifacts/prediction_summary.json", summaries)
    print("First 10 predictions (joined by load_id):")
    print(output.head(10).to_string(index=False))
    print(json.dumps(summaries, indent=2))
    print("Verified exact template IDs, output schemas, all rows, positive finite rates, and unchanged December fixed inputs.")


if __name__ == "__main__":
    main()
