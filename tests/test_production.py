"""Production contracts, ID joins, serialization, and actual submission outputs."""

import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.model_config import predict_with_spec
from src.predict import load_artifacts
from src.production import (
    ROOT, DECEMBER_COLUMNS, add_december_coordinates, align_predictions,
    check_immutable_inputs, city_coordinate_lookup, rate_summary,
    validate_december, validate_submission,
)


@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts()


def test_id_join_handles_shuffled_validation():
    template = pd.DataFrame({"load_id": ["A", "B", "C"], "predicted_rate": np.nan})
    validation = pd.DataFrame({"load_id": ["C", "A", "B"]})
    actual = align_predictions(validation, template, [300, 100, 200], expected_rows=3)
    assert actual.load_id.tolist() == ["A", "B", "C"]
    assert actual.predicted_rate.tolist() == [100, 200, 300]


@pytest.mark.parametrize("ids", [["A", "A", "C"], ["A", "B", None], ["A", "B", "extra"], ["A", "B", ""]])
def test_id_join_rejects_invalid_or_different_ids(ids):
    template = pd.DataFrame({"load_id": ["A", "B", "C"], "predicted_rate": np.nan})
    with pytest.raises(ValueError):
        align_predictions(pd.DataFrame({"load_id": ids}), template, [100, 200, 300], expected_rows=3)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 0, -1])
def test_submission_rejects_bad_rates(bad):
    template = pd.DataFrame({"load_id": ["A"], "predicted_rate": np.nan})
    with pytest.raises(ValueError):
        validate_submission(template.assign(predicted_rate=bad), template, expected_rows=1)


def test_submission_rejects_extra_index_column():
    template = pd.DataFrame({"load_id": ["A"], "predicted_rate": [1]})
    with pytest.raises(ValueError, match="exactly"):
        validate_submission(template.assign(index=0), template, expected_rows=1)


def test_real_submission_and_summary():
    predictions = pd.read_csv(ROOT / "validation_predictions.csv")
    template = pd.read_csv(ROOT / "data/validation_predictions_template.csv")
    validation = pd.read_csv(ROOT / "data/validation.csv")
    validate_submission(predictions, template)
    assert predictions.shape == (12000, 2)
    assert set(predictions.load_id) == set(validation.load_id)
    summary = json.loads((ROOT / "artifacts/prediction_summary.json").read_text())["validation"]
    for key, actual in rate_summary(predictions.predicted_rate).items():
        assert summary[key] == pytest.approx(actual, rel=1e-12)


def test_trained_model_metadata_and_feature_contract(artifacts):
    models, config, preprocessing = artifacts
    metadata = json.loads((ROOT / "artifacts/training_metadata.json").read_text())
    for role in ("primary", "december"):
        saved = metadata if role == "primary" else metadata[role]
        assert saved["target"] == "posted_rate"
        assert saved["training_rows"] == 48000
        assert saved["training_date_range"] == {"start": "2025-01-01", "end": "2025-10-31"}
        assert saved["random_seed"] == 42
        assert saved["parameters"] == config[role]["parameters"]
        assert saved["selected_features"] == models[role].feature_names_ == config[role]["feature_columns"]
        assert not {"load_id", "posted_rate"} & set(saved["selected_features"])
        assert saved["tree_count"] == config[role]["parameters"]["iterations"]
        assert saved["package_versions"]["catboost"]
        assert saved["chronological_validation_metrics"]["mean_across_folds"] == config[role]["local_metrics_mean"]
    assert not {"quote_signal", "market_index", "market_index_missing"} & set(config["december"]["feature_columns"])
    assert preprocessing["city_coordinates"]["Lexington"] == {"lat": 36.99152, "lon": -84.99876}
    assert preprocessing["city_coordinates"]["Fort Wayne"] == {"lat": 41.31561, "lon": -85.36206}


def test_loaded_model_reproduces_submission_by_id(artifacts):
    models, config, _ = artifacts
    frame = pd.read_csv(ROOT / "data/validation.csv").iloc[:128].sample(frac=1, random_state=42)
    predictions = pd.read_csv(ROOT / "validation_predictions.csv").set_index("load_id")
    actual = predict_with_spec(models["primary"], frame, config["primary"])
    np.testing.assert_allclose(actual, predictions.loc[frame.load_id, "predicted_rate"], rtol=1e-12, atol=1e-10)
    # The same prediction function ignores identifier/target columns if accidentally supplied.
    changed = frame.assign(load_id="ignored", posted_rate=-999999)
    np.testing.assert_array_equal(actual, predict_with_spec(models["primary"], changed, config["primary"]))


def test_real_unseen_cities_and_problem_weights_inference(artifacts):
    models, config, _ = artifacts
    train = pd.read_csv(ROOT / "data/train_test.csv")
    validation = pd.read_csv(ROOT / "data/validation.csv")
    cities = set(train.pickup) | set(train.delivery)
    cases = {
        "new_city": ~validation.pickup.isin(cities) | ~validation.delivery.isin(cities),
        "missing_weight": validation.weight.isna(),
        "negative_weight": validation.weight.lt(0),
        "missing_market": validation.market_index.isna(),
    }
    for mask in cases.values():
        assert mask.any()
        rates = predict_with_spec(models["primary"], validation.loc[mask], config["primary"])
        assert len(rates) == mask.sum()
        assert np.isfinite(rates).all() and (rates > 0).all()


def test_real_december_preserves_inputs_and_reproduces_rates(artifacts):
    models, config, preprocessing = artifacts
    december = pd.read_csv(ROOT / "data/december_chart_inputs.csv")
    validate_december(december, original_inputs=preprocessing["december_original_inputs"])
    assert december.shape == (31, 7)
    before = december.copy(deep=True)
    enriched = add_december_coordinates(december.drop(columns="predicted_rate"), preprocessing["city_coordinates"])
    assert "market_index" not in enriched and "quote_signal" not in enriched
    assert_frame_equal(before, december)
    actual = predict_with_spec(models["december"], enriched, config["december"])
    np.testing.assert_allclose(actual, december.predicted_rate, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("change", ["date", "pickup", "delivery", "distance", "equipment", "weight", "column_order", "rate"])
def test_december_rejects_tampered_output(change):
    december = pd.read_csv(ROOT / "data/december_chart_inputs.csv")
    if change == "column_order":
        december = december[DECEMBER_COLUMNS[::-1]]
    else:
        name, value = ("predicted_rate", np.nan) if change == "rate" else (change, {
            "date": "2025-11-30", "pickup": "Chicago", "delivery": "Chicago", "distance": 361,
            "equipment": "Reefer", "weight": 32001,
        }[change])
        december.loc[0, name] = value
    with pytest.raises(ValueError):
        validate_december(december)


def test_coordinate_lookup_rejects_conflicting_locations():
    frame = pd.DataFrame({"pickup": ["A", "A"], "pickup_lat": [1, 2], "pickup_lon": [3, 3],
                          "delivery": ["B", "B"], "delivery_lat": [4, 4], "delivery_lon": [5, 5]})
    with pytest.raises(ValueError, match="conflict"):
        city_coordinate_lookup(frame)


def test_immutable_supplied_files_unchanged():
    check_immutable_inputs()
