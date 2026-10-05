"""Prediction compatibility tests; diagnostic fits do not select final models."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor
from pandas.testing import assert_frame_equal
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from scripts.check_unseen_categories import coverage, inference_checks
from src.features import CATEGORICAL_FEATURE_COLUMNS, FreightFeatureTransformer, build_features
from src.preprocessing import build_sklearn_preprocessor

ROOT = Path(__file__).resolve().parents[1]
COORDINATES = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]


@pytest.fixture
def synthetic_data():
    train = pd.DataFrame({
        "pickup": ["Known A", "Known B"] * 4,
        "delivery": ["Known B", "Known A"] * 4,
        "equipment": ["Dry Van", "Reefer"] * 4,
        "distance": [100, 200, 300, 400, 500, 600, 700, 800],
        "weight": [10000, 20000, 15000, 22000, 31000, 30000, 18000, 40000],
        "date": pd.date_range("2025-01-01", periods=8).strftime("%Y-%m-%d"),
        "pickup_lat": [30.0, 40.0] * 4, "pickup_lon": [-85.0, -95.0] * 4,
        "delivery_lat": [40.0, 30.0] * 4, "delivery_lon": [-95.0, -85.0] * 4,
        "market_index": [1.0] * 8, "quote_signal": [2.0] * 8,
    })
    train["posted_rate"] = 500 + 2 * train.distance
    final = train.iloc[:4].copy().drop(columns="posted_rate")
    final["date"] = "2025-11-01"
    final["pickup"] = ["Never-seen origin", "Known A", "New origin", None]
    final["delivery"] = ["Known B", "Never-seen destination", "New destination", None]
    final["equipment"] = ["New equipment", "Dry Van", "Reefer", None]
    final["weight"] = [np.nan, -32000, 10000, np.nan]
    final["market_index"] = np.nan
    final["pickup_lat"] = [32.0, 30.0, 33.0, np.nan]
    final["delivery_lon"] = [-95.0, -96.0, -97.0, np.nan]
    return train, final


def test_sklearn_predicts_unknown_and_missing_labels(synthetic_data):
    train, final = synthetic_data
    model = Pipeline([
        ("features", FreightFeatureTransformer()),
        ("preprocessing", build_sklearn_preprocessor()),
        ("model", Ridge(alpha=1.0, random_state=42)),
    ])
    train_before, final_before = train.copy(deep=True), final.copy(deep=True)
    model.fit(train, train.posted_rate)
    predictions = model.predict(final)
    assert predictions.shape == (len(final),)
    assert np.isfinite(predictions).all()
    assert_frame_equal(train, train_before)
    assert_frame_equal(final, final_before)
    processor = model.named_steps["preprocessing"]
    encoder = processor.named_transformers_["categorical"]
    assert encoder.handle_unknown == "ignore"
    for name, categories in zip(CATEGORICAL_FEATURE_COLUMNS, encoder.categories_):
        assert set(categories) == set(build_features(train)[name])
    assert all(f"numeric__{name}" in processor.get_feature_names_out() for name in COORDINATES)


def test_catboost_predicts_unknown_and_missing_labels(synthetic_data):
    train, final = synthetic_data
    training_features, inference_features = build_features(train), build_features(final)
    assert set(inference_features.route).isdisjoint(set(training_features.route))
    model = CatBoostRegressor(
        iterations=8, depth=2, random_seed=42, thread_count=2,
        verbose=False, allow_writing_files=False,
    )
    model.fit(training_features, train.posted_rate, cat_features=CATEGORICAL_FEATURE_COLUMNS)
    predictions = model.predict(inference_features)
    assert predictions.shape == (len(final),)
    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()


def test_unknown_encoding_keeps_numeric_geography(synthetic_data):
    train, final = synthetic_data
    processor = build_sklearn_preprocessor().fit(build_features(train))
    one = final.iloc[[0]].copy()
    other = one.assign(pickup_lat=one.pickup_lat + 1)
    first = processor.transform(build_features(one)).toarray()
    second = processor.transform(build_features(other)).toarray()
    names = processor.get_feature_names_out().tolist()
    coordinate_index = names.index("numeric__pickup_lat")
    assert first[0, coordinate_index] != second[0, coordinate_index]
    np.testing.assert_allclose(np.delete(first, coordinate_index, axis=1), np.delete(second, coordinate_index, axis=1))


def test_learned_imputation_and_vocabulary_use_training_only(synthetic_data):
    train, final = synthetic_data
    training_features = build_features(train)
    processor = build_sklearn_preprocessor().fit(training_features)
    imputer = processor.named_transformers_["numeric"].named_steps["imputer"]
    medians_before = imputer.statistics_.copy()
    # An extreme inference observation must not update any fitted statistic.
    processor.transform(build_features(final.assign(weight=999999999, market_index=999999)))
    np.testing.assert_array_equal(imputer.statistics_, medians_before)
    numeric_names = processor.transformers_[0][2]
    assert medians_before[numeric_names.index("weight_clean")] == train.weight.median()


@pytest.fixture(scope="module")
def supplied_data():
    return pd.read_csv(ROOT / "data/train_test.csv"), pd.read_csv(ROOT / "data/validation.csv")


def test_actual_unseen_city_and_route_counts(supplied_data):
    result = coverage(*supplied_data)
    assert len(result["train_cities"]) == 64
    assert len(result["final_cities"]) == 72
    assert result["new_cities"] == ["Allentown", "Charlotte", "Chicago", "Jackson", "Knoxville", "Laredo", "Norfolk", "San Diego"]
    masks = result["masks"]
    assert masks["Completely new city at either endpoint"].sum() == 1447
    assert masks["Unseen directed route"].sum() == 1461
    assert result["final"].loc[masks["Unseen directed route"], "route"].nunique() == 736
    assert result["final"].loc[masks["Completely new city at either endpoint"], COORDINATES].notna().all().all()


def test_actual_final_validation_prediction_compatibility(supplied_data):
    result = inference_checks(*supplied_data)
    assert result["training_rows"] == 1108
    assert result["training_date_max"] < "2025-11-01"
    assert result["inference_rows"] == 12000
    assert result["sklearn_finite_predictions"] == 12000
    assert result["catboost_finite_predictions"] == 12000
    assert result["coordinates_retained"]
