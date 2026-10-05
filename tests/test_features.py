"""Shared feature contract checks, without fitting a predictive model."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder
from catboost import Pool

from src.features import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    MISSING_CATEGORY,
    NUMERIC_FEATURE_COLUMNS,
    FreightFeatureTransformer,
    build_features,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def loads():
    return pd.DataFrame({
        "load_id": ["A", "B", "C"], "posted_rate": [1000, 2000, 3000],
        "pickup": ["Lexington", None, "New city"],
        "delivery": ["Fort Wayne", " ", "Unknown destination"],
        "equipment": ["Dry Van", pd.NA, "New equipment"],
        "distance": [360, 400, 500], "weight": [-32000, np.nan, 15000],
        "date": ["2025-01-01", "2025-01-04", "2025-12-31"],
        "market_index": [1.2, pd.NA, 0.9], "quote_signal": [2.1, np.nan, 2.4],
        "pickup_lat": [37.0, np.nan, 30.0], "pickup_lon": [-85.0, np.nan, -90.0],
        "delivery_lat": [41.0, np.nan, 40.0], "delivery_lon": [-85.0, np.nan, -95.0],
    }, index=[10, 20, 30])


def test_whitelist_and_input_immutability(loads):
    before = loads.copy(deep=True)
    result = build_features(loads.assign(predicted_rate=9999, arbitrary_debug_column="unused"))
    assert_frame_equal(loads, before)
    assert result.columns.tolist() == FEATURE_COLUMNS
    assert result.index.equals(loads.index)
    assert set(CATEGORICAL_FEATURE_COLUMNS).isdisjoint(NUMERIC_FEATURE_COLUMNS)
    assert set(FEATURE_COLUMNS) == set(CATEGORICAL_FEATURE_COLUMNS + NUMERIC_FEATURE_COLUMNS)
    assert not {"load_id", "posted_rate", "predicted_rate", "date", "weight"} & set(result)
    assert_frame_equal(result, build_features(loads.drop(columns=["load_id", "posted_rate"])))
    assert_frame_equal(result, build_features(loads.assign(load_id="changed", posted_rate=-999)))


def test_weight_and_market_missing_flags(loads):
    result = build_features(loads)
    np.testing.assert_allclose(result.weight_clean, [32000, np.nan, 15000], equal_nan=True)
    assert result.weight_missing.tolist() == [0, 1, 0]
    assert result.weight_negative.tolist() == [1, 0, 0]
    assert result.market_index_missing.tolist() == [0, 1, 0]
    assert pd.isna(result.loc[20, "market_index"])
    assert pd.isna(result.loc[20, "quote_signal"])


def test_explicit_categorical_missing_and_unknown_values(loads):
    result = build_features(loads)
    for name in ("equipment", "pickup", "delivery"):
        assert result.loc[20, name] == MISSING_CATEGORY
    assert result.loc[20, "route"] == f"{MISSING_CATEGORY} -> {MISSING_CATEGORY}"
    assert result.loc[30, "route"] == "New city -> Unknown destination"
    assert all(isinstance(value, str) for value in result[CATEGORICAL_FEATURE_COLUMNS].to_numpy().ravel())
    # Fit vocabulary on one historical row; unseen labels transform safely.
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(result.loc[[10], CATEGORICAL_FEATURE_COLUMNS])
    transformed = encoder.transform(result.loc[[30], CATEGORICAL_FEATURE_COLUMNS])
    assert np.isfinite(transformed).all()
    assert (transformed == 0).all()


def test_catboost_accepts_missing_and_unseen_feature_values(loads):
    result = build_features(loads)
    pool = Pool(result, cat_features=CATEGORICAL_FEATURE_COLUMNS)
    assert pool.num_row() == len(loads)
    assert pool.num_col() == len(FEATURE_COLUMNS)


def test_calendar_definitions_and_cycles(loads):
    result = build_features(loads)
    assert result.month.tolist() == [1, 1, 12]
    assert result.day.tolist() == [1, 4, 31]
    assert result.day_of_week.tolist() == [2, 5, 2]  # Monday is zero.
    assert result.day_of_year.tolist() == [1, 4, 365]
    assert result.iso_week.tolist() == [1, 1, 1]  # Dec 31 belongs to ISO week 1 of 2026.
    assert result.weekend.tolist() == [0, 1, 0]
    assert result.days_since_reference_date.tolist() == [0, 3, 364]
    np.testing.assert_allclose(result.sin_day_of_week, np.sin(2 * np.pi * np.array([2, 5, 2]) / 7))
    np.testing.assert_allclose(result.cos_day_of_week, np.cos(2 * np.pi * np.array([2, 5, 2]) / 7))
    np.testing.assert_allclose(result.sin_day_of_year, np.sin(2 * np.pi * np.array([0, 3, 364]) / 365))
    np.testing.assert_allclose(result.cos_day_of_year, np.cos(2 * np.pi * np.array([0, 3, 364]) / 365))


def test_leap_year_and_reference_configuration(loads):
    frame = loads.iloc[[0]].assign(date="2024-02-29")
    result = build_features(frame, reference_date="2024-01-01")
    assert result.iloc[0].days_since_reference_date == 59
    assert result.iloc[0].day_of_year == 60
    assert result.iloc[0].sin_day_of_year == pytest.approx(np.sin(2 * np.pi * 59 / 366))


def test_batch_composition_and_datetime_inputs(loads):
    together = build_features(loads)
    separate = pd.concat([build_features(loads.iloc[[i]]) for i in range(len(loads))])
    assert_frame_equal(together, separate)
    assert_frame_equal(together, build_features(loads.assign(date=pd.to_datetime(loads.date))))


def test_december_available_fields_have_identical_features(loads):
    december_columns = ["pickup", "delivery", "distance", "equipment", "weight", "date"]
    december = loads[december_columns].assign(predicted_rate=np.nan)
    result = build_features(december)
    expected = build_features(loads.drop(columns=[
        "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "market_index", "quote_signal",
    ]))
    assert_frame_equal(result, expected)
    absent = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "market_index", "quote_signal"]
    assert result[absent].isna().all().all()
    assert result.market_index_missing.eq(1).all()
    present_features = [name for name in FEATURE_COLUMNS if name not in absent + ["market_index_missing"]]
    assert_frame_equal(result[present_features], build_features(loads)[present_features])


@pytest.mark.parametrize("name", ["distance", "pickup_lat", "market_index", "quote_signal"])
@pytest.mark.parametrize("value", ["bad", "", np.inf, -np.inf])
def test_malformed_numeric_values_fail_explicitly(loads, name, value):
    with pytest.raises(ValueError, match=name):
        build_features(loads.assign(**{name: value}))


@pytest.mark.parametrize("value", ["2025-02-30", "bad", None, "", "2025-01-01T00:00:00Z", 20250101])
def test_invalid_missing_or_timezone_dates_fail_explicitly(loads, value):
    with pytest.raises(ValueError, match="[Dd]ates"):
        build_features(loads.assign(date=value))


@pytest.mark.parametrize("name", ["date", "distance", "weight"])
def test_required_source_columns(loads, name):
    with pytest.raises(ValueError, match=name):
        build_features(loads.drop(columns=name))


def test_duplicate_columns_rejected(loads):
    with pytest.raises(ValueError, match="unique"):
        build_features(pd.concat([loads, loads[["distance"]]], axis=1))


def test_sklearn_adapter_uses_same_function_and_ignores_y(loads):
    transformer = FreightFeatureTransformer()
    result = transformer.fit_transform(loads, y=loads.posted_rate)
    assert_frame_equal(result, build_features(loads))
    assert transformer.get_feature_names_out().tolist() == FEATURE_COLUMNS
    assert_frame_equal(clone(transformer).fit_transform(loads, y=-loads.posted_rate), result)


@pytest.mark.parametrize("filename,rows", [("train_test.csv", 48000), ("validation.csv", 12000), ("december_chart_inputs.csv", 31)])
def test_supplied_dataset_feature_contracts(filename, rows):
    frame = pd.read_csv(ROOT / "data" / filename)
    before = frame.copy(deep=True)
    result = build_features(frame)
    assert_frame_equal(before, frame)
    assert result.shape == (rows, len(FEATURE_COLUMNS))
    assert not np.isinf(result[NUMERIC_FEATURE_COLUMNS].to_numpy(dtype=float)).any()
    assert not result[CATEGORICAL_FEATURE_COLUMNS].isna().any().any()
    assert result.weight_missing.sum() == frame.weight.isna().sum()
    assert result.weight_negative.sum() == frame.weight.lt(0).sum()
