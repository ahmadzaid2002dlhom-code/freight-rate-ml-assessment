"""Deterministic candidate features shared by training and inference.

These transforms do not learn statistics or select a final representation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


WEIGHT_FEATURE_COLUMNS = ["weight_clean", "weight_missing", "weight_negative"]
REFERENCE_DATE = "2025-01-01"
MISSING_CATEGORY = "__MISSING__"
CATEGORICAL_FEATURE_COLUMNS = ["equipment", "pickup", "delivery", "route"]
FEATURE_COLUMNS = [
    "distance", "equipment", *WEIGHT_FEATURE_COLUMNS,
    "pickup", "delivery", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "route", "month", "day", "day_of_week", "day_of_year", "iso_week", "weekend",
    "days_since_reference_date", "sin_day_of_week", "cos_day_of_week",
    "sin_day_of_year", "cos_day_of_year", "market_index", "market_index_missing",
    "quote_signal",
]
NUMERIC_FEATURE_COLUMNS = [name for name in FEATURE_COLUMNS if name not in CATEGORICAL_FEATURE_COLUMNS]
CALENDAR_FEATURE_COLUMNS = FEATURE_COLUMNS[12:23]
COORDINATE_FEATURE_COLUMNS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
FEATURE_GROUPS = {}
_group_columns = ["distance", "equipment", *WEIGHT_FEATURE_COLUMNS, *CALENDAR_FEATURE_COLUMNS]
for _label, _added in (
    ("A", []), ("B", ["pickup", "delivery"]), ("C", COORDINATE_FEATURE_COLUMNS),
    ("D", ["route"]), ("E", ["market_index", "market_index_missing"]), ("F", ["quote_signal"]),
):
    _group_columns = [*_group_columns, *_added]
    FEATURE_GROUPS[_label] = [column for column in FEATURE_COLUMNS if column in _group_columns]


def model_feature_columns(group: str, weight_representation: str = "clean") -> list[str]:
    """Resolve the controlled feature groups, keeping a canonical order."""
    if group not in FEATURE_GROUPS or weight_representation not in ("clean", "raw"):
        raise ValueError("Unknown feature group or weight representation")
    columns = FEATURE_GROUPS[group].copy()
    if weight_representation == "raw":
        columns = ["weight_raw" if name == "weight_clean" else name for name in columns
                   if name not in ("weight_missing", "weight_negative")]
    return columns


def build_weight_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return absolute weight and original-value flags without losing rows.

    Missing cleaned weights stay NaN for a model with native missing-value
    support. A model requiring imputation must fit its imputer using only the
    applicable training fold. No full-dataset statistic is computed here.
    Malformed or infinite nonmissing values raise an explicit error.
    """
    if "weight" not in frame:
        raise ValueError("Required input column 'weight' is missing")
    original = frame["weight"]
    try:
        weight = pd.to_numeric(original, errors="raise").astype("float64")
    except (ValueError, TypeError) as exc:
        raise ValueError("Weight contains malformed nonnumeric values") from exc
    if (original.notna() & weight.isna()).any():
        raise ValueError("A nonmissing weight was converted to NaN")
    if np.isinf(weight).any():
        raise ValueError("Weight contains infinite values")
    return pd.DataFrame(
        {
            "weight_clean": weight.abs(),
            "weight_missing": original.isna().astype("uint8"),
            "weight_negative": weight.lt(0).astype("uint8"),
        },
        index=frame.index,
    )


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Keep true missing values; reject malformed nonmissing and infinite values."""
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64", name=name)
    original = frame[name]
    try:
        values = pd.to_numeric(original, errors="raise").astype("float64")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Column '{name}' contains malformed nonnumeric values") from exc
    if (original.notna() & values.isna()).any():
        raise ValueError(f"Column '{name}' converted a nonmissing value to NaN")
    if np.isinf(values).any():
        raise ValueError(f"Column '{name}' contains infinite values")
    return values


def _categorical_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(MISSING_CATEGORY, index=frame.index, dtype=object)
    # Whitespace-only strings are missing. All other labels remain unchanged.
    values = frame[name].astype("string")
    missing = values.isna() | values.str.strip().eq("").fillna(False)
    return values.mask(missing, MISSING_CATEGORY).astype(object)


def build_features(frame: pd.DataFrame, reference_date: str = REFERENCE_DATE) -> pd.DataFrame:
    """Build the ordered candidate features without fitting or modifying inputs.

    Required source columns: date, distance, weight. Missing optional location,
    equipment or market columns are explicit missing values, including December
    inputs. Numeric NaNs require native model support or a fold-fitted imputer.
    The fixed reference date makes features independent of batch composition.
    Unknown city/equipment labels remain strings; no categorical vocabulary or
    target encoding is learned. Other input columns are never passed through.
    """
    if not frame.columns.is_unique:
        raise ValueError("Input column names must be unique")
    for name in ("date", "distance", "weight"):
        if name not in frame:
            raise ValueError(f"Required input column '{name}' is missing")
    if frame["date"].map(
        lambda value: isinstance(value, (int, float, complex, np.number)) and pd.notna(value)
    ).any():
        raise ValueError("Dates must be ISO dates, not numeric timestamps")
    try:
        dates = pd.to_datetime(frame["date"], format="ISO8601", errors="raise")
        reference = pd.Timestamp(reference_date)
    except (ValueError, TypeError) as exc:
        raise ValueError("Dates and reference_date must be valid ISO dates") from exc
    if dates.isna().any() or pd.isna(reference):
        raise ValueError("Dates and reference_date must not be missing")
    if dates.dt.tz is not None or reference.tzinfo is not None:
        raise ValueError("Dates and reference_date must be timezone-naive calendar dates")
    dates = dates.dt.normalize()
    reference = reference.normalize()

    features = build_weight_features(frame)
    for name in ("distance", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
                 "market_index", "quote_signal"):
        features[name] = _numeric_column(frame, name)
    for name in ("equipment", "pickup", "delivery"):
        features[name] = _categorical_column(frame, name)
    features["route"] = features["pickup"] + " -> " + features["delivery"]
    features["market_index_missing"] = features["market_index"].isna().astype("uint8")
    features["month"] = dates.dt.month.astype("int16")
    features["day"] = dates.dt.day.astype("int16")
    features["day_of_week"] = dates.dt.dayofweek.astype("int16")
    features["day_of_year"] = dates.dt.dayofyear.astype("int16")
    features["iso_week"] = dates.dt.isocalendar().week.astype("int16")
    features["weekend"] = dates.dt.dayofweek.ge(5).astype("uint8")
    features["days_since_reference_date"] = (dates - reference).dt.days.astype("int64")
    week_phase = 2 * np.pi * features["day_of_week"] / 7
    # January 1 has phase zero. Leap years use their actual 366-day length.
    year_length = np.where(dates.dt.is_leap_year, 366, 365)
    year_phase = 2 * np.pi * (features["day_of_year"] - 1) / year_length
    features["sin_day_of_week"] = np.sin(week_phase)
    features["cos_day_of_week"] = np.cos(week_phase)
    features["sin_day_of_year"] = np.sin(year_phase)
    features["cos_day_of_year"] = np.cos(year_phase)
    return features.loc[:, FEATURE_COLUMNS]


def build_model_features(frame: pd.DataFrame, columns: list[str],
                         reference_date: str = REFERENCE_DATE) -> pd.DataFrame:
    """Select validated candidate features identically for experiments/inference."""
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Model feature list must be nonempty and unique")
    if not set(columns).issubset(set(FEATURE_COLUMNS) | {"weight_raw"}):
        raise ValueError("Model feature list contains an unapproved feature")
    features = build_features(frame, reference_date)
    if "weight_raw" in columns:
        features["weight_raw"] = _numeric_column(frame, "weight")
    return features.loc[:, columns]


class FreightFeatureTransformer(TransformerMixin, BaseEstimator):
    """Sklearn adapter for the same stateless builder; accepts raw DataFrames.

    Place any learned imputer/scaler/encoder after this step in a pipeline fitted
    only on the relevant training fold. fit does not inspect y or learn values.
    """

    def __init__(self, reference_date: str = REFERENCE_DATE):
        self.reference_date = reference_date

    def fit(self, X: pd.DataFrame, y=None):
        build_features(X, self.reference_date)
        self.feature_names_out_ = np.array(FEATURE_COLUMNS, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_out_")
        return build_features(X, self.reference_date)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()
