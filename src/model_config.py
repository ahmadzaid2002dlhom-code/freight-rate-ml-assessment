"""Consume the frozen model specification without tuning or production fitting."""

import json
from pathlib import Path

import numpy as np

from .evaluate import inverse_target
from .features import COORDINATE_FEATURE_COLUMNS, build_model_features, model_feature_columns

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "docs/final_model_config.json"


def load_frozen_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    if config["status"] != "frozen_before_production_training":
        raise ValueError("Model configuration has not been frozen")
    for role in ("primary", "december"):
        spec = config[role]
        expected = model_feature_columns(spec["feature_group"], spec["weight_representation"])
        if spec["feature_columns"] != expected or not set(COORDINATE_FEATURE_COLUMNS) <= set(expected):
            raise ValueError("Frozen feature list is inconsistent or drops coordinates")
        if spec["target_strategy"] not in ("direct", "log1p"):
            raise ValueError("Unrecognized target strategy")
        if spec["parameters"]["iterations"] < 1 or spec["parameters"]["random_seed"] != 42:
            raise ValueError("Invalid frozen iteration budget or seed")
        if not np.isfinite(spec["prediction_floor"]) or spec["prediction_floor"] <= 0:
            raise ValueError("Prediction floor must be finite and positive")
    if {"market_index", "quote_signal"} & set(config["december"]["feature_columns"]):
        raise ValueError("December configuration requires unavailable market inputs")
    return config


def convert_predictions(predictions, spec):
    """Return finite dollar-scale outputs with the frozen positive-domain guard."""
    rates = inverse_target(predictions, spec["target_strategy"])
    if not np.isfinite(rates).all():
        raise ValueError("Model returned nonfinite predictions")
    return np.maximum(rates, spec["prediction_floor"])


def predict_with_spec(model, frame, spec):
    features = build_model_features(frame, spec["feature_columns"], spec["reference_date"])
    rates = convert_predictions(model.predict(features), spec)
    if rates.shape != (len(frame),):
        raise ValueError("Prediction shape does not match input rows")
    return rates
