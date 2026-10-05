"""Feature ablation, dollar-scale target handling, and frozen-budget safeguards."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluate import DollarMAE, FOLDS, inverse_target, merge_results, run_catboost
from src.features import (COORDINATE_FEATURE_COLUMNS, FEATURE_COLUMNS, FEATURE_GROUPS,
                          build_model_features, model_feature_columns)
from src.model_config import convert_predictions, load_frozen_config

ROOT = Path(__file__).resolve().parents[1]


def test_feature_ablation_groups_are_nested_and_full_group_is_exact():
    groups = [set(FEATURE_GROUPS[name]) for name in "ABCDEF"]
    assert all(first < second for first, second in zip(groups, groups[1:]))
    assert FEATURE_GROUPS["F"] == FEATURE_COLUMNS
    assert "quote_signal" not in groups[-2]
    assert groups[-2] - groups[-3] == {"market_index", "market_index_missing"}
    assert set(COORDINATE_FEATURE_COLUMNS) <= groups[2]
    assert all(not {"load_id", "posted_rate"} & group for group in groups)


def test_raw_weight_is_preserved_without_changing_other_features():
    frame = pd.DataFrame({"date": ["2025-01-01"] * 3, "distance": [360] * 3,
                          "weight": [-32000, np.nan, 10000]})
    raw = build_model_features(frame, model_feature_columns("C", "raw"))
    clean = build_model_features(frame, model_feature_columns("C", "clean"))
    np.testing.assert_allclose(raw.weight_raw, [-32000, np.nan, 10000], equal_nan=True)
    np.testing.assert_allclose(clean.weight_clean, [32000, np.nan, 10000], equal_nan=True)
    assert not {"weight_clean", "weight_missing", "weight_negative"} & set(raw)
    common = [name for name in raw if name != "weight_raw"]
    pd.testing.assert_frame_equal(raw[common], clean[common])
    for forbidden in ["load_id", "posted_rate", "predicted_rate"]:
        with pytest.raises(ValueError, match="unapproved"):
            build_model_features(frame, [forbidden])


def test_log_target_stopping_metric_is_dollar_mae_not_log_mae():
    metric = DollarMAE()
    truth = np.log1p([100, 1000])
    predicted = np.log1p([110, 800])
    error, count = metric.evaluate([predicted], truth, None)
    assert metric.get_final_error(error, count) == pytest.approx(105)
    error, count = metric.evaluate([predicted], truth, [2.0, 1.0])
    assert metric.get_final_error(error, count) == pytest.approx(220 / 3)
    np.testing.assert_allclose(inverse_target(predicted, "log1p"), [110, 800])


def test_real_log_target_fit_uses_dollar_metrics_and_fixed_budget_without_eval_set():
    n = 12
    frame = pd.DataFrame({
        "date": list(pd.date_range("2025-01-01", periods=n).strftime("%Y-%m-%d"))
                + list(pd.date_range("2025-05-01", periods=n).strftime("%Y-%m-%d")),
        "distance": np.arange(2*n) * 10 + 100,
        "weight": [10000, np.nan, -32000] * 8,
        "equipment": ["Dry Van", "Reefer"] * n,
        "posted_rate": np.arange(2*n) * 20 + 800,
    })
    config = {"catboost_test_log": {"depth": 2, "iterations": 12, "learning_rate": 0.1, "l2_leaf_reg": 5}}
    stopping = run_catboost(frame, folds=FOLDS[:1], configurations=config,
                            feature_columns=model_feature_columns("A"), target_strategy="log1p", early_stopping_rounds=2)
    assert stopping.MAE.iloc[0] > 1  # Metrics are dollars, not log-space values.
    assert "DollarMAE" in stopping.parameters.iloc[0]
    fixed = run_catboost(frame, folds=FOLDS[:1], configurations=config,
                         feature_columns=model_feature_columns("A"), target_strategy="log1p", fixed_iterations=5)
    assert fixed.tree_count.iloc[0] == fixed.iterations_executed.iloc[0] == 5
    assert pd.isna(fixed.best_iteration.iloc[0])
    parameters = json.loads(fixed.parameters.iloc[0])
    assert not parameters["use_best_model"] and parameters["early_stopping_rounds"] is None
    assert isinstance(stopping.attrs["diagnostics"], list)
    assert len(pd.concat([stopping, fixed], ignore_index=True)) == 2


def test_frozen_config_contract_when_selection_is_present():
    path = ROOT / "docs/final_model_config.json"
    if not path.exists():
        pytest.skip("Selection experiments have not finished yet")
    config = json.loads(path.read_text())
    assert config["status"] == "frozen_before_production_training"
    for name in ("primary", "december"):
        model = config[name]
        assert model["feature_columns"] == model_feature_columns(model["feature_group"], model["weight_representation"])
        assert set(COORDINATE_FEATURE_COLUMNS) <= set(model["feature_columns"])
        assert model["parameters"]["random_seed"] == 42
        assert model["parameters"]["iterations"] > 0
        assert not {"load_id", "posted_rate"} & set(model["feature_columns"])
    assert not {"market_index", "quote_signal"} & set(config["december"]["feature_columns"])
    assert load_frozen_config(path) == config


def test_frozen_prediction_guard_converts_target_and_rejects_nonfinite_outputs():
    spec = {"target_strategy": "log1p", "prediction_floor": 0.01}
    np.testing.assert_allclose(convert_predictions(np.log1p([0, 100, 1000]), spec), [0.01, 100, 1000])
    for value in (np.nan, np.inf):
        with pytest.raises((ValueError, FloatingPointError)):
            convert_predictions([value], spec)
    np.testing.assert_array_equal(convert_predictions([-1,0,100], {"target_strategy":"direct", "prediction_floor":0.01}), [0.01,0.01,100])


def test_fixed_budget_record_keeps_nullable_iterations_without_concat_warning():
    old = pd.DataFrame({"model":["catboost_old"],"fold":["fold_1"],"best_iteration":[20.0]})
    new = pd.DataFrame({"model":["catboost_fixed"],"fold":["fold_1"],"best_iteration":[None]})
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        combined = merge_results(old,new)
    assert str(combined.best_iteration.dtype)=="Int64"
    assert pd.isna(combined.best_iteration.iloc[1])
