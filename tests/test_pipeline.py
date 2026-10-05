"""Exercise pre-training checks on real inputs and deliberately malformed copies."""

import numpy as np
import pandas as pd
import pytest

from run_pipeline import require_input_contracts, validate_inputs, verify_outputs
from scripts.check_inputs import INPUTS
from scripts.run_model_experiments import check_protected
from src.production import ROOT, DECEMBER_COLUMNS, sha256


@pytest.fixture(scope="module")
def supplied_frames():
    return {name: pd.read_csv(ROOT / "data" / f"{name}.csv") for name in INPUTS}


def test_real_pipeline_preflight_and_integrity_are_read_only():
    paths = [ROOT / "data" / f"{name}.csv" for name in INPUTS]
    before = {str(path): sha256(path) for path in paths}
    count, shapes, december_inputs = validate_inputs()
    assert count > 0
    assert shapes == {"development_primary": [48000, 26], "development_december": [48000, 22],
                      "final_inference": [12000, 26], "fixed_december": [31, 22]}
    assert verify_outputs(december_inputs)["validation_rows"] == 12000
    assert before == {str(path): sha256(path) for path in paths}


def test_preflight_retains_real_missing_and_negative_observations(supplied_frames):
    require_input_contracts(supplied_frames)
    train = supplied_frames["train_test"]
    assert train.weight.isna().sum() == 300
    assert train.weight.lt(0).sum() == 292
    assert train.market_index.isna().sum() == 374
    assert len(train) == 48000


@pytest.mark.parametrize("dataset,column,value", [
    ("train_test", "weight", "malformed"),
    ("train_test", "weight", np.inf),
    ("train_test", "market_index", "malformed"),
    ("train_test", "posted_rate", np.nan),
    ("validation", "date", "2025-09-01"),
    ("validation_predictions_template", "load_id", "unexpected-ID"),
])
def test_preflight_rejects_bad_data_before_training(supplied_frames, dataset, column, value):
    changed = {**supplied_frames, dataset: supplied_frames[dataset].copy(deep=True)}
    # Object dtype permits intentionally malformed strings without pandas coercion warnings.
    changed[dataset][column] = changed[dataset][column].astype(object)
    changed[dataset].loc[0, column] = value
    with pytest.raises(ValueError, match="Input validation failed"):
        require_input_contracts(changed)


def test_preflight_rejects_schema_changes(supplied_frames):
    changed = {**supplied_frames, "validation": supplied_frames["validation"].drop(columns="weight")}
    with pytest.raises(ValueError, match="schema"):
        require_input_contracts(changed)


def test_integrity_checks_reject_december_input_reordering():
    december = pd.read_csv(ROOT / "data/december_chart_inputs.csv")
    original = december[DECEMBER_COLUMNS[:-1]].iloc[::-1].to_dict("records")
    with pytest.raises(ValueError, match="ordering changed"):
        verify_outputs(original)


def test_research_guard_accepts_permitted_december_prediction_fill():
    check_protected()


@pytest.mark.parametrize("change", ["reorder", "date_format"])
def test_preflight_preserves_original_december_dates(supplied_frames, change):
    december = supplied_frames["december_chart_inputs"].copy(deep=True)
    if change == "reorder":
        december = december.iloc[::-1].reset_index(drop=True)
    else:
        december.loc[0, "date"] = "2025-12-1"
    with pytest.raises(ValueError, match="original ordered dates"):
        require_input_contracts({**supplied_frames, "december_chart_inputs": december})


def test_unseen_analysis_guard_accepts_current_december_rates():
    from scripts.check_unseen_categories import check_analysis_inputs
    before = check_analysis_inputs()
    assert before["data/december_chart_inputs.csv"] == sha256(ROOT / "data/december_chart_inputs.csv")
    assert before["data/train_test.csv"] == sha256(ROOT / "data/train_test.csv")
