"""Chronology and training-only baseline behavior checks."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor

import src.evaluate as evaluation
from src.evaluate import ChronologicalFold, FOLDS, chronological_splits, merge_results, run_baselines, run_catboost

ROOT = Path(__file__).resolve().parents[1]


def test_real_chronological_boundaries_counts_and_whole_dates():
    frame = pd.read_csv(ROOT / "data/train_test.csv")
    counts = [(19110, 9696), (28806, 9671), (38477, 9523)]
    validation_dates = set()
    for (fold, train_idx, val_idx), (train_count, val_count) in zip(chronological_splits(frame), counts):
        train = frame.iloc[train_idx]
        val = frame.iloc[val_idx]
        assert (len(train), len(val)) == (train_count, val_count)
        assert train.date.min() == fold.train_start and train.date.max() == fold.train_end
        assert val.date.min() == fold.validation_start and val.date.max() == fold.validation_end
        assert train.date.max() < val.date.min()
        assert set(train.date).isdisjoint(set(val.date))
        assert validation_dates.isdisjoint(set(val.date))
        validation_dates.update(val.date)
        assert train.groupby("date").size().equals(frame.groupby("date").size().loc[sorted(train.date.unique())])
        assert val.groupby("date").size().equals(frame.groupby("date").size().loc[sorted(val.date.unique())])
    assert len(FOLDS) == 3


def test_unsorted_rows_and_duplicate_indices_do_not_change_date_membership():
    frame = pd.DataFrame({"date": ["2025-05-01", "2025-01-01", "2025-05-01", "2025-01-01"]}, index=[7, 7, 7, 7])
    _, train_idx, val_idx = next(chronological_splits(frame, FOLDS[:1]))
    assert train_idx.tolist() == [1, 3]
    assert val_idx.tolist() == [0, 2]


def test_overlapping_fold_rejected():
    fold = ChronologicalFold("invalid", "2025-01-01", "2025-05-01", "2025-05-01", "2025-06-30")
    with pytest.raises(ValueError, match="overlapping"):
        list(chronological_splits(pd.DataFrame({"date": ["2025-01-01", "2025-05-01"]}), [fold]))


def test_global_median_does_not_use_future_targets():
    frame = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-01", "2025-05-01", "2025-05-01"],
        "weight": [10000, np.nan, -20000, np.nan],
        "distance": [100, 200, 300, 400],
        "posted_rate": [1, 9, 100, 200],
        "load_id": ["A", "B", "C", "D"],
    })
    before = frame.copy(deep=True)
    results = run_baselines(frame, FOLDS[:1])
    median = results.loc[results.model.eq("global_median")].iloc[0]
    # Training median is 5; the full-data median 54.5 must never be used.
    assert median.MAE == pytest.approx((95 + 195) / 2)
    assert median.RMSE == pytest.approx(np.sqrt((95**2 + 195**2) / 2))
    assert median.train_rows == 2 and median.validation_rows == 2
    assert set(results.model) == {"global_median", "distance_linear", "ridge"}
    assert np.isfinite(results[["MAE", "RMSE", "R2"]].to_numpy()).all()
    pd.testing.assert_frame_equal(frame, before)
    assert results.feature_set.tolist() == ["training_fold_median", "distance_only", "all_26_candidates"]


def test_unlabeled_data_cannot_be_evaluated():
    with pytest.raises(ValueError, match="posted_rate"):
        run_baselines(pd.DataFrame({"date": ["2025-11-01"]}))


@pytest.mark.parametrize("value", [np.nan, np.inf, -1, 0])
def test_invalid_target_not_silently_dropped(value):
    with pytest.raises(ValueError, match="targets"):
        run_baselines(pd.DataFrame({"posted_rate": [value]}))


def test_catboost_uses_only_chronological_training_and_stopping_rows(monkeypatch):
    n = 16
    frame = pd.DataFrame({
        "date": list(pd.date_range("2025-01-01", periods=n).strftime("%Y-%m-%d"))
                + list(pd.date_range("2025-05-01", periods=n).strftime("%Y-%m-%d")),
        "distance": np.arange(2*n) * 10 + 100,
        "weight": [10000, np.nan, -32000, 40000] * 8,
        "equipment": ["Dry Van", "Reefer"] * n,
        "pickup": ["A"] * n + ["Unseen pickup"] * n,
        "delivery": ["B"] * n + ["Unseen delivery"] * n,
        "posted_rate": np.arange(2*n) * 30 + 800,
        "load_id": [f"ID-{i}" for i in range(2*n)],
    })
    calls = []

    class RecordingCatBoost(CatBoostRegressor):
        def fit(self, X, y, **kwargs):
            calls.append((X.copy(), np.array(y), kwargs))
            return super().fit(X, y, **kwargs)

    monkeypatch.setattr(evaluation, "CatBoostRegressor", RecordingCatBoost)
    result = run_catboost(
        frame, folds=FOLDS[:1],
        configurations={"catboost_test": {"depth": 2, "learning_rate": 0.1, "l2_leaf_reg": 3, "iterations": 20}},
        early_stopping_rounds=3,
    )
    X, y, kwargs = calls[0]
    eval_X, eval_y = kwargs["eval_set"]
    assert len(X) == len(eval_X) == n
    assert X.days_since_reference_date.max() < eval_X.days_since_reference_date.min()
    assert not {"load_id", "posted_rate"} & set(X)
    np.testing.assert_array_equal(y, frame.posted_rate.iloc[:n])
    np.testing.assert_array_equal(eval_y, frame.posted_rate.iloc[n:])
    assert kwargs["early_stopping_rounds"] == 3
    assert kwargs["use_best_model"] is True
    assert "pickup" in kwargs["cat_features"] and "route" in kwargs["cat_features"]
    assert set(eval_X.pickup).isdisjoint(set(X.pickup))
    row = result.iloc[0]
    assert 0 <= row.best_iteration < row.iterations_executed <= 20
    assert row.tree_count == row.best_iteration + 1
    assert row.training_seconds > 0 and row.prediction_seconds >= 0
    assert np.isfinite(row[["MAE", "RMSE", "R2"]].to_numpy(dtype=float)).all()


def test_catboost_run_replaces_same_models_and_retains_baselines():
    previous = pd.DataFrame({
        "model": ["ridge", "catboost_test"], "fold": ["fold_1", "fold_1"], "MAE": [200.0, 100.0],
    })
    executed = pd.DataFrame({
        "model": ["catboost_test"], "fold": ["fold_1"], "MAE": [90.0], "best_iteration": [7],
    })
    result = merge_results(previous, executed)
    assert result.model.tolist() == ["ridge", "catboost_test"]
    assert result.MAE.tolist() == [200.0, 90.0]
    assert pd.isna(result.loc[0, "best_iteration"])
    assert result.loc[1, "best_iteration"] == 7
