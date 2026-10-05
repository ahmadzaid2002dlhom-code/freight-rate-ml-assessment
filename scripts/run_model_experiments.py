"""Controlled development-only ablations and a frozen production configuration.

Only data/train_test.csv is loaded. No full-development production fit occurs.
Run: .venv/bin/python scripts/run_model_experiments.py --phase all
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import (CATBOOST_COMMON_PARAMETERS, FOLDS, inverse_target,
                          merge_results, run_catboost, save_report)
from src.features import (CATEGORICAL_FEATURE_COLUMNS, COORDINATE_FEATURE_COLUMNS,
                          FEATURE_GROUPS, REFERENCE_DATE, build_model_features, model_feature_columns)

EXPERIMENTS = ROOT / "reports/ablation_results.csv"
BASE_PARAMETERS = {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5}


def selection_table(results: pd.DataFrame) -> pd.DataFrame:
    means = results.groupby("model", sort=False).agg(
        MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), R2=("R2", "mean"),
        MAE_std=("MAE", "std"), feature_group=("feature_group", "first"),
        weight_representation=("weight_representation", "first"), target_strategy=("target_strategy", "first"),
    ).reset_index()
    recent = results.loc[results.fold.eq("fold_3"), ["model", "MAE", "RMSE", "R2"]].rename(
        columns={"MAE": "recent_MAE", "RMSE": "recent_RMSE", "R2": "recent_R2"})
    return means.merge(recent, on="model", validate="one_to_one")


def choose(results: pd.DataFrame, eligible_groups) -> str:
    """Mean-MAE shortlist within 1%, then emphasize recent performance."""
    table = selection_table(results)
    table = table.loc[table.feature_group.isin(eligible_groups)]
    shortlist = table.loc[table.MAE <= 1.01 * table.MAE.min()]
    return shortlist.sort_values(["recent_MAE", "MAE", "RMSE", "MAE_std", "model"]).iloc[0].model


def check_protected():
    manifest = json.loads((ROOT / "docs/supplied_file_checksums.json").read_text())
    for name, record in manifest["files"].items():
        if record["policy"] == "immutable":
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != record["sha256"]:
                raise AssertionError(f"Protected original changed: {name}")
        elif record["policy"] == "only-predicted_rate-may-change":
            from src.production import validate_december
            validate_december(pd.read_csv(ROOT / name), require_predictions=False)


def load_experiments():
    return pd.read_csv(EXPERIMENTS, float_precision="round_trip") if EXPERIMENTS.exists() else pd.DataFrame()


def checkpoint(rows: pd.DataFrame):
    existing = load_experiments()
    combined = merge_results(existing, rows) if not existing.empty else rows
    combined.to_csv(EXPERIMENTS, index=False)
    comparison = ROOT / "reports/model_comparison.csv"
    previous = pd.read_csv(comparison, float_precision="round_trip")
    merge_results(previous, rows).to_csv(comparison, index=False)


def experiment(frame, group, weight="clean", target="direct", fixed_iterations=None,
               name=None, callback=None):
    name = name or f"catboost_ablation_{group}_{weight}_{target}"
    columns = model_feature_columns(group, weight)
    rows = run_catboost(
        frame, configurations={name: BASE_PARAMETERS}, feature_columns=columns,
        feature_set=f"ablation_{group}_{weight}", target_strategy=target,
        fixed_iterations=fixed_iterations, model_callback=callback,
    )
    rows["feature_group"], rows["weight_representation"] = group, weight
    rows["experiment_phase"] = "fixed_budget_confirmation" if fixed_iterations else (
        "target_strategy" if target == "log1p" else "weight_representation" if weight == "raw" else "feature_ablation")
    diagnostic_path = ROOT / "reports/experiment_diagnostics.csv"
    diagnostics = pd.DataFrame(rows.attrs.pop("diagnostics"))
    if diagnostic_path.exists():
        previous = pd.read_csv(diagnostic_path)
        diagnostics = pd.concat([previous.loc[~previous.model.eq(name)], diagnostics], ignore_index=True)
    diagnostics.to_csv(diagnostic_path, index=False)
    checkpoint(rows)
    return rows


def run_features(frame):
    for group in "ABCDE":
        experiment(frame, group)
    # F is exactly the previously executed winning hyperparameter experiment.
    previous = pd.read_csv(ROOT / "reports/model_comparison.csv", float_precision="round_trip")
    rows = previous.loc[previous.model.eq("catboost_d6_lr005_l2_5")].copy()
    if len(rows) != 3 or set(rows.fold) != {f.name for f in FOLDS}:
        raise AssertionError("Complete matching F experiment is required for reuse")
    for params in rows.parameters:
        values = json.loads(params)
        assert all(values[key] == value for key, value in BASE_PARAMETERS.items())
        assert values["iterations"] == 1500 and values["early_stopping_rounds"] == 100
    rows["model"] = "catboost_ablation_F_clean_direct"
    rows["feature_set"] = "ablation_F_clean"
    rows["feature_group"], rows["weight_representation"], rows["target_strategy"] = "F", "clean", "direct"
    rows["feature_columns"] = json.dumps(model_feature_columns("F"))
    rows["experiment_phase"] = "feature_ablation"
    rows["result_reused_from"] = "catboost_d6_lr005_l2_5"
    checkpoint(rows)


def run_weights(frame):
    results = load_experiments()
    feature_rows = results.loc[results.experiment_phase.eq("feature_ablation")]
    scopes = {choose(feature_rows, list("CDEF")), choose(feature_rows, list("CD"))}
    groups = sorted({feature_rows.loc[feature_rows.model.eq(name), "feature_group"].iloc[0] for name in scopes})
    for group in groups:
        experiment(frame, group, weight="raw")


def run_targets(frame):
    results = load_experiments()
    direct = results.loc[results.target_strategy.eq("direct") & ~results.experiment_phase.eq("fixed_budget_confirmation")]
    scopes = {choose(direct, list("CDEF")), choose(direct, list("CD"))}
    for name in sorted(scopes):
        row = direct.loc[direct.model.eq(name)].iloc[0]
        experiment(frame, row.feature_group, row.weight_representation, "log1p")


def configuration(results, selected):
    rows = results.loc[results.model.eq(selected)]
    row = rows.iloc[0]
    iterations = int(np.median(rows.tree_count))
    columns = model_feature_columns(row.feature_group, row.weight_representation)
    params = {**CATBOOST_COMMON_PARAMETERS, **BASE_PARAMETERS, "iterations": iterations}
    params.pop("eval_metric")  # No production eval_set or stopping selection.
    return {
        "model_type": "CatBoostRegressor", "source_experiment": selected,
        "feature_group": row.feature_group, "weight_representation": row.weight_representation,
        "target_strategy": row.target_strategy, "feature_columns": columns,
        "categorical_features": [name for name in CATEGORICAL_FEATURE_COLUMNS if name in columns],
        "reference_date": REFERENCE_DATE, "parameters": params,
        "iteration_rule": "median retained tree count across three selected-model chronological folds",
        "prediction_floor": 0.01,
    }


def run_freeze(frame, resume=False):
    results = load_experiments()
    eligible = results.loc[~results.experiment_phase.eq("fixed_budget_confirmation")]
    table = selection_table(eligible)
    # Compare close adaptive contenders at fixed production-ready budgets.
    # The 3% finalist rule is fixed before these confirmation fits run.
    candidates = set()
    for groups in (list("CDEF"), list("CD")):
        subset = table.loc[table.feature_group.isin(groups)]
        candidates.update(subset.loc[subset.MAE <= 1.03 * subset.MAE.min(), "model"])
    configurations, confirmations, stress = {}, [], {}
    for selected in sorted(candidates):
        spec = configuration(eligible, selected)
        name = f"catboost_fixed_{spec['feature_group']}_{spec['weight_representation']}_{spec['target_strategy']}"
        spec["confirmation_model"] = name
        stress[name] = []

        def robustness(model, fold, raw_holdout):
            if fold.name != "fold_3":
                return
            base = raw_holdout.iloc[:128].copy()
            cases = {
                "unseen_cities_routes_equipment": base.assign(pickup="NEW_ORIGIN", delivery="NEW_DESTINATION", equipment="NEW_EQUIPMENT"),
                "missing_weight": base.assign(weight=np.nan),
                "negative_weight": base.assign(weight=-base.weight.abs().fillna(32000)),
                "unavailable_market_and_quote": base.assign(market_index=np.nan, quote_signal=np.nan),
                "missing_categories_and_coordinates": base.assign(pickup=None, delivery=None, equipment=None, **{column: np.nan for column in COORDINATE_FEATURE_COLUMNS}),
            }
            for case, rows in cases.items():
                predictions = inverse_target(model.predict(build_model_features(rows, spec["feature_columns"])), spec["target_strategy"])
                finite_positive = bool(np.isfinite(predictions).all() and (predictions > 0).all())
                if not finite_positive:
                    raise AssertionError(f"Finalist robustness case failed: {name}/{case}")
                stress[name].append({"case": case, "rows": len(rows), "finite_positive": finite_positive})

        cached = results.loc[results.model.eq(name)]
        if resume and len(cached) == 3:
            for row in cached.itertuples():
                assert json.loads(row.feature_columns) == spec["feature_columns"]
                recorded = json.loads(row.parameters)
                assert all(recorded[key] == value for key, value in spec["parameters"].items())
                assert row.target_strategy == spec["target_strategy"]
                assert row.tree_count == spec["parameters"]["iterations"]
            confirmation = cached.copy()
        else:
            confirmation = experiment(
                frame, spec["feature_group"], spec["weight_representation"], spec["target_strategy"],
                fixed_iterations=spec["parameters"]["iterations"], name=name, callback=robustness,
            )
        spec["local_metrics_mean"] = confirmation[["MAE", "RMSE", "R2"]].mean().to_dict()
        spec["local_metrics_primary_holdout"] = confirmation.loc[confirmation.fold.eq("fold_3"), ["MAE", "RMSE", "R2"]].iloc[0].to_dict()
        spec["robustness_checks"] = stress[name]
        configurations[name] = spec
        confirmations.append(confirmation)
    fixed_results = pd.concat(confirmations, ignore_index=True)
    primary = configurations[choose(fixed_results, list("CDEF"))]
    chart = configurations[choose(fixed_results, list("CD"))]
    if not primary["robustness_checks"]:
        # Recover after an interrupted report step without repeating comparison fits.
        spec = primary
        name = primary["confirmation_model"]
        recovered = run_catboost(
            frame, folds=FOLDS[-1:], configurations={name:BASE_PARAMETERS},
            feature_columns=spec["feature_columns"], target_strategy=spec["target_strategy"],
            fixed_iterations=spec["parameters"]["iterations"], model_callback=robustness,
        )
        expected = fixed_results.loc[fixed_results.model.eq(name) & fixed_results.fold.eq("fold_3")]
        np.testing.assert_allclose(recovered[["MAE","RMSE","R2"]],expected[["MAE","RMSE","R2"]],rtol=1e-10)
        primary["robustness_checks"] = stress[name]
    final_config = {
        "status": "frozen_before_production_training", "primary": primary, "december": chart,
        "selection_data": "data/train_test.csv only",
        "selection_rule": "adaptive mean-MAE contenders within 3% confirmed at fixed median budgets; fixed mean-MAE shortlist within 1%, prioritize Sep-Oct MAE, then mean MAE, RMSE, MAE variability",
        "validation_limit": "Chronological folds reused for stopping and selection; not an independent test",
    }
    (ROOT / "docs/final_model_config.json").write_text(json.dumps(final_config, indent=2) + "\n")
    pd.DataFrame(primary["robustness_checks"]).to_csv(ROOT / "reports/selected_model_robustness.csv", index=False)
    print("FROZEN: " + json.dumps(primary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["features", "weights", "targets", "freeze", "all"], default="all")
    parser.add_argument("--resume", action="store_true", help="Reuse verified fixed-budget comparison records after interrupted reporting")
    args = parser.parse_args()
    check_protected()
    frame = pd.read_csv(ROOT / "data/train_test.csv")
    if len(frame) != 48000:
        raise AssertionError("Expected all 48,000 development rows")
    for phase, function in [("features", run_features), ("weights", run_weights), ("targets", run_targets), ("freeze", run_freeze)]:
        if args.phase in (phase, "all"):
            if phase == "freeze":
                function(frame, resume=args.resume)
            else:
                function(frame)
    check_protected()


if __name__ == "__main__":
    main()
