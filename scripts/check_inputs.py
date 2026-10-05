"""Audit supplied inputs without modifying data or fitting a model.

Run from the project root: python scripts/check_inputs.py
The default project root is inferred from this file, so other working directories
also work. Contract failures are included in the report and produce exit code 1.
Missing predictors and negative weights are reported, not silently removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_weight_features

TARGET = "posted_rate"
FEATURE_COLUMNS = [
    "pickup", "delivery", "pickup_lat", "pickup_lon", "delivery_lat",
    "delivery_lon", "distance", "equipment", "weight", "date",
    "market_index", "quote_signal",
]
NUMERIC_COLUMNS = {
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance",
    "weight", "market_index", "quote_signal", TARGET, "predicted_rate",
}
DECEMBER_COLUMNS = [
    "pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate",
]
INPUTS = {
    "train_test": (48_000, ["load_id", *FEATURE_COLUMNS, TARGET]),
    "validation": (12_000, ["load_id", *FEATURE_COLUMNS]),
    "validation_predictions_template": (12_000, ["load_id", "predicted_rate"]),
    "december_chart_inputs": (31, DECEMBER_COLUMNS),
}
EXPECTED_IDS = {f"TE-{number:06d}" for number in range(1, 12_001)}


def format_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.6f}".rstrip("0").rstrip(".")
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def table(headers: Iterable[str], rows: Iterable[Iterable[object]]) -> str:
    headers = list(headers)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(format_value(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def numeric_values(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def finite_values(series: pd.Series) -> pd.Series:
    values = numeric_values(series)
    return values[np.isfinite(values)]


def date_values(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="coerce")


def routes(frame: pd.DataFrame) -> pd.Series:
    return frame["pickup"].astype("string").fillna("<MISSING>") + " -> " + frame["delivery"].astype("string").fillna("<MISSING>")


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_contracts(frames: dict[str, pd.DataFrame]) -> list[tuple[str, bool, str]]:
    """Return named checks instead of hiding all but the first failure."""
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append((name, bool(condition), detail))

    for name, (expected_rows, columns) in INPUTS.items():
        frame = frames[name]
        check(f"{name}: shape", frame.shape == (expected_rows, len(columns)), f"Observed {frame.shape}; expected {(expected_rows, len(columns))}")
        check(f"{name}: ordered schema", frame.columns.tolist() == columns, "Compare exact column names and order")
        if "load_id" in frame:
            check(f"{name}: unique nonmissing IDs", not frame["load_id"].isna().any() and not frame["load_id"].duplicated().any(), "Identifiers must be unique and present")
        if "date" in frame:
            check(f"{name}: valid dates", date_values(frame).notna().all(), "Expected-format date parsing; no missing/unparseable values")
        for column in frame.columns:
            if column in NUMERIC_COLUMNS or pd.api.types.is_numeric_dtype(frame[column]):
                values = numeric_values(frame[column])
                malformed = frame[column].notna() & values.isna()
                check(f"{name}.{column}: numeric validity", not malformed.any() and not np.isinf(values).any(), f"Malformed nonmissing tokens={int(malformed.sum())}; infinities={int(np.isinf(values).sum())}")

    train, validation, template, december = (frames[name] for name in INPUTS)
    check("Target is development-only", TARGET in train and TARGET not in validation, f"Target={TARGET}")
    if TARGET in train:
        target = numeric_values(train[TARGET])
        check("Target present, finite, positive", np.isfinite(target).all() and target.gt(0).all(), "No unusable supervised labels")
    for name, start, end in [
        ("train_test", "2025-01-01", "2025-10-31"),
        ("validation", "2025-11-01", "2025-12-31"),
    ]:
        if "date" in frames[name]:
            dates = date_values(frames[name])
            check(f"{name}: expected date range", dates.min() == pd.Timestamp(start) and dates.max() == pd.Timestamp(end), f"Expected {start} through {end}")
    if "date" in train and "date" in validation:
        check("Development precedes final inference", date_values(train).max() < date_values(validation).min(), "Strict separation of labeled and final periods")
    if "load_id" in validation and "load_id" in template:
        validation_ids, template_ids = set(validation["load_id"]), set(template["load_id"])
        check("Template and final inference ID sets", validation_ids == template_ids, f"Missing from template={len(validation_ids-template_ids)}; extra in template={len(template_ids-validation_ids)}")
        check("Official final ID set", validation_ids == EXPECTED_IDS, "Exactly TE-000001 through TE-012000")
    if "load_id" in train and "load_id" in validation:
        check("Development/final ID separation", not set(train["load_id"]) & set(validation["load_id"]), "No shared load identifiers")
    if "predicted_rate" in template:
        check("Original template remains blank", template["predicted_rate"].isna().all(), "Final predictions must be saved separately")
    if all(column in december for column in DECEMBER_COLUMNS):
        dates = date_values(december)
        expected_dates = set(pd.date_range("2025-12-01", "2025-12-31"))
        check("December dates", len(december) == 31 and not dates.duplicated().any() and set(dates) == expected_dates, "One row per day of December 2025")
        fixed = december["pickup"].eq("Lexington") & december["delivery"].eq("Fort Wayne") & december["equipment"].eq("Dry Van")
        fixed &= np.isclose(numeric_values(december["distance"]), 360) & np.isclose(numeric_values(december["weight"]), 32_000)
        check("December fixed inputs", fixed.all(), "Lexington → Fort Wayne; 360 miles; Dry Van; 32,000 lb")
        predictions = numeric_values(december["predicted_rate"])
        check("December output state", december["predicted_rate"].isna().all() or (np.isfinite(predictions).all() and predictions.gt(0).all()), "Allow original blank output or complete finite positive predictions")
    return checks


def describe_frame(name: str, frame: pd.DataFrame) -> list[str]:
    lines = [f"## data/{name}.csv", f"Shape: **{len(frame):,} rows × {len(frame.columns)} columns**."]
    column_rows = []
    for column in frame.columns:
        series = frame[column]
        numeric = column in NUMERIC_COLUMNS or pd.api.types.is_numeric_dtype(series)
        values = numeric_values(series) if numeric else None
        malformed = int((series.notna() & values.isna()).sum()) if numeric else "—"
        infinities = int(np.isinf(values).sum()) if numeric else "—"
        column_rows.append([column, str(series.dtype), int(series.isna().sum()), 100 * series.isna().mean(), int(series.nunique(dropna=True)), malformed, infinities])
    lines.append(table(["Column", "Inferred dtype", "Missing", "Missing %", "Distinct nonmissing", "Malformed numeric", "Infinities"], column_rows))
    duplicate_without_id = int(frame.drop(columns=["load_id"], errors="ignore").duplicated().sum())
    lines.append(f"Exact duplicate rows: **{int(frame.duplicated().sum()):,}**. Duplicate rows after excluding `load_id`: **{duplicate_without_id:,}**.")
    if name == "validation_predictions_template":
        lines.append("Excluding IDs from a blank prediction template leaves repeated empty cells; this does not indicate duplicate loads.")
    if "load_id" in frame:
        ids = frame["load_id"]
        lines.append(f"IDs: **{ids.nunique():,} unique**, **{int(ids.isna().sum()):,} missing**, **{int(ids.duplicated().sum()):,} duplicate occurrences after the first**.")
    if "date" in frame:
        dates = date_values(frame)
        valid = dates.dropna()
        if len(valid):
            absent_days = pd.date_range(valid.min(), valid.max()).difference(pd.DatetimeIndex(valid.unique()))
            lines.append(f"Dates: **{valid.min():%Y-%m-%d} through {valid.max():%Y-%m-%d}**; **{valid.nunique()} distinct dates**; **{int(dates.isna().sum())} missing/invalid dates**; **{len(absent_days)} absent calendar days within the observed range**. Rows sorted chronologically: **{dates.is_monotonic_increasing}**.")
        else:
            lines.append("No valid dates; see failed contract checks.")
    lines.append(f"Supervised target `{TARGET}` present: **{TARGET in frame}**.")

    numeric_rows = []
    for column in frame.columns:
        if column in NUMERIC_COLUMNS or pd.api.types.is_numeric_dtype(frame[column]):
            values = finite_values(frame[column])
            numeric_rows.append([column, len(values), values.min(), values.max(), values.mean(), values.std(), values.median()])
    if numeric_rows:
        lines.extend(["### Numeric ranges and summaries", table(["Column", "Finite count", "Minimum", "Maximum", "Mean", "Sample SD", "Median"], numeric_rows)])
    categorical_rows = []
    for column in frame.select_dtypes(include=["object", "string", "category"]):
        values = frame[column].astype("string")
        role = "identifier" if column == "load_id" else "calendar" if column == "date" else "categorical predictor"
        blank = values.str.strip().eq("").fillna(False)
        padded = values.notna() & values.ne(values.str.strip()).fillna(False)
        categorical_rows.append([column, role, values.nunique(), int(values.isna().sum()), int(blank.sum()), int(padded.sum())])
    if categorical_rows:
        lines.extend(["### Categorical/string cardinality", table(["Column", "Role", "Distinct", "Missing", "Blank", "Whitespace padded"], categorical_rows)])
    if "predicted_rate" in frame:
        lines.append("Blank `predicted_rate` cells are expected in the original supplied inputs. They are output placeholders, not missing predictor measurements.")
    return lines


def target_distribution(train: pd.DataFrame) -> list[str]:
    if TARGET not in train:
        return ["## Target distribution", "Target missing; see failed contract checks."]
    target = finite_values(train[TARGET])
    rows = [["Count", len(target)], ["Minimum", target.min()], ["Maximum", target.max()], ["Mean", target.mean()], ["Median", target.median()], ["Sample standard deviation", target.std()]]
    rows.extend([f"Percentile {int(q*100)}", target.quantile(q)] for q in [.01, .05, .25, .75, .95, .99])
    rows.append(["Finite nonpositive targets", int(target.le(0).sum())])
    lines = ["## Target distribution", table(["Statistic", "posted_rate"], rows)]
    edges = [0, 500, 1000, 2000, 3000, 5000, 7500, 10000, np.inf]
    labels = ["(0, 500]", "(500, 1,000]", "(1,000, 2,000]", "(2,000, 3,000]", "(3,000, 5,000]", "(5,000, 7,500]", "(7,500, 10,000]", ">10,000"]
    counts = pd.cut(target, bins=edges, labels=labels).value_counts(sort=False)
    lines.append(table(["Rate interval ($)", "Rows", "Share of finite targets %"], [[label, int(counts[label]), 100 * counts[label] / len(target) if len(target) else None] for label in labels]))
    lines.append("The interval table covers positive finite targets; any finite nonpositive values are counted above and fail the target contract.")
    if len(target):
        q1, q3 = target.quantile([.25, .75])
        upper_fence = q3 + 1.5 * (q3 - q1)
        lines.append(f"Upper 1.5×IQR fence: **{upper_fence:,.2f}**; **{int(target.gt(upper_fence).sum()):,}** targets above it. This is a statistical flag, not evidence of invalidity; no observations were removed.")
    distance = numeric_values(train["distance"])
    target_values = numeric_values(train[TARGET])
    eligible = np.isfinite(target_values) & np.isfinite(distance) & distance.gt(0)
    rate_per_mile = target_values[eligible] / distance[eligible]
    lines.append(table(["Rate per mile", "Value"], [["Eligible rows", len(rate_per_mile)], ["Minimum", rate_per_mile.min()], ["Median", rate_per_mile.median()], ["95th percentile", rate_per_mile.quantile(.95)], ["99th percentile", rate_per_mile.quantile(.99)], ["Maximum", rate_per_mile.max()]]))
    return lines


def ks_distance(first: pd.Series, second: pd.Series) -> float | None:
    """Empirical CDF distance; descriptive statistic, not a hypothesis test."""
    if first.empty or second.empty:
        return None
    a, b = np.sort(first.to_numpy()), np.sort(second.to_numpy())
    grid = np.unique(np.concatenate([a, b]))
    return float(np.max(np.abs(np.searchsorted(a, grid, side="right") / len(a) - np.searchsorted(b, grid, side="right") / len(b))))


def category_comparison(first: pd.Series, second: pd.Series) -> tuple[pd.DataFrame, float]:
    first = first.astype("string").fillna("<MISSING>")
    second = second.astype("string").fillna("<MISSING>")
    counts = pd.concat([first.value_counts().rename("train_rows"), second.value_counts().rename("final_rows")], axis=1).fillna(0).astype(int)
    counts["train_pct"] = 100 * counts["train_rows"] / len(first)
    counts["final_pct"] = 100 * counts["final_rows"] / len(second)
    counts["delta_pp"] = counts["final_pct"] - counts["train_pct"]
    tv = float(counts["delta_pp"].abs().sum() / 200)
    return counts, tv


def weight_investigation(train: pd.DataFrame, validation: pd.DataFrame) -> list[str]:
    """Descriptive analysis only; do not select a representation or fit a model."""
    lines = ["## In-depth weight investigation", "All original observations are retained. Final inference contributes only weight-quality descriptions; relationships with `posted_rate` use labeled development data exclusively."]
    summary_rows, evidence_rows, feature_rows, equipment_rows = [], [], [], []
    for name, frame in [("Development", train), ("Final inference", validation)]:
        weight = numeric_values(frame["weight"])
        positive = weight[np.isfinite(weight) & weight.gt(0)]
        negative_abs = weight[np.isfinite(weight) & weight.lt(0)].abs()
        for label, values in [("Positive", positive), ("Negative, absolute", negative_abs)]:
            summary_rows.append([name, label, len(values), values.min(), values.max(), values.mean(), values.std(), values.median(), values.quantile(.05), values.quantile(.95)])
        outside = int((~negative_abs.between(positive.min(), positive.max())).sum()) if len(positive) else len(negative_abs)
        evidence_rows.append([name, ks_distance(positive, negative_abs), outside, int(negative_abs.isin(positive).sum()), int(weight.eq(0).sum()), int((finite_values(weight) % 1 != 0).sum())])
        try:
            features = build_weight_features(frame)
        except ValueError as exc:
            lines.append(f"Candidate feature generation failed explicitly for {name}: {exc}. No invalid value was silently filled or removed.")
        else:
            assert len(features) == len(frame) and features.index.equals(frame.index)
            feature_rows.append([name, len(features), int(features["weight_clean"].isna().sum()), int(features["weight_missing"].sum()), int(features["weight_negative"].sum()), str(features["weight_clean"].dtype), str(features["weight_missing"].dtype)])
        for equipment, group in frame.groupby("equipment", dropna=False):
            values = numeric_values(group["weight"])
            pos = values[np.isfinite(values) & values.gt(0)]
            neg = values[np.isfinite(values) & values.lt(0)].abs()
            equipment_rows.append([name, equipment, len(group), int(group["weight"].isna().sum()), len(neg), 100*len(neg)/len(group), pos.mean(), neg.mean(), int((~neg.between(pos.min(), pos.max())).sum()) if len(pos) else len(neg)])
    lines.extend(["### Positive weights versus absolute negative weights", table(["Dataset", "Original status", "Measured rows", "Abs min lb", "Abs max lb", "Abs mean lb", "Abs sample SD", "Abs median lb", "Abs p05", "Abs p95"], summary_rows), table(["Dataset", "KS D: positive vs negative magnitudes", "Negative magnitudes outside positive range", "Negative magnitudes appearing among positives", "Zero weights", "Finite noninteger weights"], evidence_rows), "KS D is an empirical distribution difference, not proof that the groups are identical. Repeated integer magnitudes do not identify paired copies of the same load. Plausible magnitudes and overlapping distributions support sign corruption as a possibility; the supplied files do not establish the corruption mechanism or intended correction.", "### Equipment distribution of weight issues", table(["Dataset", "Equipment", "Rows", "Missing weight", "Negative weight", "Negative %", "Positive mean lb", "Negative abs mean lb", "Negative magnitudes outside equipment positive range"], equipment_rows)])

    weight = numeric_values(train["weight"])
    target = numeric_values(train[TARGET])
    distance = numeric_values(train["distance"])
    rate_per_mile = (target / distance).where(np.isfinite(target) & np.isfinite(distance) & distance.gt(0))
    masks = {"Positive": np.isfinite(weight) & weight.gt(0), "Negative": np.isfinite(weight) & weight.lt(0), "Missing": train["weight"].isna()}
    target_rows = []
    for label, mask in masks.items():
        rates, rpm = finite_values(target[mask]), finite_values(rate_per_mile[mask])
        target_rows.append([label, int(mask.sum()), len(rates), rates.mean(), rates.median(), rates.min(), rates.max(), rpm.mean(), rpm.median(), int(rates.gt(10_000).sum()), finite_values(distance[mask]).mean()])
    lines.extend(["### Relationship with labeled rates", table(["Weight status", "Rows", "Finite targets", "Rate mean $", "Rate median $", "Rate min $", "Rate max $", "Mean $/mile", "Median $/mile", "Rates > $10,000", "Distance mean miles"], target_rows)])
    negative, positive = masks["Negative"], masks["Positive"]
    lines.append(f"Negative versus positive `posted_rate` empirical KS D: **{format_value(ks_distance(finite_values(target[negative]), finite_values(target[positive])))}**. Their rates are ordinary positive freight rates rather than negative charges. Similar central values support retaining these rows, but do not establish equivalence or independence of the corruption process.")

    correlation_rows = []
    representations = {
        "Raw weight, measured rows": weight,
        "Absolute weight, measured rows": weight.abs(),
        "Positive weight only": weight.where(positive),
        "Absolute negative weight only": weight.abs().where(negative),
    }
    for representation, values in representations.items():
        for outcome, outcome_values in [(TARGET, target), ("rate_per_mile", rate_per_mile)]:
            pairs = pd.DataFrame({"weight": values, "outcome": outcome_values})
            pairs = pairs[np.isfinite(pairs).all(axis=1)]
            pearson = pairs["weight"].corr(pairs["outcome"]) if len(pairs) > 1 else None
            spearman = pairs["weight"].rank(method="average").corr(pairs["outcome"].rank(method="average")) if len(pairs) > 1 else None
            correlation_rows.append([representation, outcome, len(pairs), pearson, spearman])
    lines.extend([table(["Representation", "Outcome", "Complete pairs", "Pearson", "Spearman"], correlation_rows), "Correlations use the same finite pairs for both variables; missing-weight rows remain in the dataset but cannot contribute to measured-weight correlations. Spearman is Pearson correlation of average ranks within those pairs. The marginal relationship with total rate is weak; rate per mile has a positive rank association with weight. These are descriptive associations, not model validation results or a causal estimate. Distance, equipment, lane, date, and other signals can affect the association."])

    equipment_target_rows, within_equipment_rows = [], []
    for equipment in sorted(train["equipment"].dropna().unique()):
        for label, status_mask in masks.items():
            mask = status_mask & train["equipment"].eq(equipment)
            values = finite_values(rate_per_mile[mask])
            equipment_target_rows.append([equipment, label, int(mask.sum()), finite_values(target[mask]).median(), values.median(), finite_values(distance[mask]).mean()])
        mask = positive & train["equipment"].eq(equipment) & np.isfinite(rate_per_mile)
        within_equipment_rows.append([equipment, int(mask.sum()), weight[mask].rank().corr(rate_per_mile[mask].rank())])
    lines.extend(["### Equipment-stratified development relationships", table(["Equipment", "Weight status", "Rows", "Rate median $", "Median $/mile", "Distance mean miles"], equipment_target_rows), table(["Equipment: positive-weight rows", "Complete pairs", "Spearman weight vs $/mile"], within_equipment_rows), "Equipment stratification provides descriptive context; it does not fully control for distance, route, date, or other confounding variables. Small negative groups and high-rate tails limit stronger conclusions."])

    weight_bin_rows = []
    bands = pd.cut(weight.abs(), [0, 15_000, 25_000, 35_000, 45_000, np.inf], labels=["(0, 15,000]", "(15,000, 25,000]", "(25,000, 35,000]", "(35,000, 45,000]", ">45,000"])
    for band in bands.cat.categories:
        for label in ["Positive", "Negative"]:
            mask = bands.eq(band) & masks[label]
            weight_bin_rows.append([band, label, int(mask.sum()), finite_values(target[mask]).median(), finite_values(rate_per_mile[mask]).median(), finite_values(distance[mask]).mean()])
    lines.extend(["### Fixed magnitude bands in development", table(["Absolute weight lb", "Original status", "Rows", "Rate median $", "Median $/mile", "Distance mean miles"], weight_bin_rows), "Bands are fixed descriptive ranges, not fitted preprocessing. Missing weights have no band and are summarized separately above."])

    missing_market = train["market_index"].isna()
    lines.append(f"Development overlap with missing market index: **{int((masks['Missing'] & missing_market).sum())} missing-weight rows** and **{int((masks['Negative'] & missing_market).sum())} negative-weight rows**. Monthly missing/negative counts appear in the monthly audit table; the problems are not confined to one development month.")
    lines.extend(["### Implemented candidate features and missing strategy", table(["Dataset", "Retained rows", "weight_clean NaN", "weight_missing sum", "weight_negative sum", "Clean dtype", "Flag dtype"], feature_rows), "`src/features.py::build_weight_features` returns exactly these three candidate columns, preserving the index and every row:\n\n- `weight_clean = abs(weight)`; original missing values remain `NaN`.\n- `weight_missing = 1` exactly when the original weight is missing.\n- `weight_negative = 1` exactly when the original weight is negative; missing values have flag 0.\n\nThe function does not read `posted_rate` or `load_id`, does not alter raw data, and learns no statistics. Malformed or infinite nonmissing weight values raise an explicit error.", "The current candidate retains NaN for later model-native missing handling. If a later model needs imputation, fit its median or other statistic only on the relevant training fold and reuse that fitted transform on the later holdout/inference rows. Never fill using a statistic from all labeled rows before fold splitting or from final inference. Indicator flags must retain original-value meaning after any later imputation.", "### Interpretation and later comparison", "The negative values are **consistent with sign corruption**, supported by plausible magnitudes, overlap with positive weights, and broadly similar labeled rate distributions. This remains a hypothesis, not a proven correction. Preserve negative rows and the negative flag. Later compare raw versus absolute weight using identical development-only chronological folds, preprocessing policies, and model configurations. Correlations and final input-quality distributions are not evidence that the cleaned representation improves predictive MAE, RMSE, or R2. No model has been trained or selected."])
    return lines


def distribution_shifts(train: pd.DataFrame, validation: pd.DataFrame) -> list[str]:
    lines = ["## Development versus final inference distributions", "This section describes supplied inputs only. Final inference is never used for model selection, preprocessing fitting, hyperparameter tuning, or validation metrics. Differences combine time, seasonal mix, and location mix; they do not establish their causes."]
    rows = []
    for column in FEATURE_COLUMNS:
        if column in NUMERIC_COLUMNS:
            first, second = finite_values(train[column]), finite_values(validation[column])
            sd = first.std()
            standardized_delta = (second.mean() - first.mean()) / sd if pd.notna(sd) and sd > 0 else None
            rows.append([column, first.mean(), second.mean(), first.std(), second.std(), first.median(), second.median(), standardized_delta, ks_distance(first, second)])
    lines.append(table(["Feature", "Train mean", "Final mean", "Train SD", "Final SD", "Train median", "Final median", "Mean delta / train SD", "KS D"], rows))
    lines.append("Summaries use finite values only, with missing/malformed/infinite counts separately reported for every column. Sample SD uses ddof=1. KS D is the maximum empirical CDF difference (0–1); no p-value or model-quality claim is made.")

    quality_rows = []
    for name, frame in [("Development", train), ("Final inference", validation)]:
        weight = numeric_values(frame["weight"])
        clean = finite_values(frame["weight"]).abs()
        quality_rows.append([name, int(frame["weight"].isna().sum()), 100 * frame["weight"].isna().mean(), int(weight.lt(0).sum()), 100 * weight.lt(0).mean(), int(weight.eq(0).sum()), clean.min(), clean.max(), int(frame["market_index"].isna().sum()), 100 * frame["market_index"].isna().mean()])
    lines.extend(["### Weight and market-index quality", table(["Dataset", "Missing weight", "Missing %", "Negative weight", "Negative %", "Zero weight", "Abs weight min", "Abs weight max", "Missing market", "Missing market %"], quality_rows), "Negative weights are invalid physical measurements, but the intended correction is not supplied. Preserve raw values and rows. Later compare raw weight against `abs(weight)` plus distinct missing/negative indicators in development-only chronological folds. Missing and negative indicators are disjoint; taking absolute values is a candidate representation, not a validated modeling decision."])

    monthly_rows = []
    for name, frame in [("Development", train), ("Final inference", validation)]:
        monthly = pd.DataFrame({"month": date_values(frame).dt.to_period("M"), "market": numeric_values(frame["market_index"]), "weight_missing": frame["weight"].isna(), "weight_negative": numeric_values(frame["weight"]).lt(0)})
        for month, group in monthly.groupby("month"):
            values = finite_values(group["market"])
            monthly_rows.append([name, str(month), len(group), len(values), values.mean(), int(group["market"].isna().sum()), int(group["weight_missing"].sum()), int(group["weight_negative"].sum())])
    lines.extend(["### Monthly counts and market index", table(["Dataset", "Month", "Rows", "Finite market", "Market mean", "Missing market", "Missing weight", "Negative weight"], monthly_rows), "Read pooled market-index shifts alongside these monthly values. The observations cover January–October and November–December respectively, so a pooled mean difference does not establish a permanent market change. November–December target behavior is unavailable."])

    category_rows = []
    details = []
    for column in ["pickup", "delivery", "equipment", "route"]:
        first = routes(train) if column == "route" else train[column]
        second = routes(validation) if column == "route" else validation[column]
        comparison, tv = category_comparison(first, second)
        unseen = ~second.isin(first.dropna())
        category_rows.append([column, first.nunique(), second.nunique(), second[unseen].nunique(), int(unseen.sum()), 100 * unseen.mean(), tv])
        if column != "route":
            names = sorted(second[unseen].dropna().unique())
            details.append(f"Unseen `{column}` categories: " + (", ".join(map(str, names)) if names else "none") + ".")
        largest = comparison.loc[comparison["delta_pp"].abs().sort_values(ascending=False, kind="stable").head(12).index]
        details.extend([f"### Largest category-frequency changes: {column}", table(["Category", "Train rows", "Final rows", "Train %", "Final %", "Delta percentage points"], [[category, *row.tolist()] for category, row in largest.iterrows()])])
    lines.extend(["### Categories and routes", table(["Feature", "Train distinct", "Final distinct", "Unseen distinct", "Unseen final rows", "Unseen rows %", "Total variation distance"], category_rows), "Total variation distance is half the absolute difference in category proportions across their union (0–1), including missing as an explicit category if present."])
    pickup_unseen = ~validation["pickup"].isin(train["pickup"].dropna())
    delivery_unseen = ~validation["delivery"].isin(train["delivery"].dropna())
    route_unseen = ~routes(validation).isin(routes(train))
    lines.append(table(["Coverage check", "Final rows", "Final rows %"], [["Any unseen city", int((pickup_unseen | delivery_unseen).sum()), 100 * (pickup_unseen | delivery_unseen).mean()], ["Both cities unseen", int((pickup_unseen & delivery_unseen).sum()), 100 * (pickup_unseen & delivery_unseen).mean()], ["Unseen route with both cities seen", int((route_unseen & ~pickup_unseen & ~delivery_unseen).sum()), 100 * (route_unseen & ~pickup_unseen & ~delivery_unseen).mean()]]))
    for name, frame in [("Development", train), ("Final inference", validation)]:
        frequencies = routes(frame).value_counts()
        lines.append(f"{name} route frequency: **{len(frequencies):,} distinct**, **{int(frequencies.eq(1).sum()):,} singleton routes**, **{int(frame['pickup'].eq(frame['delivery']).sum()):,} same-city rows**.")
        lines.append(table([f"{name}: top 10 routes", "Rows"], frequencies.head(10).items()))
    lines.extend(details)
    return lines


def schema_and_geography(frames: dict[str, pd.DataFrame]) -> list[str]:
    train, validation, template, december = (frames[name] for name in INPUTS)
    train_only = sorted(set(train) - set(validation))
    final_only = sorted(set(validation) - set(train))
    mismatches = [[column, str(train[column].dtype), str(validation[column].dtype)] for column in validation if column in train and train[column].dtype != validation[column].dtype]
    final_ids, template_ids = set(validation["load_id"]), set(template["load_id"])
    lines = ["## Schema and ID alignment", f"Development-only columns: **{train_only}**. Final-only columns: **{final_only}**. Removing `{TARGET}` leaves the same ordered schema: **{[c for c in train if c != TARGET] == list(validation)}**.", f"Shared-column dtype differences: **{len(mismatches)}**.", f"Template/final ID differences: **{len(final_ids-template_ids)} IDs missing from template**, **{len(template_ids-final_ids)} extra IDs in template**. IDs have identical row order: **{validation['load_id'].reset_index(drop=True).equals(template['load_id'].reset_index(drop=True))}**. Matching order is descriptive; prediction code must still join by identifier."]
    if mismatches:
        lines.append(table(["Column", "Train dtype", "Final dtype"], mismatches))
    locations = []
    for frame in [train, validation]:
        for role in ["pickup", "delivery"]:
            locations.append(frame[[role, f"{role}_lat", f"{role}_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    coordinates = pd.concat(locations, ignore_index=True)
    pair_counts = coordinates.drop_duplicates().groupby("city").size()
    invalid_bounds = ~numeric_values(coordinates["lat"]).between(-90, 90) | ~numeric_values(coordinates["lon"]).between(-180, 180)
    lines.extend(["## Geography and December availability", f"Cities across datasets and roles: **{len(pair_counts)}**. Cities with multiple supplied coordinate pairs: **{int(pair_counts.gt(1).sum())}**. Missing/out-of-bounds coordinate observations: **{int(invalid_bounds.sum())}**. This checks internal consistency and valid bounds, not external geographic accuracy."])
    december_missing = [column for column in FEATURE_COLUMNS if column not in december]
    lines.append(f"December lacks these development predictors: **{', '.join(december_missing)}**. `load_id` is also absent and is never a model feature.")
    rows = []
    for city in ["Lexington", "Fort Wayne"]:
        lookup = pd.concat([train[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1), train[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1)])
        matches = lookup[lookup["city"].eq(city)]
        unique = matches[["lat", "lon"]].drop_duplicates()
        rows.append([city, len(matches), len(unique), unique.iloc[0]["lat"] if len(unique) == 1 else None, unique.iloc[0]["lon"] if len(unique) == 1 else None])
    lines.append(table(["Training lookup city", "City-role observations", "Coordinate pairs", "Latitude", "Longitude"], rows))
    lines.append("Use deterministic supplied coordinates if required. Never invent December `market_index` or `quote_signal`. Later validate an explicit missing-feature strategy or a model restricted to legitimately available inputs. The supplied documentation does not establish quote-signal provenance or prediction-time availability; its presence or correlations alone do not prove target leakage.")
    return lines


def run_audit(project_root: Path, output: Path) -> bool:
    paths = {name: project_root / "data" / f"{name}.csv" for name in INPUTS}
    protected = [*paths.values(), project_root / "score.py", project_root / "Freight_Rate_ML_Assessment.pdf", project_root / "README.md", project_root / "requirements.txt", project_root / "docs/supplied_file_checksums.json"]
    if output.resolve() in {path.resolve() for path in protected}:
        raise ValueError("Audit output must not overwrite a supplied file or checksum manifest")
    before = {path: checksum(path) for path in protected}
    frames = {name: pd.read_csv(path, low_memory=False) for name, path in paths.items()}
    checks = validate_contracts(frames)
    integrity_rows = []
    manifest = json.loads((project_root / "docs/supplied_file_checksums.json").read_text())
    for relative, record in manifest["files"].items():
        match = checksum(project_root / relative) == record["sha256"]
        integrity_rows.append([relative, record["policy"], "MATCH" if match else "CHANGED"])
        if record["policy"] == "immutable":
            checks.append((f"Original bytes: {relative}", match, "SHA-256 matches supplied manifest"))

    train, validation = frames["train_test"], frames["validation"]
    sections = ["# Data audit", "Generated by `scripts/check_inputs.py` using pandas and NumPy. No model is fitted, no raw file is written, and no observations are removed. This is an input audit, not validation of a completed submission.", "Reproduce from the project root:\n\n```bash\n.venv/bin/python scripts/check_inputs.py\n```", "## Required input checks", table(["Check", "Result", "Details"], [[name, "PASS" if passed else "FAIL", detail] for name, passed, detail in checks]), "## Original-file integrity", table(["Supplied file", "Permitted policy", "Original SHA-256 comparison"], integrity_rows), "An editable December prediction column may change later; immutable inputs must retain their original bytes. The original blank output columns are audited as placeholders."]
    for name, frame in frames.items():
        sections.extend(describe_frame(name, frame))
    # Detailed comparisons require the expected schemas. Failed schemas still get
    # their full per-file diagnostics and a report instead of a misleading pass.
    schemas_ok = all(set(columns).issubset(frames[name].columns) for name, (_, columns) in INPUTS.items())
    if schemas_ok:
        sections.extend(target_distribution(train))
        sections.extend(schema_and_geography(frames))
        sections.extend(distribution_shifts(train, validation))
        sections.extend(weight_investigation(train, validation))
        if all(result for _, result, _ in checks):
            from scripts.plot_weight_quality import save_weight_quality_plot

            figure_path = project_root / "reports/figures/weight_quality.png"
            if figure_path.resolve() in {path.resolve() for path in protected}:
                raise ValueError("Weight figure must not overwrite a supplied file")
            save_weight_quality_plot(train, figure_path)
            figure_link = Path(os.path.relpath(figure_path, output.parent)).as_posix()
            sections.extend(["### Weight-quality visualization", f"![Development weight distributions and rate-per-mile relationship]({figure_link})", "The magnitude distributions are normalized separately. The rate-per-mile panel includes all finite eligible development observations on a logarithmic vertical scale so the full tail remains visible. The 300 original missing development weights are omitted only from these plots, remain in the dataset, and are counted in the figure caption."])
        else:
            sections.append("Weight plot was not generated because an input contract failed.")
    else:
        sections.append("Cross-file analyses were not attempted because required columns are missing; see schema failures above.")
    after = {path: checksum(path) for path in protected}
    unchanged = before == after
    sections.extend(["## Decisions and next steps", "Preserve all rows and raw files. Fit learned imputation/preprocessing inside development training folds; keep dates together and validate chronologically. Evaluate missing/negative-weight representations, unseen-category robustness, and realistic December features later. Do not use final inference distributions to choose or tune models. Do not remove high-rate observations solely because they are statistical outliers.", f"Protected files unchanged during this audit: **{unchanged}**. No model metrics or Spotter hidden metrics were calculated."])
    passed = all(result for _, result, _ in checks) and unchanged
    sections.append(f"Audit contract result: **{'PASS' if passed else 'FAIL'}**. A PASS confirms input contracts; the reported missing values, negative weights, unseen categories, and distribution differences still need explicit treatment.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"Audit {'PASS' if passed else 'FAIL'}: {len(checks)} contract checks; report: {output}")
    for name, frame in frames.items():
        print(f"{name}: {frame.shape}")
    if schemas_ok:
        for name, frame in [("Development", train), ("Final inference", validation)]:
            print(f"{name}: missing weight={frame['weight'].isna().sum()}, negative weight={numeric_values(frame['weight']).lt(0).sum()}, missing market_index={frame['market_index'].isna().sum()}")
        any_unseen = ~validation["pickup"].isin(train["pickup"]) | ~validation["delivery"].isin(train["delivery"])
        unseen_routes = ~routes(validation).isin(routes(train))
        print(f"Unseen-city rows={any_unseen.sum()}; unseen-route rows={unseen_routes.sum()}")
    print(f"Protected files unchanged: {unchanged}")
    for name, result, detail in checks:
        if not result:
            print(f"FAIL: {name}: {detail}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=Path("docs/data_audit.md"))
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else project_root / args.output
    return 0 if run_audit(project_root, output) else 1


if __name__ == "__main__":
    raise SystemExit(main())
