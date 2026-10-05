"""Reproducible descriptive EDA; no modeling, row deletion, or raw-data writes.

Run: .venv/bin/python scripts/eda.py
Outputs: reports/eda.md, reports/figures/eda_*.png, and EDA summary tables.
Final inference inputs are inspected only for data quality and distribution shifts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "freight-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "freight-xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_inputs import FEATURE_COLUMNS, INPUTS, category_comparison, checksum, ks_distance, routes, table, validate_contracts
from scripts.plot_weight_quality import save_weight_quality_plot
from src.features import build_weight_features

TRAIN_COLOR = "#16697A"
FINAL_COLOR = "#D67732"
SEED = 42


def finite(series: pd.Series) -> pd.Series:
    return series[np.isfinite(series)]


def image(name: str, caption: str) -> str:
    return f"![{caption}](figures/eda_{name}.png)\n\n{caption}"


def save_figure(figure: plt.Figure, directory: Path, name: str) -> None:
    figure.tight_layout()
    output = directory / f"eda_{name}.png"
    if output.is_symlink():
        raise ValueError(f"Refusing to overwrite a figure symlink: {output}")
    figure.savefig(output, dpi=160, bbox_inches="tight", metadata={"Software": "Freight assessment EDA"})
    plt.close(figure)


def pair_summary(first: pd.Series, second: pd.Series) -> tuple[int, float, float]:
    pairs = pd.DataFrame({"x": first, "y": second})
    pairs = pairs[np.isfinite(pairs).all(axis=1)]
    return len(pairs), float(pairs.x.corr(pairs.y)), float(pairs.x.rank().corr(pairs.y.rank()))


def scatter(axis: plt.Axes, x: pd.Series, y: pd.Series, xlabel: str, ylabel: str, log_y: bool = False) -> int:
    eligible = np.isfinite(x) & np.isfinite(y)
    if log_y:
        eligible &= y.gt(0)
    axis.scatter(x[eligible], y[eligible], s=3, alpha=0.13, color=TRAIN_COLOR, rasterized=True)
    if log_y:
        axis.set_yscale("log")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(f"All {int(eligible.sum()):,} eligible observations")
    return int(eligible.sum())


def overlaid_hist(axis: plt.Axes, first: pd.Series, second: pd.Series, name: str) -> None:
    first, second = finite(first), finite(second)
    low, high = min(first.min(), second.min()), max(first.max(), second.max())
    bins = np.linspace(low, high, 46) if high > low else np.linspace(low-.5, high+.5, 46)
    axis.hist(first, bins=bins, density=True, histtype="step", linewidth=1.8, color=TRAIN_COLOR, label=f"Development n={len(first):,}")
    axis.hist(second, bins=bins, density=True, histtype="step", linewidth=1.8, color=FINAL_COLOR, label=f"Final inputs n={len(second):,}")
    axis.set_xlabel(name)
    axis.set_ylabel("Density (each dataset normalized)")
    axis.legend(fontsize=7)


def group_summary(train: pd.DataFrame, column: str) -> pd.DataFrame:
    return train.groupby(column, dropna=False).agg(
        rows=("posted_rate", "size"), mean_rate=("posted_rate", "mean"),
        median_rate=("posted_rate", "median"), median_rpm=("rate_per_mile", "median"),
        median_distance=("distance", "median"),
    ).sort_values("rows", ascending=False, kind="stable")


def summary_rows(frame: pd.DataFrame) -> list[list[object]]:
    return [[index, *row.tolist()] for index, row in frame.iterrows()]


def target_and_time(train: pd.DataFrame, directory: Path, report_directory: Path) -> tuple[list[str], dict]:
    target, rpm = train.posted_rate, train.rate_per_mile
    target_stats = target.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
    rpm_stats = rpm.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].hist(target, bins=90, color=TRAIN_COLOR)
    axes[0].set(xlabel="Posted rate ($)", ylabel="Loads", title="Full observed target range")
    axes[1].hist(target, bins=np.geomspace(target.min(), np.nextafter(target.max(), np.inf), 70), color=TRAIN_COLOR)
    axes[1].set_xscale("log")
    axes[1].set(xlabel="Posted rate ($, log x scale)", ylabel="Loads", title="Same observations; log-spaced bins")
    save_figure(figure, directory, "target_distribution")

    monthly = train.groupby(train.date.dt.to_period("M")).agg(
        rows=("posted_rate", "size"), mean_rate=("posted_rate", "mean"),
        median_rate=("posted_rate", "median"), median_rpm=("rate_per_mile", "median"),
        median_distance=("distance", "median"), market_mean=("market_index", "mean"),
    )
    monthly.to_csv(report_directory / "eda_monthly_summary.csv")
    daily = train.groupby("date").posted_rate.agg(["mean", "median", "count"])
    figure, axes = plt.subplots(2, 1, figsize=(12, 7))
    axes[0].plot(daily.index, daily["median"], color=TRAIN_COLOR, alpha=.45, linewidth=.8, label="Daily median")
    axes[0].plot(daily.index, daily["median"].rolling(14, min_periods=1).mean(), color=TRAIN_COLOR, linewidth=2, label="Trailing 14-day mean of daily medians")
    axes[0].plot(daily.index, daily["mean"].rolling(14, min_periods=1).mean(), color=FINAL_COLOR, label="Trailing 14-day mean of daily means")
    axes[0].set(ylabel="Posted rate ($)", title="Development target over time; no November–December labels")
    axes[0].legend(fontsize=8)
    axes[1].plot(monthly.index.to_timestamp(), monthly.median_rpm, marker="o", color=TRAIN_COLOR)
    axes[1].set(ylabel="Monthly median rate per mile ($/mile)", xlabel="Development month")
    save_figure(figure, directory, "calendar")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    scatter(axes[0], train.distance, target, "Distance (miles)", "Posted rate ($, log scale)", log_y=True)
    scatter(axes[1], train.distance, rpm, "Distance (miles)", "Rate per mile ($/mile, log scale)", log_y=True)
    save_figure(figure, directory, "distance")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].hist(rpm, bins=90, color=TRAIN_COLOR)
    axes[0].set(xlabel="Rate per mile ($/mile)", ylabel="Loads", title="Full observed rate-per-mile range")
    axes[1].hist(rpm, bins=np.geomspace(rpm.min(), np.nextafter(rpm.max(), np.inf), 70), color=TRAIN_COLOR)
    axes[1].set_xscale("log")
    axes[1].set(xlabel="Rate per mile ($/mile, log x scale)", ylabel="Loads", title="Same observations; log-spaced bins")
    save_figure(figure, directory, "rate_per_mile")
    distance_correlation = pair_summary(train.distance, target)
    distance_rpm_correlation = pair_summary(train.distance, rpm)
    lines = ["## 1–2. posted_rate distribution and summary statistics", table(["Statistic", "Posted rate ($)", "Rate per mile ($/mile)"], [[name, value, rpm_stats[name]] for name, value in target_stats.items()]), image("target_distribution", "Both panels retain all labeled targets, including the complete high-rate tail."), "## 3. posted_rate over calendar time", table(["Month", "Rows", "Mean rate", "Median rate", "Median $/mile", "Median distance", "Mean market index"], summary_rows(monthly)), image("calendar", "Monthly and trailing summaries are descriptive only; trailing windows are never used here as model features."), "Changes can reflect freight mix, equipment, route, market, and date effects. Ten observed months cannot establish recurring annual seasonality or November–December target behavior.", "## 4. Rate versus distance", table(["Association", "Complete pairs", "Pearson", "Spearman"], [["Distance / posted_rate", *distance_correlation], ["Distance / rate_per_mile", *distance_rpm_correlation]]), image("distance", "Logarithmic vertical axes show all positive observations, including outliers; no axes were trimmed to remove them."), "Distance is strongly associated with total rate. Rate per mile also changes with distance, so dividing by miles does not fully control freight mix or establish a constant per-mile price.", "## 5. Rate-per-mile distribution", image("rate_per_mile", "Rate per mile is posted_rate / distance, calculated for EDA only. It depends on the target and must never become an inference feature.")]
    return lines, {"target": target_stats.to_dict(), "rate_per_mile": rpm_stats.to_dict(), "distance_target_pearson": distance_correlation[1], "distance_target_spearman": distance_correlation[2], "distance_rpm_spearman": distance_rpm_correlation[2], "monthly": {str(key): row.to_dict() for key, row in monthly.iterrows()}}


def weight_and_equipment(train: pd.DataFrame, directory: Path, report_directory: Path) -> tuple[list[str], dict]:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    scatter(axes[0], train.weight, train.posted_rate, "Original signed weight (lb)", "Posted rate ($, log scale)", log_y=True)
    scatter(axes[1], train.weight_clean, train.posted_rate, "Absolute candidate weight (lb)", "Posted rate ($, log scale)", log_y=True)
    save_figure(figure, directory, "weight_relationship")
    save_weight_quality_plot(train, directory / "eda_weight_quality.png")
    equipment = group_summary(train, "equipment")
    equipment.to_csv(report_directory / "eda_equipment_summary.csv")
    labels = equipment.index.tolist()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, column, ylabel in [(axes[0], "posted_rate", "Posted rate ($, log scale)"), (axes[1], "rate_per_mile", "Rate per mile ($/mile, log scale)")]:
        axis.boxplot([train.loc[train.equipment.eq(label), column] for label in labels], positions=np.arange(len(labels)), showfliers=True, flierprops={"markersize": 2, "alpha": .2})
        axis.set_xticks(np.arange(len(labels)), labels)
        axis.set_yscale("log")
        axis.set_ylabel(ylabel)
        axis.set_title("Full distributions, outliers retained")
    save_figure(figure, directory, "equipment")
    correlations = [[name, *pair_summary(train[name], train.posted_rate)] for name in ["weight", "weight_clean"]]
    lines = ["## 6. Rate versus weight", table(["Weight representation / posted_rate", "Complete pairs", "Pearson", "Spearman"], correlations), image("weight_relationship", f"Only {int(train.weight.isna().sum())} missing weights are unavailable for these plots; their rows and targets remain in the dataset."), "Absolute weight and original missing/negative flags are candidates, not selected features. The weak marginal association with total price does not imply that weight is irrelevant in interactions or within freight subgroups.", "## 7. Equipment differences", table(["Equipment", "Rows", "Mean rate", "Median rate", "Median $/mile", "Median distance"], summary_rows(equipment)), image("equipment", "Outliers are shown for every equipment group. Equipment comparisons remain observational, with route and distance mix as potential confounders.")]
    return lines, {"equipment": {str(key): row.to_dict() for key, row in equipment.iterrows()}, "weight_target_pearson": correlations[0][2], "clean_weight_target_pearson": correlations[1][2]}


def markets_and_routes(train: pd.DataFrame, validation: pd.DataFrame, directory: Path, report_directory: Path) -> tuple[list[str], dict]:
    lines, summary = [], {}
    for number, role in [(8, "pickup"), (9, "delivery")]:
        groups = group_summary(train, role)
        groups.to_csv(report_directory / f"eda_{role}_markets.csv")
        top_count = groups.head(12).iloc[::-1]
        top_rpm = groups[groups.rows.ge(100)].sort_values("median_rpm", ascending=False, kind="stable").head(12).iloc[::-1]
        figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        axes[0].barh(top_count.index, top_count.rows, color=TRAIN_COLOR)
        axes[0].set(xlabel="Development loads", title=f"Most frequent {role} markets")
        axes[1].barh(top_rpm.index, top_rpm.median_rpm, color=FINAL_COLOR)
        axes[1].set(xlabel="Median observed $/mile", title="Highest median $/mile; at least 100 loads")
        save_figure(figure, directory, f"{role}_markets")
        lines.extend([f"## {number}. {role.title()} markets", f"Development has **{len(groups)}** {role} categories. Full summaries for every market are saved in `eda_{role}_markets.csv`.", table([role.title(), "Rows", "Mean rate", "Median rate", "Median $/mile", "Median distance"], summary_rows(groups.head(12))), image(f"{role}_markets", "These are unadjusted summaries of observed lane and equipment mixes. The 100-load cutoff is a fixed display rule, not a training-data filter or model-selection rule.")])
        summary[f"{role}_categories"] = len(groups)
    counts = routes(train).value_counts()
    final_counts = routes(validation).value_counts()
    counts.rename("loads").to_csv(report_directory / "eda_route_counts.csv", index_label="route")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].hist(counts, bins=np.arange(.5, counts.max()+1.5), color=TRAIN_COLOR)
    axes[0].set(xlabel="Development loads per route", ylabel="Distinct routes", title="All observed development routes")
    top = counts.head(12).iloc[::-1]
    axes[1].barh(top.index, top, color=TRAIN_COLOR)
    axes[1].set(xlabel="Development loads", title="Most frequent development routes")
    save_figure(figure, directory, "routes")
    unseen = ~routes(validation).isin(counts.index)
    lines.extend(["## 10. Route frequency", table(["Statistic", "Development", "Final inputs"], [["Distinct observed routes", len(counts), len(final_counts)], ["Routes observed once", int(counts.eq(1).sum()), int(final_counts.eq(1).sum())], ["Median observed loads per route", counts.median(), final_counts.median()], ["Maximum observed loads per route", counts.max(), final_counts.max()]]), table(["Top development route", "Loads"], counts.head(12).items()), image("routes", "Route categories are relatively sparse even with 48,000 labeled observations."), f"Final inputs contain **{routes(validation)[unseen].nunique()}** unseen routes across **{int(unseen.sum()):,} rows**. Different sample sizes affect frequency summaries; unseen routes must not cause inference failures and route identity should not be the sole geographic signal."])
    summary.update({"train_routes": len(counts), "final_routes": len(final_counts), "unseen_route_rows": int(unseen.sum()), "unseen_routes": int(routes(validation)[unseen].nunique()), "train_singleton_routes": int(counts.eq(1).sum())})
    return lines, summary


def signals_and_correlations(train: pd.DataFrame, validation: pd.DataFrame, directory: Path, report_directory: Path) -> tuple[list[str], dict]:
    lines, summary = [], {}
    for number, feature in [(11, "market_index"), (12, "quote_signal")]:
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        overlaid_hist(axes[0], train[feature], validation[feature], feature)
        scatter(axes[1], train[feature], train.posted_rate, feature, "Posted rate ($, log scale)", log_y=True)
        scatter(axes[2], train[feature], train.rate_per_mile, feature, "Rate per mile ($/mile, log scale)", log_y=True)
        save_figure(figure, directory, feature)
        rows = [[outcome, *pair_summary(train[feature], train[outcome])] for outcome in ["posted_rate", "rate_per_mile"]]
        lines.extend([f"## {number}. {feature}", table(["Development outcome", "Complete pairs", "Pearson", "Spearman"], rows), image(feature, "Outcome relationships use development labels only. The overlaid feature densities describe final inputs without fitting or selecting a model.")])
        summary[f"{feature}_target_pearson"] = rows[0][2]
        summary[f"{feature}_target_spearman"] = rows[0][3]
        summary[f"{feature}_rpm_spearman"] = rows[1][3]
        if feature == "market_index":
            monthly = pd.concat([train.assign(dataset="Development"), validation.assign(dataset="Final inputs")]).groupby(["dataset", pd.Grouper(key="date", freq="MS")]).market_index.agg(["count", "mean", "median"])
            lines.extend([table(["Dataset", "Month", "Nonmissing", "Mean index", "Median index"], [[dataset, f"{month:%Y-%m}", *row.tolist()] for (dataset, month), row in monthly.iterrows()]), "Market-index differences mix calendar periods and load composition. Monthly means contextualize the pooled shift; ten labeled months do not prove a repeatable full-year seasonal pattern."])
        else:
            lines.append("Quote-signal marginal correlations are weak in this dataset. They do not establish whether interactions are useful. Its provenance, units, and availability before a rate is posted are undocumented; presence in supplied inference data does not resolve that uncertainty. Any future use needs development-only chronological ablation and a realistic December strategy.")
    columns = ["distance", "weight", "weight_clean", "weight_missing", "weight_negative", "market_index", "quote_signal", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "posted_rate", "rate_per_mile"]
    numeric = train[columns]
    correlations = {method: numeric.corr(method=method) for method in ["pearson", "spearman"]}
    valid = numeric.notna().astype("int64")
    pair_counts = valid.T @ valid
    pair_counts.to_csv(report_directory / "eda_correlation_pair_counts.csv")
    figure, axes = plt.subplots(1, 2, figsize=(21, 9))
    for axis, (method, values) in zip(axes, correlations.items()):
        values.to_csv(report_directory / f"eda_correlations_{method}.csv")
        heatmap = axis.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1)
        axis.set_xticks(np.arange(len(columns)), columns, rotation=70, ha="right", fontsize=7)
        axis.set_yticks(np.arange(len(columns)), columns, fontsize=8)
        axis.set_title(f"Development {method.title()} correlations")
        for row in range(len(columns)):
            for column in range(len(columns)):
                value = values.iloc[row, column]
                text = f"{value:.2f}" if pd.notna(value) else "—"
                axis.text(column, row, text, ha="center", va="center", fontsize=6, color="white" if pd.notna(value) and abs(value)>.6 else "black")
    figure.subplots_adjust(left=.08, right=.88, bottom=.25, wspace=.38)
    color_axis = figure.add_axes([.92, .3, .012, .52])
    figure.colorbar(heatmap, cax=color_axis, label="Correlation")
    output = directory / "eda_numeric_correlations.png"
    if output.is_symlink():
        raise ValueError("Refusing to overwrite a correlation-figure symlink")
    figure.savefig(output, dpi=160, bbox_inches="tight", metadata={"Software": "Freight assessment EDA"})
    plt.close(figure)
    rows = [[column, int(pair_counts.loc[column, "posted_rate"]), correlations["pearson"].loc[column, "posted_rate"], correlations["spearman"].loc[column, "posted_rate"], correlations["spearman"].loc[column, "rate_per_mile"]] for column in columns if column not in ["posted_rate", "rate_per_mile"]]
    lines.extend(["## 13. Numeric correlations", table(["Variable", "Target pair count", "Pearson / target", "Spearman / target", "Spearman / $ per mile"], rows), image("numeric_correlations", "Full Pearson and Spearman matrices and complete-pair counts are exported as CSVs. Correlations use pairwise available development observations."), "A dash denotes undefined correlation when the available pairs leave a variable constant; for example, the missing-weight flag is always zero among observed weights. This is not a zero-correlation claim.", "Identifiers are excluded from this analysis. `posted_rate` and its derived `rate_per_mile` appear only as analysis outcomes, never model features. Correlation is not causation or evidence of out-of-sample predictive improvement."])
    return lines, summary


def quality_and_geography(train: pd.DataFrame, validation: pd.DataFrame, directory: Path) -> tuple[list[str], dict]:
    missing = []
    for column in FEATURE_COLUMNS:
        missing.append([column, int(train[column].isna().sum()), 100*train[column].isna().mean(), int(validation[column].isna().sum()), 100*validation[column].isna().mean()])
    plotted = [row for row in missing if row[1] or row[3]]
    figure, axis = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(plotted))
    axis.bar(positions-.18, [row[2] for row in plotted], width=.36, color=TRAIN_COLOR, label="Development")
    axis.bar(positions+.18, [row[4] for row in plotted], width=.36, color=FINAL_COLOR, label="Final inputs")
    axis.set_xticks(positions, [row[0] for row in plotted])
    axis.set(ylabel="Missing predictor observations (%)", title="All predictor columns with any missing values")
    axis.legend()
    save_figure(figure, directory, "missingness")
    any_city = ~validation.pickup.isin(train.pickup) | ~validation.delivery.isin(train.delivery)
    weight_rows = [[name, len(frame), int(frame.weight_missing.sum()), int(frame.weight_negative.sum()), 100*frame.weight_negative.mean(), frame.weight_clean.min(), frame.weight_clean.max()] for name, frame in [("Development", train), ("Final inputs", validation)]]
    lines = ["## 14. Missing values", table(["Shared predictor", "Train missing", "Train %", "Final missing", "Final %"], missing), image("missingness", "Both datasets retain all rows. Missing output placeholders in the template and December file are intentional and are not counted as missing predictors."), "No imputation is fitted during EDA. Missing cleaned weight stays NaN. Later learned preprocessing must be fitted only within each development training fold.", "## 15. Negative weights", table(["Dataset", "Rows", "Missing weight", "Negative weight", "Negative %", "Abs min lb", "Abs max lb"], weight_rows), image("weight_quality", "Negative magnitudes resemble positive weights, supporting sign corruption as a hypothesis. The full development rate-per-mile tail is shown."), "Negative rows are preserved. Absolute weight plus original-value indicators remains a candidate representation until compared with raw weight in identical chronological folds. See `../docs/data_audit.md` for the detailed weight investigation."]
    geography_rows = []
    market_frames = {}
    for role in ["pickup", "delivery"]:
        market = group_summary(train, role).join(train.groupby(role)[[f"{role}_lat", f"{role}_lon"]].first())
        market_frames[role] = market
        for column in [f"{role}_lat", f"{role}_lon"]:
            geography_rows.append([column, train[column].min(), train[column].max(), validation[column].min(), validation[column].max(), train[column].isna().sum(), validation[column].isna().sum()])
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    color_low = min(frame.median_rpm.min() for frame in market_frames.values())
    color_high = max(frame.median_rpm.max() for frame in market_frames.values())
    for axis, role in zip(axes, ["pickup", "delivery"]):
        market = market_frames[role]
        points = axis.scatter(market[f"{role}_lon"], market[f"{role}_lat"], s=market.rows/10, c=market.median_rpm, cmap="viridis", vmin=color_low, vmax=color_high, alpha=.85)
        new_names = sorted(set(validation[role])-set(train[role]))
        new = validation.loc[validation[role].isin(new_names)].groupby(role)[[f"{role}_lat", f"{role}_lon"]].first()
        axis.scatter(new[f"{role}_lon"], new[f"{role}_lat"], marker="x", color=FINAL_COLOR, s=65, label="Unseen final-input city")
        for city, row in market.head(4).iterrows():
            axis.annotate(city, (row[f"{role}_lon"], row[f"{role}_lat"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
        axis.set(xlabel="Supplied longitude", ylabel="Supplied latitude", title=f"{role.title()}: supplied coordinates; bubble size = loads")
        axis.legend(fontsize=7)
    figure.subplots_adjust(right=.87, wspace=.25)
    color_axis = figure.add_axes([.9, .2, .018, .6])
    figure.colorbar(points, cax=color_axis, label="Development city median $/mile")
    geography_output = directory / "eda_geography.png"
    if geography_output.is_symlink():
        raise ValueError("Refusing to overwrite a geography-figure symlink")
    figure.savefig(geography_output, dpi=160, bbox_inches="tight", metadata={"Software": "Freight assessment EDA"})
    plt.close(figure)
    lines.extend(["## 16. Geographic variables", table(["Coordinate", "Train min", "Train max", "Final min", "Final max", "Train missing", "Final missing"], geography_rows), image("geography", "Coordinates are the supplied lookup values, not externally verified geocodes. City colors are unadjusted development summaries, not learned target encodings."), f"There are **{int(any_city.sum()):,} final-input rows ({100*any_city.mean():.2f}%)** with an unseen city. Coordinates offer numeric location context when city or route labels are new; location names alone need unknown-category handling. Coordinate/rate associations may largely reflect route length and lane composition."])
    return lines, {"unseen_city_rows": int(any_city.sum()), "missing_weight_train": int(train.weight_missing.sum()), "missing_weight_final": int(validation.weight_missing.sum()), "negative_weight_train": int(train.weight_negative.sum()), "negative_weight_final": int(validation.weight_negative.sum()), "missing_market_train": int(train.market_index.isna().sum()), "missing_market_final": int(validation.market_index.isna().sum())}


def outliers_and_shifts(train: pd.DataFrame, validation: pd.DataFrame, directory: Path, report_directory: Path) -> tuple[list[str], dict]:
    target = train.posted_rate
    q1, q3 = target.quantile([.25, .75])
    lower, upper = q1-1.5*(q3-q1), q3+1.5*(q3-q1)
    flagged = target.lt(lower) | target.gt(upper)
    figure, axis = plt.subplots(figsize=(10, 5))
    for mask, label, color in [(~flagged, "Within IQR fences", TRAIN_COLOR), (flagged, "Statistical flag; retained", FINAL_COLOR)]:
        axis.scatter(train.loc[mask, "distance"], target[mask], s=4 if color==TRAIN_COLOR else 10, alpha=.18 if color==TRAIN_COLOR else .6, color=color, label=f"{label}: n={int(mask.sum()):,}", rasterized=True)
    axis.axhline(upper, linestyle="--", color=FINAL_COLOR, label=f"Upper IQR fence ${upper:,.2f}")
    axis.set_yscale("log")
    axis.set(xlabel="Distance (miles)", ylabel="Posted rate ($, log scale)", title="All targets; statistical flags do not establish invalidity")
    axis.legend(fontsize=8)
    save_figure(figure, directory, "target_outliers")
    examples = train.nlargest(10, "posted_rate")[["load_id", "date", "pickup", "delivery", "distance", "equipment", "weight", "posted_rate", "rate_per_mile"]]
    exported = examples.copy()
    exported["date"] = exported.date.dt.strftime("%Y-%m-%d")
    exported.to_csv(report_directory / "eda_high_rate_examples.csv", index=False)
    lines = ["## 17. Potential target outliers", table(["Rule", "Threshold", "Flagged rows"], [["Below lower 1.5×IQR fence", lower, int(target.lt(lower).sum())], ["Above upper 1.5×IQR fence", upper, int(target.gt(upper).sum())], ["Above 99th percentile", target.quantile(.99), int(target.gt(target.quantile(.99)).sum())], ["Rate above $10,000", 10_000, int(target.gt(10_000).sum())]]), image("target_outliers", "No target is clipped, winsorized, corrected, or removed. Largest-rate examples remain in the dataset."), table(["load_id (locator only)", "Date", "Pickup", "Delivery", "Miles", "Equipment", "Raw weight", "Posted rate", "$ per mile"], exported.to_numpy().tolist()), "Identifiers above locate observations for review and are never predictive features. A large price is not sufficient evidence of invalidity; pricing provenance is unavailable. Evaluate direct versus log-target models later on identical chronological folds, and record both MAE and RMSE to understand sensitivity to tails."]
    features = ["distance", "weight_clean", "market_index", "quote_signal", "pickup_lat", "delivery_lat"]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    shift_rows = []
    for axis, feature in zip(axes.ravel(), features):
        first, second = finite(train[feature]), finite(validation[feature])
        overlaid_hist(axis, first, second, feature)
        shift_rows.append([feature, len(first), len(second), first.mean(), second.mean(), first.std(), second.std(), ks_distance(first, second)])
    save_figure(figure, directory, "distribution_shifts")
    categories = []
    for column in ["pickup", "delivery", "equipment", "route"]:
        first = routes(train) if column == "route" else train[column]
        second = routes(validation) if column == "route" else validation[column]
        _, tv = category_comparison(first, second)
        unseen = ~second.isin(first)
        categories.append([column, first.nunique(), second.nunique(), second[unseen].nunique(), int(unseen.sum()), tv])
    lines.extend(["## 18. Development versus final-input distribution shifts", table(["Feature", "Train finite", "Final finite", "Train mean", "Final mean", "Train SD", "Final SD", "Empirical KS D"], shift_rows), table(["Category", "Train distinct", "Final distinct", "Unseen distinct", "Unseen final rows", "Total variation distance"], categories), image("distribution_shifts", "Both densities are normalized over each dataset and use shared full-range bins; missing counts are reported separately."), "KS D and total variation are descriptive distribution distances, not model-quality metrics. Differences include seasonal mix, new locations, sample size, and sampling variability. Do not use final-input distributions to select features, tune hyperparameters, fit preprocessing, or claim prediction accuracy. The actual task extrapolates from January–October labels into November–December; use chronological development validation.", "The fixed December chart scenario omits market and quote signals. Coordinates can use deterministic supplied city lookups, but genuinely unavailable future values must not be fabricated. Feature availability must be considered alongside later development-only validation."])
    return lines, {"target_lower_iqr_fence": float(lower), "target_upper_iqr_fence": float(upper), "target_iqr_flags": int(flagged.sum()), "target_over_10000": int(target.gt(10_000).sum()), "market_mean_train": float(train.market_index.mean()), "market_mean_final": float(validation.market_index.mean())}


def modeling_findings(summary: dict) -> list[str]:
    equipment = summary["equipment"]
    return [
        f"Distance is the strongest observed marginal driver of total rate: Pearson {summary['distance_target_pearson']:.3f}, Spearman {summary['distance_target_spearman']:.3f}. A distance baseline is essential; these are correlations, not model metrics.",
        f"Rates are right-tailed: median ${summary['target']['50%']:,.2f}, mean ${summary['target']['mean']:,.2f}, maximum ${summary['target']['max']:,.2f}. Retain the {summary['target_over_10000']} rates above $10,000 and investigate provenance; compare direct/log targets and examine MAE/RMSE later.",
        f"Equipment changes the observed rate-per-mile distribution: median Dry Van ${equipment['Dry Van']['median_rpm']:.2f}, Flatbed ${equipment['Flatbed']['median_rpm']:.2f}, Reefer ${equipment['Reefer']['median_rpm']:.2f}. Lane and distance confounding prevents a causal premium claim.",
        f"Calendar and freight mix matter: monthly median $/mile moves from {summary['monthly']['2025-01']['median_rpm']:.3f} in January to {summary['monthly']['2025-06']['median_rpm']:.3f} in June. No November–December labels exist, so validation must respect chronology and cannot establish full-year seasonality.",
        f"Handle quality problems explicitly: development/final inputs have {summary['missing_weight_train']}/{summary['missing_weight_final']} missing weights, {summary['negative_weight_train']}/{summary['negative_weight_final']} negative weights, and {summary['missing_market_train']}/{summary['missing_market_final']} missing market indices. Preserve rows; compare raw weight against absolute weight plus flags using training-fold preprocessing only.",
        f"Unseen categories are a real inference requirement: {summary['unseen_city_rows']:,} final rows contain new cities and {summary['unseen_route_rows']:,} contain new routes. Keep numeric geography as a candidate and test unknown-category handling. Route-only categories are brittle.",
        f"Market index shifts from a pooled mean of {summary['market_mean_train']:.3f} to {summary['market_mean_final']:.3f}, but monthly values contextualize the difference. Short-haul/long-haul and calendar mix also affect rate per mile; descriptive shifts must not become final-data tuning.",
        f"Quote signal has weak marginal associations (target Pearson {summary['quote_signal_target_pearson']:.3f}, $/mile Spearman {summary['quote_signal_rpm_spearman']:.3f}). Do not assume usefulness or leakage without provenance and chronological ablation. December omits both quote and market signals; use legitimately available features or explicit missing handling.",
    ]


def run_eda(project_root: Path) -> None:
    manifest = json.loads((project_root / "docs/supplied_file_checksums.json").read_text())
    protected = [project_root / name for name in manifest["files"]]
    before = {path: checksum(path) for path in protected}
    for name, record in manifest["files"].items():
        if record["policy"] == "immutable" and before[project_root / name] != record["sha256"]:
            raise ValueError(f"Immutable supplied file changed: {name}")
    frames = {name: pd.read_csv(project_root / "data" / f"{name}.csv", low_memory=False) for name in INPUTS}
    failed = [(name, detail) for name, passed, detail in validate_contracts(frames) if not passed]
    if failed:
        raise ValueError(f"Input contracts failed; run scripts/check_inputs.py for diagnostics: {failed}")
    train, validation = frames["train_test"].copy(), frames["validation"].copy()
    original_rows = (len(train), len(validation))
    for frame in [train, validation]:
        frame["date"] = pd.to_datetime(frame.date, format="%Y-%m-%d", errors="raise")
        if frame.distance.isna().any() or not frame.distance.gt(0).all():
            raise ValueError("Distance must be present and positive for rate-per-mile EDA")
        features = build_weight_features(frame)
        for column in features:
            frame[column] = features[column]
    train["rate_per_mile"] = train.posted_rate / train.distance
    report_directory, directory = project_root / "reports", project_root / "reports/figures"
    if report_directory.is_symlink() or directory.is_symlink():
        raise ValueError("Refusing to write EDA outputs through a symlinked report directory")
    directory.mkdir(parents=True, exist_ok=True)
    if any(path.is_symlink() for path in directory.glob("eda_*.png")):
        raise ValueError("Refusing to overwrite symlinked EDA figures")
    output_names = ["eda.md", "eda_summary.json", "eda_monthly_summary.csv", "eda_equipment_summary.csv", "eda_pickup_markets.csv", "eda_delivery_markets.csv", "eda_route_counts.csv", "eda_correlation_pair_counts.csv", "eda_correlations_pearson.csv", "eda_correlations_spearman.csv", "eda_high_rate_examples.csv"]
    if any((report_directory / name).is_symlink() for name in output_names):
        raise ValueError("Refusing to overwrite symlinked EDA tables or report")
    summary = {"seed": SEED, "training_rows": len(train), "final_input_rows": len(validation), "training_date_range": [f"{train.date.min():%Y-%m-%d}", f"{train.date.max():%Y-%m-%d}"], "final_date_range": [f"{validation.date.min():%Y-%m-%d}", f"{validation.date.max():%Y-%m-%d}"], "model_training_performed": False}
    sections = ["# Exploratory data analysis", "Generated reproducibly by `scripts/eda.py`. No model is fitted, no rows are removed, no targets are clipped, and no raw files are modified. Plots use all eligible observations; missing values are omitted only where a variable cannot be plotted and counts are explicitly reported. No random sampling or jitter is used (reserved random seed: 42).", "Reproduce from the project root:\n\n```bash\n.venv/bin/python scripts/eda.py\n```", table(["Dataset", "Rows", "Date range", "Role"], [["Development", len(train), f"{train.date.min():%Y-%m-%d} through {train.date.max():%Y-%m-%d}", "Labeled EDA only"], ["Final inputs", len(validation), f"{validation.date.min():%Y-%m-%d} through {validation.date.max():%Y-%m-%d}", "Quality and descriptive shifts only"]]), "Target: **posted_rate**. `load_id` is an identifier. Final inputs contain no labels and are never used here for model selection, training, preprocessing fitting, or validation metrics. A complete input audit is in `../docs/data_audit.md`."]
    with plt.rc_context({"font.size": 9, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"}):
        analyses = [
            ("target, calendar, distance, rate per mile", lambda: target_and_time(train, directory, report_directory)),
            ("weight and equipment", lambda: weight_and_equipment(train, directory, report_directory)),
            ("markets and routes", lambda: markets_and_routes(train, validation, directory, report_directory)),
            ("market, quote, correlations", lambda: signals_and_correlations(train, validation, directory, report_directory)),
            ("missing data, negative weights, geography", lambda: quality_and_geography(train, validation, directory)),
            ("outliers and distribution shifts", lambda: outliers_and_shifts(train, validation, directory, report_directory)),
        ]
        for label, analysis in analyses:
            lines, result = analysis()
            sections.extend(lines)
            summary.update(result)
            print(f"Completed EDA: {label}", flush=True)
    assert original_rows == (len(train), len(validation))
    after = {path: checksum(path) for path in protected}
    if before != after:
        raise RuntimeError("A supplied file changed during EDA")
    findings = modeling_findings(summary)
    sections.extend(["## Eight findings for modeling and the Loom", "\n".join(f"{number}. {finding}" for number, finding in enumerate(findings, 1)), "## Reproducibility and limitations", "All supplied-file hashes were identical before and after this run. All immutable inputs matched the original checksum manifest. All 48,000 development and 12,000 final-input rows are retained. The template and fixed December inputs are unchanged. This EDA does not calculate MAE, RMSE, R2, or hidden Spotter metrics; no model has been selected or trained.", "Scatter plots show complete finite pairs and logarithmic price axes preserve the full positive tail. Histograms use fixed bin counts over the full observed range. Market ranking cutoffs affect display only. All outcome associations are observational and may reflect multiple correlated influences. Learned feature encoding, imputation, model selection, and tuning belong exclusively inside later chronological development experiments."])
    summary["findings"] = findings
    summary["figure_files"] = sorted(path.name for path in directory.glob("eda_*.png"))
    summary["supplied_files_unchanged"] = True
    summary["retained_rows"] = {"development": len(train), "final_inputs": len(validation)}
    (report_directory / "eda.md").write_text("\n\n".join(sections)+"\n", encoding="utf-8")
    (report_directory / "eda_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"EDA complete: {len(summary['figure_files'])} figures; report: {report_directory/'eda.md'}")
    for number, finding in enumerate(findings, 1):
        print(f"{number}. {finding}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    run_eda(args.project_root.resolve())


if __name__ == "__main__":
    main()
