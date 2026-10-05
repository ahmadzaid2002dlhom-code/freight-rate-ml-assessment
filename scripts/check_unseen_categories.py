"""Reproducible category coverage audit and prediction compatibility checks.

Small fixed diagnostic regressors are fitted on development rows only. They are
not candidate comparisons or production models. No accuracy metrics or saved
predictions are produced for the unlabeled final-validation dataset.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import CATEGORICAL_FEATURE_COLUMNS, FreightFeatureTransformer, build_features
from src.preprocessing import build_sklearn_preprocessor

COORDINATES = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]


def coverage(train: pd.DataFrame, validation: pd.DataFrame) -> dict:
    """Identify global new cities, role-specific unknowns, and directed routes."""
    development = build_features(train)
    final = build_features(validation)
    train_cities = set(development.pickup) | set(development.delivery)
    final_cities = set(final.pickup) | set(final.delivery)
    new_cities = final_cities - train_cities
    pickup_unknown = ~final.pickup.isin(development.pickup)
    delivery_unknown = ~final.delivery.isin(development.delivery)
    new_city_rows = final.pickup.isin(new_cities) | final.delivery.isin(new_cities)
    unseen_route = ~final.route.isin(development.route)
    return {
        "development": development, "final": final,
        "train_cities": sorted(train_cities), "final_cities": sorted(final_cities),
        "new_cities": sorted(new_cities),
        "masks": {
            "Unseen pickup label": pickup_unknown,
            "Unseen delivery label": delivery_unknown,
            "Any role-specific unseen city label": pickup_unknown | delivery_unknown,
            "Completely new city at either endpoint": new_city_rows,
            "Both endpoints completely new": final.pickup.isin(new_cities) & final.delivery.isin(new_cities),
            "Unseen directed route": unseen_route,
            "Unseen route with both endpoint labels previously seen in their roles":
                unseen_route & ~pickup_unknown & ~delivery_unknown,
        },
    }


def inference_checks(train: pd.DataFrame, validation: pd.DataFrame) -> dict:
    """Check real predictions using fixed, development-only smoke-test fits.

    The first seven complete development dates keep this a quick check. No
    final labels, eval_set, parameter search, early stopping, or scores are used.
    These fits are never serialized or used for submission outputs.
    """
    dates = pd.to_datetime(train.date, format="ISO8601", errors="raise")
    first_dates = dates.drop_duplicates().sort_values().iloc[:7]
    fit_rows = train.loc[dates.isin(first_dates)]
    X_train = build_features(fit_rows)
    X_final = build_features(validation)
    y_train = fit_rows.posted_rate
    sklearn_model = Pipeline([
        ("features", FreightFeatureTransformer()),
        ("preprocessing", build_sklearn_preprocessor()),
        ("regressor", Ridge(alpha=1.0, random_state=42)),
    ])
    sklearn_model.fit(fit_rows, y_train)
    sklearn_predictions = sklearn_model.predict(validation)
    catboost_model = CatBoostRegressor(
        iterations=12, depth=3, learning_rate=0.1, loss_function="RMSE",
        random_seed=42, thread_count=2, verbose=False, allow_writing_files=False,
    )
    catboost_model.fit(X_train, y_train, cat_features=CATEGORICAL_FEATURE_COLUMNS)
    catboost_predictions = catboost_model.predict(X_final)
    for name, values in [("sklearn", sklearn_predictions), ("catboost", catboost_predictions)]:
        if values.shape != (len(validation),) or not np.isfinite(values).all():
            raise AssertionError(f"{name} failed to predict every input row finitely")
    names = sklearn_model.named_steps["preprocessing"].get_feature_names_out().tolist()
    if not all(f"numeric__{column}" in names for column in COORDINATES):
        raise AssertionError("Sklearn preprocessing discarded geographic coordinates")
    return {
        "training_rows": len(fit_rows),
        "training_date_min": str(fit_rows.date.min()),
        "training_date_max": str(fit_rows.date.max()),
        "inference_rows": len(validation),
        "sklearn_finite_predictions": int(np.isfinite(sklearn_predictions).sum()),
        "catboost_finite_predictions": int(np.isfinite(catboost_predictions).sum()),
        "coordinates_retained": True,
    }


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
    ])


def check_analysis_inputs():
    """Protect immutable originals and accept only the permitted December state."""
    manifest = json.loads((ROOT / "docs/supplied_file_checksums.json").read_text())
    protected = {name: record for name, record in manifest["files"].items()
                 if record["policy"] != "may-update"}
    for name, record in protected.items():
        if record["policy"] == "immutable" and hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != record["sha256"]:
            raise AssertionError(f"Protected supplied file changed: {name}")
    from src.production import validate_december
    validate_december(pd.read_csv(ROOT / "data/december_chart_inputs.csv"), require_predictions=False)
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in protected}


def main():
    before = check_analysis_inputs()
    train = pd.read_csv(ROOT / "data/train_test.csv")
    validation = pd.read_csv(ROOT / "data/validation.csv")
    result = coverage(train, validation)
    checks = inference_checks(train, validation)
    development, final = result["development"], result["final"]
    masks = result["masks"]
    unseen_routes = final.loc[masks["Unseen directed route"], ["pickup", "delivery", "route"]]
    route_counts = unseen_routes.groupby(["pickup", "delivery", "route"]).size().rename("final_rows").reset_index()
    route_counts = route_counts.sort_values(["final_rows", "route"], ascending=[False, True])
    output_dir = ROOT / "reports"
    output_dir.mkdir(exist_ok=True)
    route_counts.to_csv(output_dir / "unseen_routes.csv", index=False)
    coordinate_missing = int(final.loc[masks["Completely new city at either endpoint"], COORDINATES].isna().any(axis=1).sum())
    city_rows = []
    for city in result["new_cities"]:
        pickup = final.pickup.eq(city)
        delivery = final.delivery.eq(city)
        coordinates = pd.concat([
            final.loc[pickup, ["pickup_lat", "pickup_lon"]].rename(columns={"pickup_lat": "lat", "pickup_lon": "lon"}),
            final.loc[delivery, ["delivery_lat", "delivery_lon"]].rename(columns={"delivery_lat": "lat", "delivery_lon": "lon"}),
        ]).drop_duplicates()
        pairs = "; ".join(f"({row.lat:.5f}, {row.lon:.5f})" for row in coordinates.itertuples())
        city_rows.append([city, int(pickup.sum()), int(delivery.sum()), int((pickup | delivery).sum()), pairs])
    lines = [
        "# Unseen categories and inference robustness", "",
        "Reproduce with `.venv/bin/python scripts/check_unseen_categories.py`. Coverage uses all 48,000 labeled development rows and all 12,000 unlabeled final-validation rows. Final-validation data is used only to describe coverage and test prediction compatibility; no accuracy metrics, tuning, or model selection use it.", "",
        "## City coverage", "",
        table(["Coverage", "Development", "Final validation"], [
            ["Distinct pickup labels", development.pickup.nunique(), final.pickup.nunique()],
            ["Distinct delivery labels", development.delivery.nunique(), final.delivery.nunique()],
            ["Distinct cities across both roles", len(result["train_cities"]), len(result["final_cities"])],
            ["Distinct directed routes", development.route.nunique(), final.route.nunique()],
            ["Distinct equipment labels", development.equipment.nunique(), final.equipment.nunique()],
        ]), "",
        "**Equipment labels:** development and final validation both contain Dry Van, Flatbed, and Reefer; no new equipment label occurs in the supplied final inputs. Synthetic inference tests cover a new equipment label as well.", "",
        "**Cities seen in development:** " + ", ".join(result["train_cities"]) + ".", "",
        "**Cities seen in final validation:** " + ", ".join(result["final_cities"]) + ".", "",
        "**Completely new cities:** " + ", ".join(result["new_cities"]) + ".", "",
        "A completely new city is absent from both development endpoint roles. A role-specific unknown label is absent from its corresponding development column. These measures are kept distinct; they identify the same new cities in the supplied data. Missing categories, if present, are explicit strings rather than dropped rows.", "",
        table(["Final-validation coverage check", "Rows", "% of all 12,000 rows"], [
            [name, int(mask.sum()), f"{100 * mask.mean():.6f}%"] for name, mask in masks.items()
        ]), "",
        table(["New city", "Pickup rows", "Delivery rows", "Any endpoint rows", "Supplied (lat, lon)"], city_rows), "",
        f"Rows involving a completely new city with at least one missing coordinate: **{coordinate_missing}**. The listed pairs are supplied coordinates, not independently verified geographic centroids.", "",
        "## Unseen directed routes", "",
        f"There are **{len(route_counts):,} distinct unseen pickup → delivery routes**, affecting **{len(unseen_routes):,} rows ({100 * len(unseen_routes) / len(final):.3f}%)**. Direction matters: A → B and B → A are distinct. The full list and row counts are in [unseen_routes.csv](unseen_routes.csv).", "",
        "Most frequent unseen routes:", "",
        table(["Pickup", "Delivery", "Final rows"], route_counts[["pickup", "delivery", "final_rows"]].head(15).values.tolist()), "",
        "Unseen routes can also connect familiar cities. A route lookup alone cannot cover these observations; city labels, coordinates, distance, and operational features remain available.", "",
        "## Model design and executed prediction checks", "",
        "The shared feature builder retains all four numeric coordinates and preserves new categorical strings. `src/preprocessing.py` provides an unfitted sklearn processor with median imputation and scaling for numeric features and `OneHotEncoder(handle_unknown=\"ignore\")` for categoricals. Put this processor after `FreightFeatureTransformer` inside the estimator pipeline; fit the entire pipeline on each chronological training fold only. Unknown labels produce zero indicators for that categorical block while numeric coordinates and other features remain available.", "",
        "CatBoost receives the same engineered DataFrame and explicit string categories. Native inference accepts labels absent from its training vocabulary; it does not require pre-extending categories using final data. Numeric missing values remain NaN for CatBoost.", "",
        f"**Executed diagnostic checks:** fitted a fixed Ridge pipeline and a fixed 12-tree, depth-3 CatBoost regressor on the first seven complete development dates: **{checks['training_rows']:,} rows ({checks['training_date_min']} through {checks['training_date_max']})**. Both predicted all **{checks['inference_rows']:,} final-validation rows** without exceptions, including every new-city and unseen-route row. Each returned **{checks['inference_rows']:,} finite predictions**. All four coordinate columns survived sklearn preprocessing.", "",
        "These bounded diagnostic fits verify inference compatibility only. They are not production models, were not saved, use no final-validation labels or evaluation set, and produce no submission predictions. No MAE, RMSE, R², or hidden score is available from unlabeled final data. Robustness tests also fit tiny synthetic models to check completely unseen pickup, delivery, route and equipment labels, missing categoricals, and geographic feature retention.", "",
        "Prediction compatibility does not establish accuracy for new cities. The separate frozen production configuration and development-only selection results are in `docs/model_selection.md`; completed output/scorer checks are recorded in `README.md` and `docs/worklog.md`.", "",
        "Original scorer, assessment PDF, development/final datasets and blank validation template remain unchanged. The December prediction column may be filled by production; this diagnostic does not modify any supplied input.", "",
    ]
    for name, digest in before.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise AssertionError(f"Protected supplied file changed during analysis: {name}")
    (output_dir / "unseen_categories.md").write_text("\n".join(lines))
    print(f"Cities: development={len(result['train_cities'])}, final={len(result['final_cities'])}, completely new={len(result['new_cities'])}")
    for name, mask in masks.items():
        print(f"{name}: {int(mask.sum()):,} / {len(final):,} ({100 * mask.mean():.6f}%)")
    print(f"Unseen distinct routes: {len(route_counts)}")
    print(json.dumps(checks, indent=2))
    print("Report: reports/unseen_categories.md; full route list: reports/unseen_routes.csv")


if __name__ == "__main__":
    main()
