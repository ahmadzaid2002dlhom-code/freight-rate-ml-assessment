"""Small shared checks for production artifacts and submission contracts."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREDICTION_COLUMNS = ["load_id", "predicted_rate"]
DECEMBER_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check_immutable_inputs(root=ROOT):
    manifest = json.loads((root / "docs/supplied_file_checksums.json").read_text())
    for relative, record in manifest["files"].items():
        if record["policy"] == "immutable" and sha256(root / relative) != record["sha256"]:
            raise ValueError(f"Immutable supplied file changed: {relative}")


def check_ids(frame, rows, label):
    if len(frame) != rows or "load_id" not in frame:
        raise ValueError(f"{label} must have {rows} rows and load_id")
    ids = frame["load_id"]
    if ids.isna().any() or ids.duplicated().any() or not ids.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError(f"{label} contains missing, duplicate, or invalid IDs")


def check_rates(values):
    rates = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(rates).all() or not (rates > 0).all():
        raise ValueError("Predictions must be finite and strictly positive")


def align_predictions(validation, template, rates, expected_rows=12000):
    """Explicit one-to-one ID join in template order, independent of input order."""
    check_ids(validation, expected_rows, "Validation")
    check_ids(template, expected_rows, "Template")
    if template.columns.tolist() != PREDICTION_COLUMNS:
        raise ValueError("Unexpected template schema")
    if set(validation.load_id) != set(template.load_id):
        raise ValueError("Validation IDs do not match template IDs")
    rates = np.asarray(rates, dtype=float)
    if rates.shape != (len(validation),):
        raise ValueError("Prediction count does not match validation rows")
    check_rates(pd.Series(rates))
    by_id = pd.DataFrame({"load_id": validation.load_id.to_numpy(), "predicted_rate": rates})
    output = template[["load_id"]].merge(by_id, on="load_id", how="left", sort=False, validate="one_to_one")
    validate_submission(output, template, expected_rows)
    return output


def validate_submission(output, template, expected_rows=12000):
    if output.columns.tolist() != PREDICTION_COLUMNS:
        raise ValueError("Predictions must have exactly load_id,predicted_rate")
    check_ids(output, expected_rows, "Predictions")
    check_ids(template, expected_rows, "Template")
    if output.load_id.tolist() != template.load_id.tolist():
        raise ValueError("Prediction IDs must exactly follow the template")
    if output.isna().any().any():
        raise ValueError("Missing prediction values")
    check_rates(output.predicted_rate)


def validate_december(frame, require_predictions=True, original_inputs=None):
    if frame.columns.tolist() != DECEMBER_COLUMNS or len(frame) != 31:
        raise ValueError("December requires the original seven-column schema and 31 rows")
    dates = pd.to_datetime(frame.date, format="%Y-%m-%d", errors="raise")
    ordered_dates = pd.date_range("2025-12-01", "2025-12-31")
    expected = set(ordered_dates)
    if dates.isna().any() or dates.duplicated().any() or set(dates) != expected:
        raise ValueError("December dates must cover each day exactly once")
    if frame.date.tolist() != ordered_dates.strftime("%Y-%m-%d").tolist():
        raise ValueError("December must preserve the original ordered dates and YYYY-MM-DD strings")
    if not (frame.pickup.eq("Lexington") & frame.delivery.eq("Fort Wayne") & frame.equipment.eq("Dry Van")).all():
        raise ValueError("December fixed lane or equipment changed")
    if not (pd.to_numeric(frame.distance, errors="raise").eq(360) & pd.to_numeric(frame.weight, errors="raise").eq(32000)).all():
        raise ValueError("December fixed distance or weight changed")
    if original_inputs is not None and frame[DECEMBER_COLUMNS[:-1]].to_dict("records") != original_inputs:
        raise ValueError("Original December inputs or ordering changed")
    if require_predictions:
        check_rates(frame.predicted_rate)


def city_coordinate_lookup(training):
    """Use consistent supplied coordinates across both historical city roles."""
    cities = pd.concat([
        training[[role, f"{role}_lat", f"{role}_lon"]].set_axis(["city", "lat", "lon"], axis=1)
        for role in ("pickup", "delivery")
    ], ignore_index=True).drop_duplicates()
    if cities.isna().any().any() or not np.isfinite(cities[["lat", "lon"]].to_numpy(dtype=float)).all():
        raise ValueError("City lookup has missing or nonfinite coordinates")
    if cities.city.duplicated().any():
        raise ValueError("Supplied coordinates conflict for a city")
    return {row.city: {"lat": float(row.lat), "lon": float(row.lon)} for row in cities.itertuples(index=False)}


def add_december_coordinates(frame, lookup):
    enriched = frame.copy(deep=True)
    for role in ("pickup", "delivery"):
        unknown = set(frame[role]) - set(lookup)
        if unknown:
            raise ValueError(f"Missing supplied coordinates for {sorted(unknown)}")
        for axis in ("lat", "lon"):
            enriched[f"{role}_{axis}"] = frame[role].map(lambda city: lookup[city][axis])
    return enriched


def rate_summary(values):
    rates = np.asarray(values, dtype=float)
    check_rates(pd.Series(rates))
    return {"rows": len(rates), "minimum": float(rates.min()), "maximum": float(rates.max()),
            "mean": float(rates.mean()), "median": float(np.median(rates))}


def write_csv(path, frame):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
