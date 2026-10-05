# Shared candidate feature contract

Historical feature-contract checkpoint before model training. The execution counts and original-file claims below describe that stage, when the December output was blank. Production models and predictions now exist; current status and checks are in `README.md`, `docs/worklog.md`, and `artifacts/training_metadata.json`.

`src/features.py` provides `build_features(frame)` for direct use and
`FreightFeatureTransformer()` for sklearn pipelines. Both call the same
deterministic transformation. The existing `build_weight_features(frame)` API
remains available to the audit and EDA scripts.

```python
from src.features import build_features, FreightFeatureTransformer

X = build_features(raw_loads)
# In a later training pipeline, put learned preprocessing after this adapter.
feature_step = FreightFeatureTransformer()
```

The output is a DataFrame with 26 ordered candidate columns:

| Group | Features |
| --- | --- |
| Operational | distance, equipment, weight_clean, weight_missing, weight_negative |
| Location | pickup, delivery, pickup_lat, pickup_lon, delivery_lat, delivery_lon |
| Route | route |
| Calendar | month, day, day_of_week, day_of_year, iso_week, weekend, days_since_reference_date |
| Cyclical | sin_day_of_week, cos_day_of_week, sin_day_of_year, cos_day_of_year |
| Market | market_index, market_index_missing, quote_signal |

Only this whitelist is returned. `load_id`, `posted_rate`, `predicted_rate`, raw
`weight`, raw `date`, and any other input columns are excluded. No target encoding,
target-derived aggregate, fitted statistic, or categorical vocabulary is learned.
The sklearn adapter ignores `y`; its `fit` validates inputs and records output
column names only. The jointly validated settings are now frozen in
`docs/final_model_config.json` and explained in `docs/model_selection.md`.
`build_model_features` selects the exact frozen column list from this shared
builder. It supports `weight_raw` for the executed raw-weight comparisons;
both selected configurations retain cleaned weight and its original-value flags.

## Missing and malformed inputs

- `date`, `distance`, and `weight` source columns are required. Invalid, missing,
  numeric-timestamp, or timezone-aware dates raise an explicit error. Calendar
  calculations use normalized timezone-naive dates.
- Missing numeric measurements remain NaN. Nonmissing malformed values and
  infinities raise explicit errors; rows are never silently removed. Numeric
  strings are parsed consistently.
- `weight_clean = abs(weight)`. `weight_missing` records original missingness;
  `weight_negative` records the original negative sign. Missing weights stay NaN
  and receive missing=1, negative=0.
- `market_index_missing` is 1 for missing market values, including when the source
  column is absent. An absent optional numeric column is entirely NaN.
- Missing, empty, or whitespace-only categorical values become the explicit string
  `__MISSING__`. Other categorical labels are converted to strings and preserved.
  The route is the transformed pickup string plus `" -> "` plus the delivery string.
- Unseen labels are retained. A later sklearn encoder must use
  `OneHotEncoder(handle_unknown="ignore")`; CatBoost receives explicit string
  categories. A later model must support numeric NaN natively or use an imputer
  fitted inside its applicable training fold. Scaling and encoding follow the same
  fold-only fitting rule.

`src/preprocessing.py` now provides `build_sklearn_preprocessor()` implementing
this unfitted sklearn imputation/scaling/encoding configuration. The executed
unknown-category prediction checks and city/route coverage are documented in
`reports/unseen_categories.md`; chronological predictive accuracy remains to be
evaluated separately.

The output preserves row count, order, and index without changing the input.

## Calendar definitions

The default reference is the fixed date **2025-01-01**, never the earliest date in
the current input batch. Changing the reference is an explicit configuration
choice that must be the same for training and inference.

Month and day use ordinary calendar numbering. Monday is day_of_week=0, Sunday=6;
Saturday and Sunday have weekend=1. `iso_week` uses ISO week numbering: December
31, 2025 belongs to week 1 of ISO year 2026. `days_since_reference_date` is the
integer calendar-day difference.

Weekly sine/cosine features use phase `2*pi*day_of_week/7`. Annual features use
`2*pi*(day_of_year-1)/year_length`, where year_length is 365 or 366 according to
the row's calendar year. January 1 therefore has annual sine=0 and cosine=1.
Features depend on each row and the fixed configuration, not other rows in a batch.

## December availability

The supplied December file contains the operational fields, pickup, delivery,
date, and the output placeholder. The same builder returns all 26 columns, with
the four coordinates, market_index, and quote_signal as NaN, and
market_index_missing=1. It does not infer future market or quote values, perform a
coordinate lookup, or change the seven-column source file. A deterministic lookup
from supplied city coordinates can be added to the inference inputs later if the
selected model requires it. The frozen December configuration uses validated
group C (operational, calendar, city, coordinate features), omitting unavailable
market fields. The production pipeline must apply the deterministic coordinate
lookup and preserve the original seven-column file.

## Executed verification

Command: `.venv/bin/python -m pytest -q`.

Result: **45 tests passed, plus 4 subtests**. Tests cover feature exclusion and
input immutability, missing/negative weight, missing market index, explicit
categorical strings, unseen sklearn categories, CatBoost Pool construction,
calendar boundaries and leap years, batch independence, sklearn cloning and
adapter equivalence, malformed values, and the three real input schemas.
CatBoost Pool construction checks input compatibility; trained-model robustness
must still be checked during the later modeling stage.

| Source | Output shape | Missing weight | Negative weight | Missing market index |
| --- | --- | --- | --- | --- |
| Development | 48,000 × 26 | 300 | 292 | 374 |
| Final validation | 12,000 × 26 | 165 | 145 | 249 |
| December | 31 × 26 | 0 | 0 | 31 |

The original scorer, all four supplied CSV files, and assessment PDF match their
recorded original SHA-256 hashes. No production model has been trained. Subsequent
unseen-category checks are recorded in `reports/unseen_categories.md`; full
feature/weight/target experiments and the frozen choice are in
`reports/feature_ablation.md` and `docs/model_selection.md`.
