# Freight Rate ML Assessment

Predict `posted_rate` in dollars for every final-validation freight load and generate the required December fixed-scenario chart. The primary model is a chronologically validated CatBoost regressor. Production training uses all 48,000 labeled rows; final unlabeled data is used only for input audits and inference.

## Quick start

Run from this repository's root. The executed environment uses **Python 3.12.14**. Compatible direct dependency ranges are in `requirements.txt`; `constraints.txt` pins the actual installed direct/transitive versions for reproduction.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -c constraints.txt
python run_pipeline.py
pytest -q
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. In a managed environment with restricted home-directory caches, set `MPLCONFIGDIR` and `XDG_CACHE_HOME` to writable directories; this workspace used `/tmp/freight-matplotlib` and `/tmp/freight-xdg-cache` respectively.

The direct dependencies are NumPy, pandas, matplotlib, scikit-learn, CatBoost, joblib, pytest, and python-docx. CatBoost saves native `.cbm` models. The DOCX dependency supports the assessment's report stage; it is not needed by the prediction code itself. Exact training package versions are also recorded in `artifacts/training_metadata.json`.

## Dataset and data quality

| Supplied file | Rows × columns | Date range / purpose |
| --- | ---: | --- |
| `data/train_test.csv` | 48,000 × 14 | 2025-01-01–2025-10-31; labeled development data |
| `data/validation.csv` | 12,000 × 13 | 2025-11-01–2025-12-31; unlabeled final inference |
| `data/validation_predictions_template.csv` | 12,000 × 2 | Unique final load IDs and blank rate placeholders |
| `data/december_chart_inputs.csv` | 31 × 7 | Fixed load, one row per day of December 2025 |

The target is `posted_rate`. `load_id` identifies a load and is never predictive. The development/final predictor schemas otherwise match. Final template IDs are exactly TE-000001 through TE-012000. Raw development and final-validation files, the blank template, the assessment PDF, and the supplied scorer remain byte-identical to the manifest in [docs/supplied_file_checksums.json](docs/supplied_file_checksums.json).

The supplied datasets and prediction template are included so the pipeline and tests run from a fresh clone.

| Finding | Development | Final inference |
| --- | ---: | ---: |
| Missing weight | 300 | 165 |
| Negative weight | 292 | 145 |
| Missing market_index | 374 | 249 |
| Missing quote_signal | 0 | 0 |

Missing and negative rows are retained. Weight becomes `abs(weight)` with separate indicators for original missingness and negativity. Negative values are consistent with a possible sign error, but the source does not prove their cause. Numeric NaNs are handled natively by CatBoost; malformed nonmissing numeric values and infinities fail explicitly. Ridge's imputation/scaling and categorical vocabulary are fitted on its training fold only. No targets or observations are removed merely for being large.

Development rates range from **$57.22 to $25,533**, with mean **$2,373.98** and median **$2,030.76**. The right tail explains why RMSE is much larger than MAE. Equipment and calendar periods differ in observed rate per mile; these descriptive associations do not prove causal effects.

Final inference introduces **eight new cities**, affecting **1,447 rows (12.0583%)**, and **736 distinct unseen directed routes**, affecting **1,461 rows (12.175%)**. Coordinates provide geographic information alongside categorical labels. CatBoost accepts unseen categorical strings, and the sklearn baseline uses `OneHotEncoder(handle_unknown="ignore")`. Final-data distribution comparisons are descriptive audits and do not influence model selection. Full findings and plots: [data audit](docs/data_audit.md), [EDA](reports/eda.md), and [unseen categories](reports/unseen_categories.md).

## Chronological validation

Final inference follows the development period, so local evaluation predicts the next two months from earlier observations. Random splitting would mix future and past observations. Dates are never shuffled across fold boundaries, and all rows from a date remain together.

| Fold | Train through | Validate | Train rows | Validation rows |
| --- | --- | --- | ---: | ---: |
| 1 | 2025-04-30 | 2025-05-01–2025-06-30 | 19,110 | 9,696 |
| 2 | 2025-06-30 | 2025-07-01–2025-08-31 | 28,806 | 9,671 |
| 3, primary holdout | 2025-08-31 | 2025-09-01–2025-10-31 | 38,477 | 9,523 |

Each training window starts 2025-01-01. Primary metric is MAE; secondary metrics are RMSE and R², evaluated on the original dollar scale. Reported means are unweighted means of the three fold metrics, not pooled metrics. Learned preprocessing and model fitting use only each applicable training fold. Final-validation inputs/labels are not used for selection, tuning, or accuracy metrics.

Initial CatBoost experiments used early stopping on the chronological validation window. Shortlisted models were subsequently confirmed at fixed median tree budgets without an eval_set. The same historical folds were reused for selection, so the local results are model-selection estimates rather than an independent test. See [validation plan](docs/validation_plan.md) and [selection evidence](docs/model_selection.md).

## Features and models evaluated

[src/features.py](src/features.py) implements one row-wise feature builder for training and inference. It uses a fixed calendar reference of 2025-01-01, explicit categorical missing strings, no target encoding, and an ordered whitelist excluding `load_id`, `posted_rate`, and output predictions.

The primary model's 26 selected features comprise:

- Operational: distance, equipment, absolute weight, weight missing/negative flags.
- Geographic: pickup/delivery city names and their four latitude/longitude fields.
- Route: directed pickup → delivery string.
- Calendar: month, day, weekday, day of year, ISO week, weekend, days since reference, and weekday/year sine/cosine pairs.
- Market: market_index, its missingness flag, and quote_signal.

Executed experiments compare global median, distance-only linear regression, Ridge with fold-fitted preprocessing, three modest CatBoost hyperparameter configurations, nested A–F feature groups, raw/cleaned weight, and direct/log targets. All use the same chronological folds. Full parameters, timings, retained trees, per-fold metrics, and limitations are saved in [model comparison](reports/model_comparison.md), [CSV results](reports/model_comparison.csv), and [feature ablation](reports/feature_ablation.md).

Actual mean chronological results:

| Model | MAE ($) | RMSE ($) | R² |
| --- | ---: | ---: | ---: |
| Global median | 1,151.00 | 1,572.59 | -0.0685 |
| Distance linear | 201.23 | 658.83 | 0.8124 |
| Ridge | 252.81 | 687.48 | 0.7954 |
| CatBoost F-clean/direct, fixed 284 trees | 131.00 | 634.46 | 0.8260 |
| **Selected CatBoost F-clean/log1p, fixed 399 trees** | **129.41** | **634.96** | **0.8258** |
| December CatBoost C-clean/direct, fixed 541 trees | 143.85 | 637.14 | 0.8245 |

## Frozen final models

The primary model is **CatBoostRegressor with 399 trees**, depth 6, learning_rate 0.05, l2_leaf_reg 5, loss_function MAE, random_seed 42, thread_count 4, and nan_mode Min. It learns `log1p(posted_rate)` and converts predictions with `expm1`. Its Sep–Oct holdout metrics are **MAE $116.71, RMSE $636.50, R² 0.8260**.

This configuration improves fixed-budget mean MAE over the direct-target alternative and outperforms the simple baselines. It does not win every metric: the direct alternative has slightly better recent MAE and mean RMSE. Selection considered chronological MAE, recent performance, fold stability, missing/unseen robustness, and input availability. Exact features/parameters and the selection rule were fixed before fitting on all labeled rows in [docs/final_model_config.json](docs/final_model_config.json).

December uses a **separately validated 541-tree, direct-target group-C model** with the same depth, learning rate, regularization, seed, and native missing handling. Its 22 features exclude route and all market/quote fields. Sep–Oct metrics are **MAE $113.66, RMSE $635.31, R² 0.8267**. Coordinates come from the supplied development city lookup. No future market_index or quote_signal is invented.

The scenario holds Lexington → Fort Wayne, 360 miles, Dry Van, and 32,000 lb fixed while dates cover December 1–31. Only `predicted_rate` is filled in the existing seven-column CSV. Historical freight-fold metrics do not establish accuracy for this future fixed lane.

## Production command and outputs

```bash
python run_pipeline.py
```

This command checks all four inputs, immutable hashes, schemas, IDs, dates, and saved chronological results; builds the frozen features; fits both fixed configurations on **all 48,000 rows** without early stopping; saves native models and preprocessing metadata; predicts both outputs; and checks the written CSVs. It does not rerun research, tune parameters, or regenerate model selection. It works with the original blank December file or an already-filled file. Rerunning replaces production artifacts and output predictions while preserving raw development/final data and the blank prediction template.

Native model binaries and runtime JSON under `artifacts/` are generated locally and ignored by Git. Run the pipeline before tests on a fresh clone. Frozen specifications, executed experiment records, report, and submission CSV/chart are versioned; rerunning does not create a new history of volatile model serialization or timing logs.

| Output | Contents |
| --- | --- |
| `validation_predictions.csv` | Exactly 12,000 rows, `load_id,predicted_rate`; joined one-to-one by ID in template order |
| `data/december_chart_inputs.csv` | Original seven columns and 31 fixed inputs; only rates filled |
| `artifacts/primary_model.cbm`, `december_model.cbm` | Full-development native CatBoost models |
| `artifacts/preprocessing.json` | Frozen specifications, missing policies, reference date, city lookup, source hashes |
| `artifacts/training_metadata.json` | Parameters, features, rows/dates, seed, package versions, historical metrics, model hashes/times |
| `artifacts/prediction_summary.json` | Actual output summaries and CSV hashes |
| `artifacts/pipeline_checks.json` | Preflight/feature/output verification and run duration |

Both outputs must be finite and strictly positive. A frozen $0.01 guard follows nonfinite rejection; it changed zero predictions in the executed production run. Native model reload and CSV round-trip checks verify saved outputs. For inference from saved artifacts alone, run `python -m src.predict`; `python -m src.train` performs only production training.

## Tests and official scorer

```bash
pytest -q
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Tests cover feature exclusion, whole-date chronological folds, missing/negative weight, missing market values, unseen labels, fold-fitted preprocessing, IDs and shuffled ID joins, exact output schemas/counts, finite positive rates, complete December dates, input preservation, saved-model reproduction, and malformed-input rejection. Production-output tests require generated artifacts: run the pipeline first in a fresh clone. Small synthetic model fits in tests do not rerun the full experiment suite. Two optional-numba optimization warnings arise from the historical custom dollar-scale stopping-metric test and do not indicate failures.

Executed verification: **109 tests and 4 subtests passed**. `pytest -q`, `python run_pipeline.py`, and the official scorer all succeeded in that order. The production rerun passed 47 input contracts and generated byte-identical validation and December CSVs compared with the previous executed run. `.gitattributes` preserves raw CSV/scorer bytes and stable source-file line endings across clones. The execution evidence is in [docs/worklog.md](docs/worklog.md).

The supplied `score.py` is immutable. It validates output contracts and creates [scorer_results/candidate_december.png](scorer_results/candidate_december.png). Successful output is:

```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```

The scorer does not report prediction accuracy. Spotter's hidden validation metrics remain unavailable.

## Repository structure

```text
data/                       Supplied raw data, blank ID template, editable December rate column
src/features.py             Shared deterministic feature engineering
src/evaluate.py             Development-only chronological experiments
src/preprocessing.py        Fold-fitted sklearn preprocessing
src/model_config.py         Frozen configuration and target conversion
src/train.py                Full-development production training
src/predict.py              Saved-model inference and ID joins
src/production.py           Shared integrity and output contracts
scripts/                    Input audit, EDA, unseen-category checks, research reporting
tests/                      Feature, validation, model, and production checks
reports/                    Executed experiments, EDA, ablations, figures, unseen categories
docs/                       Data/feature/validation notes, frozen config, engineering log
artifacts/                  Models, preprocessing, metadata, production summaries/checks
scorer_results/             Official chart and saved scorer stdout
report/                     Markdown and DOCX assessment report
run_pipeline.py             Single production entry point
score.py                    Original assessment scorer
requirements.txt            Compatible direct dependency ranges
constraints.txt             Exact tested dependency versions
pytest.ini                  Test discovery and repository import configuration
Freight_Rate_ML_Assessment.pdf
```

Detailed decisions and executed commands are in [docs/worklog.md](docs/worklog.md). EDA can be reproduced separately with `python scripts/eda.py`; it is not part of the production command.

## Report and walkthrough

- [Markdown report](report/report.md) and [DOCX report](report/Freight_Rate_ML_Assessment_Report.docx): all 12 requested sections, actual metrics, chronological splits, and the embedded original December chart. The DOCX was opened and rendered successfully to four pages.

Regenerate both report formats from the current saved evidence with `python scripts/build_report.py`. The builder uses python-docx, verifies metric agreement, reopens the DOCX, checks all required headings, and verifies the embedded PNG is byte-identical to the scorer chart. Rendered page count was separately checked using the available office renderer. The completed assessment was published on 2026-10-05 to [GitHub](https://github.com/ahmadzaid2002dlhom-code/freight-rate-ml-assessment), on both `main` and `feature/freight-rate-ml-assessment`.

The assessment also requires an actual 2–3 minute Loom recording covering EDA findings, data-quality handling, model choice, chronological validation, and the important code. A recording link remains a separate submission item.

## Limitations

- No labeled November/December data or untouched independent test is available; historical folds were reused for stopping and selection.
- New-city inference works mechanically, but its future accuracy is unmeasured. Ten development months do not establish full-year seasonality, and tree models have limited extrapolation beyond observed calendar ranges.
- Market/quote provenance and timestamp availability are unspecified. They exist in the supplied primary inference schema; there is no evidence sufficient to call them leakage. Their gains vary by chronological fold.
- Large-rate tail errors remain substantial. Log modeling improves mean MAE but not every metric. No unverified outlier was deleted or clipped.
- Supplied coordinates and negative-weight origins are not independently verified. December uses available fields only and remains a scenario forecast, not a known future rate.
- Pins and seed aid reproduction; native model bytes and training timestamps can vary across runs, while package/platform changes can affect numerical results.
