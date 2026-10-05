# Engineering log

This log records executed analysis, model decisions, and verification. Proposed improvements are identified separately from completed work.

## Data audit and feature decisions

- Development data contains 48,000 labeled rows from 2025-01-01 through 2025-10-31. Final inference contains 12,000 unlabeled rows from 2025-11-01 through 2025-12-31. Final inputs were audited for compatibility and used for inference only.
- `posted_rate` is the supervised target. `load_id`, targets, and prediction columns are excluded by the feature whitelist. Raw development/final data, the blank prediction template, assessment PDF, and original scorer match the SHA-256 manifest.
- Missing weight affects 300 development and 165 final rows; negative weight affects 292 and 145. Missing market_index affects 374 and 249. All rows are retained. Absolute weight plus original missing/negative indicators was compared against raw weight in the same chronological folds; negative-value origins remain unverified.
- Shared row-wise features use a fixed 2025-01-01 calendar reference, explicit categorical missing strings, supplied coordinates, and no external target encoding. Learned preprocessing is fitted only on the applicable training fold. Unseen sklearn categories use `handle_unknown="ignore"`.
- Final inputs introduce eight cities and 736 directed routes. Compatibility was checked without treating final-input distributions as model-selection evidence.

## Chronological evaluation and frozen model selection

The three expanding folds train through April, June, and August 2025, then validate on May-June, July-August, and September-October respectively. Whole dates stay together; dates are never shuffled across fold boundaries. Train/validation counts are 19,110/9,696, 28,806/9,671, and 38,477/9,523. MAE is primary; RMSE and R2 use the original dollar scale. Fold means are unweighted.

Executed comparisons cover global median, distance linear regression, Ridge, three modest CatBoost configurations, nested feature groups, raw/cleaned weight, and direct/log targets. Early-stopping candidates were followed by fixed-tree-budget confirmation without an eval_set. Reusing historical folds for selection means these are selection estimates, not an independent test. Complete parameters, folds, seeds, timings, and metrics are in `reports/model_comparison.csv`, `reports/ablation_results.csv`, and the selection documents.

| Model | Mean MAE ($) | Mean RMSE ($) | Mean R2 |
| --- | ---: | ---: | ---: |
| Global median | 1,151.00 | 1,572.59 | -0.0685 |
| Distance linear | 201.23 | 658.83 | 0.8124 |
| Ridge | 252.81 | 687.48 | 0.7954 |
| F-clean/direct, fixed 284 trees | 131.00 | 634.46 | 0.8260 |
| Selected F-clean/log, fixed 399 trees | 129.41 | 634.96 | 0.8258 |
| December C-clean/direct, fixed 541 trees | 143.85 | 637.14 | 0.8245 |

The primary model has 26 features and fits log1p(posted_rate), with expm1 prediction conversion. The December model has 22 available features and a direct target; it excludes route and market/quote fields. Both use depth 6, learning_rate 0.05, l2_leaf_reg 5, MAE loss, random_seed 42, thread_count 4, and native numeric missing handling. Exact settings were frozen before fitting all 48,000 labeled rows. Selection improves mean MAE without claiming to win every metric. Market/quote timestamps and future-lane accuracy remain limitations.

## 2026-10-03 to 2026-10-04 - Production and verification

The production command validates supplied inputs and frozen experiment evidence, fits both fixed models without early stopping, saves native models and metadata, joins final predictions one-to-one by load_id, and verifies both CSVs. No future market or quote values are fabricated for December. Its coordinates come from the supplied development city lookup.

Final audit execution on 2026-10-04 ran `pytest -q`, `python run_pipeline.py`, and the unchanged official scorer successfully. **109 tests passed, 4 subtests passed**, with two existing optional-numba optimization warnings. The pipeline passed 47 input contracts. Both model budgets stayed at 399/541 trees. Two regressions in auxiliary integrity checks and December date/order checks were repaired and verified before the final run.

- Validation: exactly 12,000 unique template IDs, columns `load_id,predicted_rate`, finite positive rates. SHA-256: `b8fb2f10e218c3357ae3487d91818e3d404c074dbbe32b3ab8f53ca85f085397`.
- December: exactly 31 original ordered dates and fixed inputs, unchanged seven-column schema, finite positive rates. SHA-256: `bc96e22b7ca8b5dd1a7b9e683b7283d0193fec46a7ce3b0b64dd876f473ddc23`.
- Production reruns and saved-model round trips reproduced the numeric outputs. The frozen positivity guard changed zero predictions.
- The original scorer accepted both files and created `scorer_results/candidate_december.png`. It validates contracts and draws the chart; it reports no local accuracy. Hidden Spotter metrics are unavailable.
- The report generator verified all 12 sections, seven tables, and the exact embedded scorer chart. The DOCX was rendered and visually checked as four pages. The technical report documents validation, model choice, data quality, limitations, and reproduction commands.

## 2026-10-05 - Submission packaging

A read-only audit of the 85-file submission verified all required dependencies, five immutable supplied hashes, input/output CSV contracts, unchanged prediction hashes, and the report's exact embedded chart. All 24 Python modules parsed successfully, and relative Markdown links resolved. The repository retains runnable code/tests/dependencies, required inputs/outputs, frozen settings, recorded research evidence, the report, and the official chart. Generated models/runtime metadata, environments, caches, credentials, and temporary files remain outside version control.

## 2026-10-06 - Repository organization

The submission history groups assessment setup, shared features and validation, data analysis, model selection, the production pipeline, tests, documentation, and final outputs into separate commits. Packaging changes affect documentation and Git history only; source, tests, frozen model settings, supplied inputs, predictions, report, and chart are unchanged. The prior execution results above remain the evidence for training, tests, and scoring; packaging did not rerun those commands.

An actual 2-3 minute Loom recording/link is still required. An independent later labeled test, further tail/new-city evaluation, and drift monitoring are proposed future work, not executed experiments.

## 2026-10-06 - Windows README and demo verification

Executed the README workflow on the owner's Windows 11 device using a newly created project `.venv` and Python 3.12.5. Installation with `-r requirements.txt -c constraints.txt` succeeded. All 31 applicable pinned packages matched their installed versions, and `python -m pip check` reported no broken requirements. Added the Windows-only pytest dependency `colorama==0.4.6` to the constraints. README commands use the venv interpreter directly and enable UTF-8 without depending on PowerShell activation policy.

The initial full run and 109 existing tests succeeded. Inspection found platform-dependent CSV newlines, so the production writer now explicitly emits LF. Added a regression for CSV newlines and round-trip contents. Its initial setup encountered an access-denied Windows shared pytest temporary directory; `pytest.ini` now uses the dedicated ignored `.cache/pytest-temp` directory. The final suite passed: **110 tests passed, 4 subtests passed** in 17.29 seconds, with the same two optional-numba optimization warnings.

The final full pipeline passed all 47 input contracts in 77.76 seconds, fitting the frozen 399-tree primary model in 33.15 seconds and the 541-tree December model in 41.55 seconds on all 48,000 labeled rows. Native serialization checks passed; research experiments were not rerun. Both outputs were positive and finite, with zero predictions changed by the floor guard. All five immutable supplied files retained their original SHA-256 values.

- Validation CSV: 12,000 exact unique template IDs in the required two-column schema; SHA-256 `e78efbd071dd54242e9b53d4c8c7e9f105864d2fb58e5b1e011a754d5cfeeaa3`. Compared by `load_id` against the preserved earlier submission, 55 rates differ by at most `9.094947017729282e-13` dollars. These are platform floating-point differences, not a new model selection or accuracy result.
- December CSV: all 31 fixed rows and ordered dates preserved; SHA-256 `bc96e22b7ca8b5dd1a7b9e683b7283d0193fec46a7ce3b0b64dd876f473ddc23`, identical to the earlier submission.
- The original official scorer accepted both actual Windows outputs and regenerated `scorer_results/candidate_december.png`. Its Windows-rendered PNG hash is `6d34be6a92a4c013dc840c60914c28b8fa1d6386919a6ab3cb7cad55b82ed260`. The December numeric series is unchanged; rendering differs across platforms. No hidden accuracy metrics are available.
- Executed `python -m src.predict` using the saved native models in 3.67 seconds; both CSV hashes were reproduced exactly. This command is suitable for the short live demo; full fitting should be completed before recording.

Local run logs, environment records, comparisons, and pre-run backups are retained under the ignored `.local/demo-readiness/` directory. Generated models, the environment, and caches remain local.

The README report-regeneration command succeeded with the project interpreter, checking all 12 sections, seven tables, actual metrics, and the exact embedded scorer PNG. A read-only Windows Word render exposed absent DejaVu fonts, two nearly empty overflow pages, and a feature-table overrun. Updated the builder to installed Calibri/Consolas, explicitly suppressed the inherited Title border, and included the verified PowerShell commands. The corrected report rendered to four Letter pages; every page was visually inspected, with complete tables, readable commands, and the full official chart/caption. Preview generation preserved DOCX bytes and closed its own hidden Word instance. Rendered QA files remain local.
