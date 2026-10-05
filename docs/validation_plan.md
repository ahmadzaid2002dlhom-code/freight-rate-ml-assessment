# Chronological validation plan

This records the strategy defined before experimentation. The fold definitions remain authoritative; references to later experiments describe that historical checkpoint. Model selection and production are now completed and documented in `docs/model_selection.md` and `README.md`.

The labeled development file has 48,000 rows from January 1 through October 31,
2025. Final unlabeled inference covers November 1–December 31, 2025. All local
model evaluation uses development targets only; final inference data is excluded
from model selection, tuning, and accuracy metrics.

| Fold | Training dates | Validation dates | Training rows | Validation rows |
| --- | --- | --- | --- | --- |
| 1 | 2025-01-01–2025-04-30 | 2025-05-01–2025-06-30 | 19,110 | 9,696 |
| 2 | 2025-01-01–2025-06-30 | 2025-07-01–2025-08-31 | 28,806 | 9,671 |
| 3 (primary) | 2025-01-01–2025-08-31 | 2025-09-01–2025-10-31 | 38,477 | 9,523 |

Counts were computed directly from the supplied development file. Training
expands with time. Every row of a calendar date remains together; no fold
contains a future observation in training relative to its validation period.
The three validation windows partition May–October with no overlapping dates.

Primary metric: **MAE**. Secondary metrics: **RMSE** and **R2**. Record per-fold
metrics and unweighted fold means, and emphasize the primary September–October
holdout when assessing models. RMSE highlights larger errors; R2 compares squared
errors against the holdout mean predictor. Unweighted mean fold RMSE/R2 are not
pooled metrics, and expanding training windows make folds statistically dependent.

A random 80/20 split would let models learn from later observations while
predicting earlier ones. It also mixes calendar/market regimes and familiar
locations across sides. Chronological splits test extrapolation into the next two
months, matching the ordering and horizon of the November–December task more
closely. They cannot guarantee that future regimes or new-city accuracy match
historical results; November/December target seasonality is unobserved locally.

Fit imputation, scaling, categorical vocabularies, estimators, and any future
learned encodings on each training fold only. Stateless row-wise features may be
generated before splitting. Use fresh preprocessing and estimators for every
fold. Keep target and identifiers outside predictive feature lists. Preserve
missing and negative-weight rows and target outliers.

The fixed folds are implemented in `src/evaluate.py`. Baseline configurations
were fixed before execution. The limited CatBoost comparison uses the same
three outer folds and three configurations fixed before execution:

| Configuration | Depth | Learning rate | l2_leaf_reg |
| --- | --- | --- | --- |
| catboost_d7_lr003_l2_3 | 7 | 0.03 | 3 |
| catboost_d6_lr005_l2_5 | 6 | 0.05 | 5 |
| catboost_d8_lr004_l2_7 | 8 | 0.04 | 7 |

All use MAE loss/evaluation metric, an upper bound of 1,500 iterations,
random_seed=42, native categorical handling, native numeric NaN handling,
100-round early-stopping patience, and use_best_model=True. Internal CatBoost
categorical statistics and tree fitting use only training-fold observations.
Each fold's later outer validation period serves as its eval_set, and its labels
select the best iteration. This is nine CatBoost fits with joint configuration
changes; it does not isolate the effect of any single parameter.

These reused validation scores are model-selection estimates, not results from
an untouched independent test. Comparing configurations and choosing iterations
on the same holdout can make its score optimistic; the baseline models do not use
the validation labels for stopping. Future final claims must retain this
limitation. Final unlabeled November–December inputs remain excluded. An
independent later evaluation or nested chronological stopping split could provide
a stricter estimate if additional labeled data become available.

Fix the final configuration before any production fit on all 48,000 labeled
observations. Feature ablation and target-transform comparisons remain later
development-only experiments.
