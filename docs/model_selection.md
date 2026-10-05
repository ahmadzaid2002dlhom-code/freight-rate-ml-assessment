# Frozen model selection before production training

**Decision: CatBoostRegressor, group F, clean weight representation, log1p target, 399 fixed iterations.** The status is frozen_before_production_training. `docs/final_model_config.json` is the machine-readable configuration. No model has been fitted on all 48,000 development rows, saved for production, or used to generate final submission files at this checkpoint.

## Selection evidence and rule

All decisions use labeled development folds only. Intermediate comparisons use a 1% mean-MAE shortlist. Before fixed-budget results were observed, adaptive contenders within 3% of the respective primary/chart minimum mean MAE were chosen for confirmation. Each received a fixed iteration budget equal to its median retained trees. Final criterion is mean chronological MAE on these **fixed-budget** models: candidates within 1% of the minimum enter a shortlist ordered by September–October MAE, mean MAE, RMSE, and fold-MAE variability. Final candidates retain coordinates, handle missing measurements and unseen labels, and use realistically supplied fields. RMSE/R2 and instability are reported, not hidden by a single MAE. Source adaptive experiment: `catboost_ablation_F_clean_log1p`; frozen confirmation model: `catboost_fixed_F_clean_log1p`.

Overall model comparison (unweighted fold means, plus primary holdout MAE):

| model | MAE | RMSE | R2 | Sep_Oct_MAE |
| --- | --- | --- | --- | --- |
| global_median | 1150.997962 | 1572.594471 | -0.068519 | 1148.923725716686 |
| distance_linear | 201.231984 | 658.826278 | 0.812447 | 196.95014685582817 |
| ridge | 252.811958 | 687.475618 | 0.795407 | 228.67561180380835 |
| catboost_d7_lr003_l2_3 | 136.353690 | 637.064128 | 0.824627 | 117.95628548608885 |
| catboost_d6_lr005_l2_5 | 129.661646 | 633.980861 | 0.826301 | 115.17521094050014 |
| catboost_d8_lr004_l2_7 | 136.297802 | 637.050283 | 0.824635 | 119.22025359074578 |
| catboost_ablation_A_clean_direct | 147.832460 | 639.745084 | 0.823128 | 125.78080781609356 |
| catboost_ablation_B_clean_direct | 144.021589 | 638.547036 | 0.823788 | 117.15117416418775 |
| catboost_ablation_C_clean_direct | 136.766525 | 636.735147 | 0.824764 | 113.29608026695786 |
| catboost_ablation_D_clean_direct | 138.475611 | 637.121197 | 0.824581 | 110.47187475374803 |
| catboost_ablation_E_clean_direct | 130.469460 | 633.912924 | 0.826323 | 116.65574570611177 |
| catboost_ablation_F_clean_direct | 129.661646 | 633.980861 | 0.826301 | 115.17521094050014 |
| catboost_ablation_C_raw_direct | 137.110003 | 636.088447 | 0.825129 | 116.96066477218562 |
| catboost_ablation_F_raw_direct | 130.151436 | 635.338031 | 0.825561 | 123.85278543504938 |
| catboost_ablation_C_clean_log1p | 141.406184 | 637.806966 | 0.824190 | 123.13071566489248 |
| catboost_ablation_F_clean_log1p | 127.877247 | 634.503153 | 0.826025 | 116.06764637030437 |
| catboost_fixed_C_clean_direct | 143.849955 | 637.140984 | 0.824534 | 113.66471948293021 |
| catboost_fixed_C_raw_direct | 144.109937 | 636.938452 | 0.824654 | 116.96066477218562 |
| catboost_fixed_D_clean_direct | 147.605337 | 638.593691 | 0.823747 | 110.59383875330292 |
| catboost_fixed_E_clean_direct | 135.778982 | 634.555627 | 0.825962 | 116.65574570611177 |
| catboost_fixed_F_clean_direct | 130.996857 | 634.462983 | 0.826030 | 115.29372038885386 |
| catboost_fixed_F_clean_log1p | 129.408586 | 634.963714 | 0.825767 | 116.70764912764066 |
| catboost_fixed_F_raw_direct | 131.515007 | 635.578035 | 0.825428 | 124.74889845689927 |

Eligible feature/weight/target candidates:

| model | MAE | RMSE | R2 | MAE_std | feature_group | weight_representation | target_strategy | recent_MAE | recent_RMSE | recent_R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| catboost_ablation_A_clean_direct | 147.832460 | 639.745084 | 0.823128 | 19.784350477250527 | A | clean | direct | 125.78080781609356 | 638.512540249539 | 0.8249347349097955 |
| catboost_ablation_B_clean_direct | 144.021589 | 638.547036 | 0.823788 | 24.456548846310245 | B | clean | direct | 117.15117416418775 | 637.5292103534495 | 0.8254735320114723 |
| catboost_ablation_C_clean_direct | 136.766525 | 636.735147 | 0.824764 | 21.539416716064927 | C | clean | direct | 113.29608026695786 | 635.2869178230682 | 0.8266990482308275 |
| catboost_ablation_D_clean_direct | 138.475611 | 637.121197 | 0.824581 | 24.252066481952603 | D | clean | direct | 110.47187475374803 | 634.6898815367465 | 0.8270246281700762 |
| catboost_ablation_E_clean_direct | 130.469460 | 633.912924 | 0.826323 | 14.26641969305784 | E | clean | direct | 116.65574570611177 | 636.6830081120578 | 0.825936527820693 |
| catboost_ablation_F_clean_direct | 129.661646 | 633.980861 | 0.826301 | 13.082052090870604 | F | clean | direct | 115.17521094050014 | 635.7867052383148 | 0.8264262653282447 |
| catboost_ablation_C_raw_direct | 137.110003 | 636.088447 | 0.825129 | 18.632932098226007 | C | raw | direct | 116.96066477218562 | 636.9259130117804 | 0.8258036864062518 |
| catboost_ablation_F_raw_direct | 130.151436 | 635.338031 | 0.825561 | 9.155720645775352 | F | raw | direct | 123.85278543504938 | 639.3656881448693 | 0.8244665958072811 |
| catboost_ablation_C_clean_log1p | 141.406184 | 637.806966 | 0.824190 | 16.359211671761717 | C | clean | log1p | 123.13071566489248 | 639.2070750581938 | 0.8245536772446419 |
| catboost_ablation_F_clean_log1p | 127.877247 | 634.503153 | 0.826025 | 13.478710163120711 | F | clean | log1p | 116.06764637030437 | 636.3296470681424 | 0.8261296857378859 |

This configuration wins the stated fixed-budget development-only shortlist rule within the strongest model family. Earlier baselines establish the comparison floor; feature ablation and direct/log dollar-scale comparison support its representation. The early-stopping winner need not be the fixed-budget winner, because validation-adaptive tree counts cannot be used for production without future targets. All native CatBoost fits use fold-local categorical statistics; no global target encoding or learned preprocessing crosses fold boundaries. No final-validation row values, labels, distribution summaries, or prediction outcomes were consulted to choose the model.

For the fixed full-feature comparison, clean/log MAE averages 129.408586 versus 130.996857 for clean/direct (1.212% lower). Its mean RMSE is 634.963714 versus 634.462983, and September–October MAE is 116.707649 versus 115.293720. The log target wins mean MAE but slightly worsens recent MAE and squared-error metrics; this is a modest tradeoff, not a win on every metric. Full-feature raw/direct fixed-budget mean MAE is 131.515007. The cleaned representation preserves original missing/negative flags and performs better on the recent holdout.

The complete F bundle was the tested candidate, rather than an assertion that every component helps individually. The cumulative route addition worsened mean direct MAE before adding market signals; its marginal contribution conditional on the final log/market bundle was not separately isolated. quote_signal has mixed per-fold gains and uncertain upstream timing. The selected bundle wins among executed configurations, while these feature-specific limitations remain explicit.

## Exact final feature set and parameters

Feature group: **F**. Exact ordered 26-feature list:

`distance, equipment, weight_clean, weight_missing, weight_negative, pickup, delivery, pickup_lat, pickup_lon, delivery_lat, delivery_lon, route, month, day, day_of_week, day_of_year, iso_week, weekend, days_since_reference_date, sin_day_of_week, cos_day_of_week, sin_day_of_year, cos_day_of_year, market_index, market_index_missing, quote_signal`

Categorical features: `equipment, pickup, delivery, route`. Target strategy: **log1p**. Reference date: **2025-01-01**. Both training and inference must use `build_model_features` with this frozen feature list. load_id and posted_rate are excluded. Missing categoricals are explicit strings; numeric NaN is native. The selected weight representation is intentional, not automatic row deletion or undocumented correction.

```json
{
  "loss_function": "MAE",
  "iterations": 399,
  "random_seed": 42,
  "verbose": false,
  "thread_count": 4,
  "allow_writing_files": false,
  "nan_mode": "Min",
  "depth": 6,
  "learning_rate": 0.05,
  "l2_leaf_reg": 5
}
```

Production fit must use **all 48,000 labeled rows with no eval_set and no early stopping**. The iteration count was fixed from the median of the three source model's retained tree counts; no unavailable future validation target is needed. Do not select a new iteration count during production. Direct predictions stay in dollars; log1p predictions use expm1. Frozen output policy: reject nonfinite outputs and enforce a $0.01 lower bound. That lower bound did not change any selected-model local evaluation or robustness prediction; original unmodified predictions were already positive.

Source adaptive-budget selection results:

| model | fold | MAE | RMSE | R2 | tree_count |
| --- | --- | --- | --- | --- | --- |
| catboost_ablation_F_clean_log1p | fold_1 | 142.561320 | 644.491670 | 0.826451 | 399 |
| catboost_ablation_F_clean_log1p | fold_2 | 125.002776 | 622.688141 | 0.825494 | 116 |
| catboost_ablation_F_clean_log1p | fold_3 | 116.067646 | 636.329647 | 0.826130 | 558 |

Actual frozen-budget confirmation results (trained on each historical training fold without eval_set):

| model | fold | MAE | RMSE | R2 | tree_count |
| --- | --- | --- | --- | --- | --- |
| catboost_fixed_F_clean_log1p | fold_1 | 142.561320 | 644.491670 | 0.826451 | 399 |
| catboost_fixed_F_clean_log1p | fold_2 | 128.956789 | 623.894968 | 0.824817 | 399 |
| catboost_fixed_F_clean_log1p | fold_3 | 116.707649 | 636.504505 | 0.826034 | 399 |

Unweighted mean fixed-budget MAE/RMSE/R2: **129.408586 / 634.963714 / 0.825767**. September–October fixed-budget MAE/RMSE/R2: **116.707649 / 636.504505 / 0.826034**. These actual fixed-budget results are the recommended local metrics for subsequent training metadata and the report; source stopping-selected scores must be labeled separately.

## Missingness and unseen-category robustness

Actual local confirmation-subset metrics and coverage:

| model | fold | subset | rows | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- | --- | --- |
| catboost_fixed_F_clean_log1p | fold_1 | missing_weight | 54 | 165.017224 | 317.899354 | 0.945703 |
| catboost_fixed_F_clean_log1p | fold_1 | negative_weight | 70 | 336.824934 | 1422.260135 | 0.449894 |
| catboost_fixed_F_clean_log1p | fold_1 | missing_market_index | 85 | 143.518564 | 357.170916 | 0.935497 |
| catboost_fixed_F_clean_log1p | fold_1 | unseen_route | 177 | 177.305367 | 647.638439 | 0.797260 |
| catboost_fixed_F_clean_log1p | fold_1 | unseen_city | 0 | nan | nan | nan |
| catboost_fixed_F_clean_log1p | fold_2 | missing_weight | 58 | 112.868542 | 290.354699 | 0.954201 |
| catboost_fixed_F_clean_log1p | fold_2 | negative_weight | 37 | 70.254152 | 103.047746 | 0.994901 |
| catboost_fixed_F_clean_log1p | fold_2 | missing_market_index | 69 | 203.515069 | 758.064560 | 0.755980 |
| catboost_fixed_F_clean_log1p | fold_2 | unseen_route | 52 | 78.658140 | 113.587210 | 0.994455 |
| catboost_fixed_F_clean_log1p | fold_2 | unseen_city | 0 | nan | nan | nan |
| catboost_fixed_F_clean_log1p | fold_3 | missing_weight | 65 | 101.178370 | 138.112190 | 0.990585 |
| catboost_fixed_F_clean_log1p | fold_3 | negative_weight | 59 | 113.834269 | 331.744855 | 0.951378 |
| catboost_fixed_F_clean_log1p | fold_3 | missing_market_index | 73 | 90.450440 | 298.629196 | 0.960201 |
| catboost_fixed_F_clean_log1p | fold_3 | unseen_route | 21 | 60.102357 | 76.644527 | 0.997095 |
| catboost_fixed_F_clean_log1p | fold_3 | unseen_city | 0 | nan | nan | nan |

Some chronological folds have no entirely new cities. Their zero-count subsets have no estimable accuracy; this does not demonstrate new-city accuracy. New-route subsets can contain familiar endpoint labels. Synthetic mechanical checks alter 128 September–October rows per case while preserving coordinates for new-name tests:

| case | rows | finite_positive |
| --- | --- | --- |
| unseen_cities_routes_equipment | 128 | True |
| missing_weight | 128 | True |
| negative_weight | 128 | True |
| unavailable_market_and_quote | 128 | True |
| missing_categories_and_coordinates | 128 | True |

All five cases produced finite strictly positive predictions using the selected full-size local holdout model. No accuracy is claimed for edited scenarios. Native CatBoost accepts unknown labels; sklearn's retained baseline preprocessing uses handle_unknown="ignore". Numeric latitude/longitude remain in the final model for geographic information beyond labels. No final unlabeled inference rows were used for these selection-stage checks.

## December inference configuration

Frozen December group: **C**, weight **clean**, target **direct**, iterations **541**. Its source adaptive experiment is `catboost_ablation_C_clean_direct`; frozen confirmation is `catboost_fixed_C_clean_direct`. Exact features: `distance, equipment, weight_clean, weight_missing, weight_negative, pickup, delivery, pickup_lat, pickup_lon, delivery_lat, delivery_lon, month, day, day_of_week, day_of_year, iso_week, weekend, days_since_reference_date, sin_day_of_week, cos_day_of_week, sin_day_of_year, cos_day_of_year`.

```json
{
  "loss_function": "MAE",
  "iterations": 541,
  "random_seed": 42,
  "verbose": false,
  "thread_count": 4,
  "allow_writing_files": false,
  "nan_mode": "Min",
  "depth": 6,
  "learning_rate": 0.05,
  "l2_leaf_reg": 5
}
```

The chart configuration is restricted to freight/load/calendar/city/coordinate information. It does not require market_index or quote_signal. Lookup Lexington/Fort Wayne coordinates deterministically from the supplied development city mapping when the eventual chart pipeline is built; no future market or quote values may be invented. If primary and chart features/target match, the same final model can be reused; otherwise train the separately frozen chart configuration at the future production stage. Its development confirmation results are:

| model | fold | MAE | RMSE | R2 | tree_count |
| --- | --- | --- | --- | --- | --- |
| catboost_fixed_C_clean_direct | fold_1 | 141.374326 | 644.968913 | 0.826193 | 541 |
| catboost_fixed_C_clean_direct | fold_2 | 176.510820 | 631.141887 | 0.820723 | 541 |
| catboost_fixed_C_clean_direct | fold_3 | 113.664719 | 635.312153 | 0.826685 | 541 |

The fixed December file changes date while keeping lane/load characteristics fixed; these local freight-fold metrics do not measure fixed-lane December accuracy.

## Important limitations

- Stopping and sequential model selection reuse the three chronological validation windows. Fixed-budget confirmation does not restore an untouched test. Local scores may be optimistic and must not be called Spotter hidden performance.
- Labeled data stops in October; November/December target regimes and entirely new-city accuracy remain unknown. One partial year cannot establish stable annual seasonality.
- Market/quote fields are supplied in the documented assessment inference schema, but upstream provenance, units, and posting-time availability are unspecified. Their usefulness does not prove leakage or availability in a real deployment. December uses the separately validated available-feature configuration.
- The target has a long positive tail; all rates, including the largest, remain in the experiments. MAE-oriented fitting can improve typical errors while leaving large-error RMSE substantial.
- Group/weight/target search is sequential and modest; interactions across every possible combination were not exhaustively tested. Joint hyperparameter alternatives do not isolate causal effects.
- Coordinates are supplied internally consistent values, not externally verified geocodes. Compatibility with unseen categories establishes successful prediction, not geographic accuracy.
- Production model fitting, all 12,000 final predictions, scorer execution, final report and Loom remain later deliverables. This checkpoint deliberately stops before production training.

## Reproducibility

Run `.venv/bin/python scripts/run_model_experiments.py --phase all`, then `.venv/bin/python scripts/report_model_selection.py`, then `.venv/bin/python -m pytest -q`. The initial baseline/hyperparameter command is `.venv/bin/python -m src.evaluate --models all`. Production should consume the frozen JSON and must not automatically rerun these comparison experiments. Seeds and explicit parameters are fixed; original supplied files remain unchanged.
