# Controlled feature, weight, and target experiments

Reproduce experiments with `.venv/bin/python scripts/run_model_experiments.py --phase all`; regenerate this report with `.venv/bin/python scripts/report_model_selection.py`. Only labeled development rows are loaded. The fixed folds are January–April/May–June, January–June/July–August, and January–August/September–October (19,110/9,696; 28,806/9,671; 38,477/9,523 rows). No final-validation values are consulted for these comparisons.

## Controlled feature ablation

Use the strongest previously tested family/configuration: CatBoost, depth=6, learning_rate=0.05, l2_leaf_reg=5, MAE loss, random_seed=42, 1,500-iteration cap, and 100-round patience. Each ablation uses the same folds, canonical column order, and training-only native categorical statistics. Calendar includes the seven ordinary calendar features and four cyclic features. Cleaned weight includes abs(weight), original missingness, and original negative-sign flags. E includes both market_index and its missing flag.

| group | count | features |
| --- | --- | --- |
| A | 16 | distance, equipment, weight_clean, weight_missing, weight_negative, month, day, day_of_week, day_of_year, iso_week, weekend, days_since_reference_date, sin_day_of_week, cos_day_of_week, sin_day_of_year, cos_day_of_year |
| B | 18 | distance, equipment, weight_clean, weight_missing, weight_negative, pickup, delivery, month, day, day_of_week, day_of_year, iso_week, weekend, days_since_reference_date, sin_day_of_week, cos_day_of_week, sin_day_of_year, cos_day_of_year |
| C | 22 | distance, equipment, weight_clean, weight_missing, weight_negative, pickup, delivery, pickup_lat, pickup_lon, delivery_lat, delivery_lon, month, day, day_of_week, day_of_year, iso_week, weekend, days_since_reference_date, sin_day_of_week, cos_day_of_week, sin_day_of_year, cos_day_of_year |
| D | 23 | distance, equipment, weight_clean, weight_missing, weight_negative, pickup, delivery, pickup_lat, pickup_lon, delivery_lat, delivery_lon, route, month, day, day_of_week, day_of_year, iso_week, weekend, days_since_reference_date, sin_day_of_week, cos_day_of_week, sin_day_of_year, cos_day_of_year |
| E | 25 | distance, equipment, weight_clean, weight_missing, weight_negative, pickup, delivery, pickup_lat, pickup_lon, delivery_lat, delivery_lon, route, month, day, day_of_week, day_of_year, iso_week, weekend, days_since_reference_date, sin_day_of_week, cos_day_of_week, sin_day_of_year, cos_day_of_year, market_index, market_index_missing |
| F | 26 | distance, equipment, weight_clean, weight_missing, weight_negative, pickup, delivery, pickup_lat, pickup_lon, delivery_lat, delivery_lon, route, month, day, day_of_week, day_of_year, iso_week, weekend, days_since_reference_date, sin_day_of_week, cos_day_of_week, sin_day_of_year, cos_day_of_year, market_index, market_index_missing, quote_signal |

Unweighted mean fold metrics:

| feature_group | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| A | 147.832460 | 639.745084 | 0.823128 |
| B | 144.021589 | 638.547036 | 0.823788 |
| C | 136.766525 | 636.735147 | 0.824764 |
| D | 138.475611 | 637.121197 | 0.824581 |
| E | 130.469460 | 633.912924 | 0.826323 |
| F | 129.661646 | 633.980861 | 0.826301 |

Per-fold metrics and retained iterations:

| model | fold | MAE | RMSE | R2 | best_iteration | tree_count |
| --- | --- | --- | --- | --- | --- | --- |
| catboost_ablation_A_clean_direct | fold_1 | 153.689727 | 650.684442 | 0.823099 | 508 | 509 |
| catboost_ablation_A_clean_direct | fold_2 | 164.026845 | 630.038269 | 0.821350 | 89 | 90 |
| catboost_ablation_A_clean_direct | fold_3 | 125.780808 | 638.512540 | 0.824935 | 288 | 289 |
| catboost_ablation_B_clean_direct | fold_1 | 149.932943 | 649.084891 | 0.823968 | 905 | 906 |
| catboost_ablation_B_clean_direct | fold_2 | 164.980651 | 629.027007 | 0.821923 | 101 | 102 |
| catboost_ablation_B_clean_direct | fold_3 | 117.151174 | 637.529210 | 0.825474 | 1496 | 1497 |
| catboost_ablation_C_clean_direct | fold_1 | 141.374326 | 644.968913 | 0.826193 | 540 | 541 |
| catboost_ablation_C_clean_direct | fold_2 | 155.629169 | 629.949610 | 0.821400 | 83 | 84 |
| catboost_ablation_C_clean_direct | fold_3 | 113.296080 | 635.286918 | 0.826699 | 759 | 760 |
| catboost_ablation_D_clean_direct | fold_1 | 152.553674 | 649.674286 | 0.823648 | 896 | 897 |
| catboost_ablation_D_clean_direct | fold_2 | 152.401283 | 626.999422 | 0.823069 | 93 | 94 |
| catboost_ablation_D_clean_direct | fold_3 | 110.471875 | 634.689882 | 0.827025 | 925 | 926 |
| catboost_ablation_E_clean_direct | fold_1 | 129.603500 | 640.213171 | 0.828747 | 950 | 951 |
| catboost_ablation_E_clean_direct | fold_2 | 145.149135 | 624.842593 | 0.824284 | 104 | 105 |
| catboost_ablation_E_clean_direct | fold_3 | 116.655746 | 636.683008 | 0.825937 | 709 | 710 |
| catboost_ablation_F_clean_direct | fold_1 | 140.612623 | 642.631102 | 0.827451 | 283 | 284 |
| catboost_ablation_F_clean_direct | fold_2 | 133.197105 | 623.524775 | 0.825025 | 161 | 162 |
| catboost_ablation_F_clean_direct | fold_3 | 115.175211 | 635.786705 | 0.826426 | 291 | 292 |

The F result reuses the identical prior depth-6/lr0.05/l2=5 full-feature experiment, including its metrics and timing. Five other groups required 15 new fits. All 48,000 rows are retained in the source; nothing is removed for missingness, negative weights, or target outliers. Coordinates remain mandatory in eligible final groups C–F; A/B are comparison baselines.

## Weight representation

Raw weight means the original signed value, preserving numeric NaN with native handling. Clean means abs(weight) plus weight_missing and weight_negative. Other feature groups and parameters remain identical. This is a comparison of complete representations; it does not isolate absolute-value correction from the contribution of the two flags. The strongest full-inference and December-feasible direct feature groups were chosen before these representation tests; the raw alternative is evaluated on both where different.

| model | MAE | RMSE | R2 | MAE_std | feature_group | weight_representation | target_strategy | recent_MAE | recent_RMSE | recent_R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| catboost_ablation_C_clean_direct | 136.766525 | 636.735147 | 0.824764 | 21.539416716064927 | C | clean | direct | 113.29608026695786 | 635.2869178230682 | 0.8266990482308275 |
| catboost_ablation_F_clean_direct | 129.661646 | 633.980861 | 0.826301 | 13.082052090870604 | F | clean | direct | 115.17521094050014 | 635.7867052383148 | 0.8264262653282447 |
| catboost_ablation_C_raw_direct | 137.110003 | 636.088447 | 0.825129 | 18.632932098226007 | C | raw | direct | 116.96066477218562 | 636.9259130117804 | 0.8258036864062518 |
| catboost_ablation_F_raw_direct | 130.151436 | 635.338031 | 0.825561 | 9.155720645775352 | F | raw | direct | 123.85278543504938 | 639.3656881448693 | 0.8244665958072811 |

| model | fold | MAE | RMSE | R2 | best_iteration | tree_count |
| --- | --- | --- | --- | --- | --- | --- |
| catboost_ablation_C_clean_direct | fold_1 | 141.374326 | 644.968913 | 0.826193 | 540 | 541 |
| catboost_ablation_C_clean_direct | fold_2 | 155.629169 | 629.949610 | 0.821400 | 83 | 84 |
| catboost_ablation_C_clean_direct | fold_3 | 113.296080 | 635.286918 | 0.826699 | 759 | 760 |
| catboost_ablation_F_clean_direct | fold_1 | 140.612623 | 642.631102 | 0.827451 | 283 | 284 |
| catboost_ablation_F_clean_direct | fold_2 | 133.197105 | 623.524775 | 0.825025 | 161 | 162 |
| catboost_ablation_F_clean_direct | fold_3 | 115.175211 | 635.786705 | 0.826426 | 291 | 292 |
| catboost_ablation_C_raw_direct | fold_1 | 140.650968 | 643.686038 | 0.826884 | 1160 | 1161 |
| catboost_ablation_C_raw_direct | fold_2 | 153.718376 | 627.653388 | 0.822700 | 94 | 95 |
| catboost_ablation_C_raw_direct | fold_3 | 116.960665 | 636.925913 | 0.825804 | 412 | 413 |
| catboost_ablation_F_raw_direct | fold_1 | 140.654160 | 643.299146 | 0.827092 | 204 | 205 |
| catboost_ablation_F_raw_direct | fold_2 | 125.947363 | 623.349259 | 0.825123 | 129 | 130 |
| catboost_ablation_F_raw_direct | fold_3 | 123.852785 | 639.365688 | 0.824467 | 260 | 261 |

Negative signs are consistent with the earlier sign-corruption investigation, not a proven correction. The winning representation is chosen by chronological performance and robustness, not that hypothesis alone. Numeric NaN is never filled from holdout/final inputs.

## Market feature quality and availability

Development-only overall summaries:

| feature | missing | missing_pct | mean | std | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| market_index | 374 | 0.7791666666666667 | 1.0833867156595136 | 0.1680909157944615 | 0.67639 | 1.46778 |
| quote_signal | 0 | 0.0 | 2.0624678412499997 | 0.29139139082151877 | 0.69228 | 3.61035 |

Monthly development-only summaries (full precision in `reports/market_feature_monthly.csv`):

| month | feature | missing | missing_pct | mean | std | median |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-01 | market_index | 36 | 0.7320048800325335 | 0.9266961122490781 | 0.07786807041218738 | 0.9329050000000001 |
| 2025-01 | quote_signal | 0 | 0.0 | 2.068436260675071 | 0.2503346686075269 | 2.028715 |
| 2025-02 | market_index | 36 | 0.830066866497579 | 0.9959967193675888 | 0.08337498014812396 | 0.99313 |
| 2025-02 | quote_signal | 0 | 0.0 | 2.0971300230574133 | 0.2516785837475214 | 2.0604 |
| 2025-03 | market_index | 44 | 0.8737092930897538 | 1.0728797656249998 | 0.08025831467417545 | 1.06485 |
| 2025-03 | quote_signal | 0 | 0.0 | 2.179538661636219 | 0.2729023945618156 | 2.13958 |
| 2025-04 | market_index | 31 | 0.6432869890018675 | 1.2076491478696743 | 0.1002180387031018 | 1.207125 |
| 2025-04 | quote_signal | 0 | 0.0 | 1.9633883523552604 | 0.2604236990897642 | 2.00233 |
| 2025-05 | market_index | 43 | 0.875228984327295 | 1.3014521519507187 | 0.07416314902656278 | 1.304025 |
| 2025-05 | quote_signal | 0 | 0.0 | 1.9128285914919598 | 0.2658293622657984 | 1.95692 |
| 2025-06 | market_index | 42 | 0.8781099728204057 | 1.280315676017718 | 0.076005884738246 | 1.27513 |
| 2025-06 | quote_signal | 0 | 0.0 | 2.2960069307965716 | 0.28888859260861455 | 2.24699 |
| 2025-07 | market_index | 36 | 0.7328990228013029 | 1.2008926579163248 | 0.11603645245317804 | 1.2121 |
| 2025-07 | quote_signal | 0 | 0.0 | 1.922996532980456 | 0.26514677685795585 | 1.96338 |
| 2025-08 | market_index | 33 | 0.6934229880226939 | 0.9814802052475666 | 0.07795966471826517 | 0.98234 |
| 2025-08 | quote_signal | 0 | 0.0 | 2.0521998108846398 | 0.2226623767233182 | 2.05223 |
| 2025-09 | market_index | 31 | 0.6638115631691649 | 0.893038579435223 | 0.07977071664065198 | 0.88781 |
| 2025-09 | quote_signal | 0 | 0.0 | 2.1997849914346896 | 0.2798644015315704 | 2.15975 |
| 2025-10 | market_index | 42 | 0.865444055223573 | 0.9575017418416129 | 0.07138990321436947 | 0.95818 |
| 2025-10 | quote_signal | 0 | 0.0 | 1.942758493715228 | 0.2692398413926412 | 1.98584 |

Development market_index monthly means range from 0.893039 to 1.301452; quote_signal monthly means range from 1.912829 to 2.296007. This shows calendar-associated distribution changes, not guaranteed repeatable annual seasonality. Missingness is tracked monthly; no missing observation is silently deleted.

Paired chronological feature additions (new minus old metric; negative MAE/RMSE is better, positive R2 is better):

| change | fold | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- |
| Add market_index + missing flag | fold_1 | -22.950174 | -9.461116 | 0.005099 |
| Add market_index + missing flag | fold_2 | -7.252148 | -2.156828 | 0.001215 |
| Add market_index + missing flag | fold_3 | 6.183871 | 1.993127 | -0.001088 |
| Add quote_signal | fold_1 | 11.009123 | 2.417931 | -0.001296 |
| Add quote_signal | fold_2 | -11.952030 | -1.317819 | 0.000740 |
| Add quote_signal | fold_3 | -1.480535 | -0.896303 | 0.000490 |

Gains vary by fold; pooled marginal correlations do not decide usefulness. Identical folds/configurations isolate the addition's predictive effect, although adaptive stopping permits different retained tree counts. quote_signal's contribution must be assessed alongside its mixed fold behavior.

The earlier schema audit established that the assessment's supplied final-inference inputs include market_index and quote_signal. That is a schema-level availability assumption, not a source of model-selection row values. Both fields are absent from the fixed December scenario. The supplied instructions do not define units, provenance, publication lag, or when these signals become available relative to posting a rate. No executed analysis proves leakage, so neither is called leakage. A real deployment must verify timestamp alignment and availability before using them. For this assessment, provided inference values may be consumed; unavailable scenario values must never be forecast or invented. Native NaN handling and a separately validated feature-restricted December configuration address availability.

## Direct versus log1p target

The selected direct-model feature/weight choices for full inference and December form the controlled target-comparison scopes. The sequential experiment is intentionally limited, not a factorial search across every feature/weight/target interaction. MAE loss in log space changes the fitting emphasis; no smearing correction or post-hoc target calibration is learned. Predictions are transformed back with expm1. **Log models stop using custom DollarMAE**, which also transforms the evaluation truth/predictions back to dollars; stopping does not accidentally optimize relative/log-space error. Every reported MAE/RMSE/R2 is on the original posted-rate scale.

| model | MAE | RMSE | R2 | MAE_std | feature_group | weight_representation | target_strategy | recent_MAE | recent_RMSE | recent_R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| catboost_ablation_C_clean_direct | 136.766525 | 636.735147 | 0.824764 | 21.539416716064927 | C | clean | direct | 113.29608026695786 | 635.2869178230682 | 0.8266990482308275 |
| catboost_ablation_C_clean_log1p | 141.406184 | 637.806966 | 0.824190 | 16.359211671761717 | C | clean | log1p | 123.13071566489248 | 639.2070750581938 | 0.8245536772446419 |
| catboost_ablation_F_clean_direct | 129.661646 | 633.980861 | 0.826301 | 13.082052090870604 | F | clean | direct | 115.17521094050014 | 635.7867052383148 | 0.8264262653282447 |
| catboost_ablation_F_clean_log1p | 127.877247 | 634.503153 | 0.826025 | 13.478710163120711 | F | clean | log1p | 116.06764637030437 | 636.3296470681424 | 0.8261296857378859 |

| model | fold | MAE | RMSE | R2 | best_iteration | tree_count |
| --- | --- | --- | --- | --- | --- | --- |
| catboost_ablation_C_clean_direct | fold_1 | 141.374326 | 644.968913 | 0.826193 | 540 | 541 |
| catboost_ablation_C_clean_direct | fold_2 | 155.629169 | 629.949610 | 0.821400 | 83 | 84 |
| catboost_ablation_C_clean_direct | fold_3 | 113.296080 | 635.286918 | 0.826699 | 759 | 760 |
| catboost_ablation_C_clean_log1p | fold_1 | 146.405171 | 645.810708 | 0.825739 | 611 | 612 |
| catboost_ablation_C_clean_log1p | fold_2 | 154.682667 | 628.403114 | 0.822276 | 94 | 95 |
| catboost_ablation_C_clean_log1p | fold_3 | 123.130716 | 639.207075 | 0.824554 | 746 | 747 |
| catboost_ablation_F_clean_direct | fold_1 | 140.612623 | 642.631102 | 0.827451 | 283 | 284 |
| catboost_ablation_F_clean_direct | fold_2 | 133.197105 | 623.524775 | 0.825025 | 161 | 162 |
| catboost_ablation_F_clean_direct | fold_3 | 115.175211 | 635.786705 | 0.826426 | 291 | 292 |
| catboost_ablation_F_clean_log1p | fold_1 | 142.561320 | 644.491670 | 0.826451 | 398 | 399 |
| catboost_ablation_F_clean_log1p | fold_2 | 125.002776 | 622.688141 | 0.825494 | 115 | 116 |
| catboost_ablation_F_clean_log1p | fold_3 | 116.067646 | 636.329647 | 0.826130 | 557 | 558 |

Development target summary (all observations retained):

| statistic | posted_rate |
| --- | --- |
| count | 48000.0 |
| mean | 2373.9806822916667 |
| std | 1486.4932447622946 |
| min | 57.22 |
| 1% | 327.16880000000003 |
| 5% | 599.7385 |
| 50% | 2030.76 |
| 95% | 4953.7665 |
| 99% | 5972.834000000003 |
| max | 25533.0 |

The previously generated target histogram and rate-per-mile figures are `reports/figures/eda_target_distribution.png` and `reports/figures/eda_rate_per_mile.png`. Large rates remain in all training and validation folds. RMSE remains sensitive to large errors; a log target is retained only if executed chronological metrics support it.

## Selection and methodological limits

Intermediate feature/weight scopes use an unweighted mean-MAE shortlist within 1% of the minimum, emphasizing September–October MAE, then mean MAE, RMSE, and MAE variability. Before final confirmation fits, shortlist adaptive primary and December-feasible contenders within 3% of their respective minimum mean MAE. For each finalist, derive a fixed iteration budget from its median retained tree count and evaluate that exact budget on all three folds without an eval_set. Choose among fixed-budget models using the 1% mean-MAE shortlist and the same September–October/MAE/RMSE/stability ordering. This separates the adaptive stopping winner from the configuration actually deployable without future labels. Missing-value/unseen-category compatibility and supplied-field availability are required. The frozen decision and actual fixed-budget metrics are in `docs/model_selection.md`; machine-readable settings are in `docs/final_model_config.json`.

Adaptive stopping, sequential feature/representation choice, and model selection reuse these historical validation windows. Fixed-budget confirmation removes evaluation labels from each fit but does not undo prior selection reuse. These are local model-selection estimates, not independent test scores or Spotter hidden metrics. Geographic compatibility is not evidence of accuracy for entirely new cities. No production training or final submission output occurs in these scripts.
