# Chronological model comparison

Reproduce the initial baselines/hyperparameters: `.venv/bin/python -m src.evaluate --models all`. Reproduce the controlled feature/weight/target experiments and frozen-budget confirmation: `.venv/bin/python scripts/run_model_experiments.py --phase all`. Regenerate analysis/selection documents: `.venv/bin/python scripts/report_model_selection.py`. Incremental initial comparison options are `--models catboost` and `--models baselines`.

All 69 recorded model/fold results use only `data/train_test.csv` (48,000 labeled rows, January 1–October 31, 2025). The F-clean/direct ablation reuses the identical previously executed depth-6 full-feature result, rather than refitting it. No final-validation features, targets, or inference distributions are used for these experiments. Dates are never shuffled, and entire dates stay together. Fold 3 is the primary September–October local holdout; earlier folds provide additional evidence of temporal stability. See `docs/validation_plan.md`.

## Fold sizes

| fold | train_rows | validation_rows |
| --- | --- | --- |
| fold_1 | 19110 | 9696 |
| fold_2 | 28806 | 9671 |
| fold_3 | 38477 | 9523 |

## Per-fold results

| model | fold | train_start | train_end | validation_start | validation_end | MAE | RMSE | R2 | feature_set |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global_median | fold_1 | 2025-01-01 | 2025-04-30 | 2025-05-01 | 2025-06-30 | 1175.958842 | 1616.170952 | -0.091348 | training_fold_median |
| distance_linear | fold_1 | 2025-01-01 | 2025-04-30 | 2025-05-01 | 2025-06-30 | 217.735480 | 677.418401 | 0.808264 | distance_only |
| ridge | fold_1 | 2025-01-01 | 2025-04-30 | 2025-05-01 | 2025-06-30 | 215.064136 | 684.586324 | 0.804185 | all_26_candidates |
| global_median | fold_2 | 2025-01-01 | 2025-06-30 | 2025-07-01 | 2025-08-31 | 1128.111318 | 1532.188282 | -0.056560 | training_fold_median |
| distance_linear | fold_2 | 2025-01-01 | 2025-06-30 | 2025-07-01 | 2025-08-31 | 189.010326 | 644.644793 | 0.812970 | distance_only |
| ridge | fold_2 | 2025-01-01 | 2025-06-30 | 2025-07-01 | 2025-08-31 | 314.696125 | 702.806712 | 0.777699 | all_26_candidates |
| global_median | fold_3 | 2025-01-01 | 2025-08-31 | 2025-09-01 | 2025-10-31 | 1148.923726 | 1569.424179 | -0.057650 | training_fold_median |
| distance_linear | fold_3 | 2025-01-01 | 2025-08-31 | 2025-09-01 | 2025-10-31 | 196.950147 | 654.415640 | 0.816106 | distance_only |
| ridge | fold_3 | 2025-01-01 | 2025-08-31 | 2025-09-01 | 2025-10-31 | 228.675612 | 675.033817 | 0.804335 | all_26_candidates |
| catboost_d7_lr003_l2_3 | fold_1 | 2025-01-01 | 2025-04-30 | 2025-05-01 | 2025-06-30 | 152.763917 | 650.293272 | 0.823312 | all_26_candidates |
| catboost_d6_lr005_l2_5 | fold_1 | 2025-01-01 | 2025-04-30 | 2025-05-01 | 2025-06-30 | 140.612623 | 642.631102 | 0.827451 | all_26_candidates |
| catboost_d8_lr004_l2_7 | fold_1 | 2025-01-01 | 2025-04-30 | 2025-05-01 | 2025-06-30 | 155.669607 | 649.469151 | 0.823760 | all_26_candidates |
| catboost_d7_lr003_l2_3 | fold_2 | 2025-01-01 | 2025-06-30 | 2025-07-01 | 2025-08-31 | 138.340869 | 624.387230 | 0.824540 | all_26_candidates |
| catboost_d6_lr005_l2_5 | fold_2 | 2025-01-01 | 2025-06-30 | 2025-07-01 | 2025-08-31 | 133.197105 | 623.524775 | 0.825025 | all_26_candidates |
| catboost_d8_lr004_l2_7 | fold_2 | 2025-01-01 | 2025-06-30 | 2025-07-01 | 2025-08-31 | 134.003546 | 624.072239 | 0.824717 | all_26_candidates |
| catboost_d7_lr003_l2_3 | fold_3 | 2025-01-01 | 2025-08-31 | 2025-09-01 | 2025-10-31 | 117.956285 | 636.511881 | 0.826030 | all_26_candidates |
| catboost_d6_lr005_l2_5 | fold_3 | 2025-01-01 | 2025-08-31 | 2025-09-01 | 2025-10-31 | 115.175211 | 635.786705 | 0.826426 | all_26_candidates |
| catboost_d8_lr004_l2_7 | fold_3 | 2025-01-01 | 2025-08-31 | 2025-09-01 | 2025-10-31 | 119.220254 | 637.609460 | 0.825430 | all_26_candidates |
| catboost_ablation_A_clean_direct | fold_1 | 2025-01-01 | 2025-04-30 | 2025-05-01 | 2025-06-30 | 153.689727 | 650.684442 | 0.823099 | ablation_A_clean |
| catboost_ablation_A_clean_direct | fold_2 | 2025-01-01 | 2025-06-30 | 2025-07-01 | 2025-08-31 | 164.026845 | 630.038269 | 0.821350 | ablation_A_clean |
| catboost_ablation_A_clean_direct | fold_3 | 2025-01-01 | 2025-08-31 | 2025-09-01 | 2025-10-31 | 125.780808 | 638.512540 | 0.824935 | ablation_A_clean |
| catboost_ablation_B_clean_direct | fold_1 | 2025-01-01 | 2025-04-30 | 2025-05-01 | 2025-06-30 | 149.932943 | 649.084891 | 0.823968 | ablation_B_clean |
| catboost_ablation_B_clean_direct | fold_2 | 2025-01-01 | 2025-06-30 | 2025-07-01 | 2025-08-31 | 164.980651 | 629.027007 | 0.821923 | ablation_B_clean |
| catboost_ablation_B_clean_direct | fold_3 | 2025-01-01 | 2025-08-31 | 2025-09-01 | 2025-10-31 | 117.151174 | 637.529210 | 0.825474 | ablation_B_clean |
| catboost_ablation_C_clean_direct | fold_1 | 2025-01-01 | 2025-04-30 | 2025-05-01 | 2025-06-30 | 141.374326 | 644.968913 | 0.826193 | ablation_C_clean |
| catboost_ablation_C_clean_direct | fold_2 | 2025-01-01 | 2025-06-30 | 2025-07-01 | 2025-08-31 | 155.629169 | 629.949610 | 0.821400 | ablation_C_clean |
| catboost_ablation_C_clean_direct | fold_3 | 2025-01-01 | 2025-08-31 | 2025-09-01 | 2025-10-31 | 113.296080 | 635.286918 | 0.826699 | ablation_C_clean |
| catboost_ablation_D_clean_direct | fold_1 | 2025-01-01 | 2025-04-30 | 2025-05-01 | 2025-06-30 | 152.553674 | 649.674286 | 0.823648 | ablation_D_clean |
| catboost_ablation_D_clean_direct | fold_2 | 2025-01-01 | 2025-06-30 | 2025-07-01 | 2025-08-31 | 152.401283 | 626.999422 | 0.823069 | ablation_D_clean |
| catboost_ablation_D_clean_direct | fold_3 | 2025-01-01 | 2025-08-31 | 2025-09-01 | 2025-10-31 | 110.471875 | 634.689882 | 0.827025 | ablation_D_clean |
| catboost_ablation_E_clean_direct | fold_1 | 2025-01-01 | 2025-04-30 | 2025-05-01 | 2025-06-30 | 129.603500 | 640.213171 | 0.828747 | ablation_E_clean |
| catboost_ablation_E_clean_direct | fold_2 | 2025-01-01 | 2025-06-30 | 2025-07-01 | 2025-08-31 | 145.149135 | 624.842593 | 0.824284 | ablation_E_clean |
| catboost_ablation_E_clean_direct | fold_3 | 2025-01-01 | 2025-08-31 | 2025-09-01 | 2025-10-31 | 116.655746 | 636.683008 | 0.825937 | ablation_E_clean |
| catboost_ablation_F_clean_direct | fold_1 | 2025-01-01 | 2025-04-30 | 2025-05-01 | 2025-06-30 | 140.612623 | 642.631102 | 0.827451 | ablation_F_clean |
| catboost_ablation_F_clean_direct | fold_2 | 2025-01-01 | 2025-06-30 | 2025-07-01 | 2025-08-31 | 133.197105 | 623.524775 | 0.825025 | ablation_F_clean |
| catboost_ablation_F_clean_direct | fold_3 | 2025-01-01 | 2025-08-31 | 2025-09-01 | 2025-10-31 | 115.175211 | 635.786705 | 0.826426 | ablation_F_clean |
| catboost_ablation_C_raw_direct | fold_1 | 2025-01-01 | 2025-04-30 | 2025-05-01 | 2025-06-30 | 140.650968 | 643.686038 | 0.826884 | ablation_C_raw |
| catboost_ablation_C_raw_direct | fold_2 | 2025-01-01 | 2025-06-30 | 2025-07-01 | 2025-08-31 | 153.718376 | 627.653388 | 0.822700 | ablation_C_raw |
| catboost_ablation_C_raw_direct | fold_3 | 2025-01-01 | 2025-08-31 | 2025-09-01 | 2025-10-31 | 116.960665 | 636.925913 | 0.825804 | ablation_C_raw |
| catboost_ablation_F_raw_direct | fold_1 | 2025-01-01 | 2025-04-30 | 2025-05-01 | 2025-06-30 | 140.654160 | 643.299146 | 0.827092 | ablation_F_raw |
| catboost_ablation_F_raw_direct | fold_2 | 2025-01-01 | 2025-06-30 | 2025-07-01 | 2025-08-31 | 125.947363 | 623.349259 | 0.825123 | ablation_F_raw |
| catboost_ablation_F_raw_direct | fold_3 | 2025-01-01 | 2025-08-31 | 2025-09-01 | 2025-10-31 | 123.852785 | 639.365688 | 0.824467 | ablation_F_raw |
| catboost_ablation_C_clean_log1p | fold_1 | 2025-01-01 | 2025-04-30 | 2025-05-01 | 2025-06-30 | 146.405171 | 645.810708 | 0.825739 | ablation_C_clean |
| catboost_ablation_C_clean_log1p | fold_2 | 2025-01-01 | 2025-06-30 | 2025-07-01 | 2025-08-31 | 154.682667 | 628.403114 | 0.822276 | ablation_C_clean |
| catboost_ablation_C_clean_log1p | fold_3 | 2025-01-01 | 2025-08-31 | 2025-09-01 | 2025-10-31 | 123.130716 | 639.207075 | 0.824554 | ablation_C_clean |
| catboost_ablation_F_clean_log1p | fold_1 | 2025-01-01 | 2025-04-30 | 2025-05-01 | 2025-06-30 | 142.561320 | 644.491670 | 0.826451 | ablation_F_clean |
| catboost_ablation_F_clean_log1p | fold_2 | 2025-01-01 | 2025-06-30 | 2025-07-01 | 2025-08-31 | 125.002776 | 622.688141 | 0.825494 | ablation_F_clean |
| catboost_ablation_F_clean_log1p | fold_3 | 2025-01-01 | 2025-08-31 | 2025-09-01 | 2025-10-31 | 116.067646 | 636.329647 | 0.826130 | ablation_F_clean |
| catboost_fixed_C_clean_direct | fold_1 | 2025-01-01 | 2025-04-30 | 2025-05-01 | 2025-06-30 | 141.374326 | 644.968913 | 0.826193 | ablation_C_clean |
| catboost_fixed_C_clean_direct | fold_2 | 2025-01-01 | 2025-06-30 | 2025-07-01 | 2025-08-31 | 176.510820 | 631.141887 | 0.820723 | ablation_C_clean |
| catboost_fixed_C_clean_direct | fold_3 | 2025-01-01 | 2025-08-31 | 2025-09-01 | 2025-10-31 | 113.664719 | 635.312153 | 0.826685 | ablation_C_clean |
| catboost_fixed_C_raw_direct | fold_1 | 2025-01-01 | 2025-04-30 | 2025-05-01 | 2025-06-30 | 143.070483 | 644.285276 | 0.826562 | ablation_C_raw |
| catboost_fixed_C_raw_direct | fold_2 | 2025-01-01 | 2025-06-30 | 2025-07-01 | 2025-08-31 | 172.298664 | 629.604167 | 0.821596 | ablation_C_raw |
| catboost_fixed_C_raw_direct | fold_3 | 2025-01-01 | 2025-08-31 | 2025-09-01 | 2025-10-31 | 116.960665 | 636.925913 | 0.825804 | ablation_C_raw |
| catboost_fixed_D_clean_direct | fold_1 | 2025-01-01 | 2025-04-30 | 2025-05-01 | 2025-06-30 | 152.553674 | 649.674286 | 0.823648 | ablation_D_clean |
| catboost_fixed_D_clean_direct | fold_2 | 2025-01-01 | 2025-06-30 | 2025-07-01 | 2025-08-31 | 179.668498 | 631.384462 | 0.820585 | ablation_D_clean |
| catboost_fixed_D_clean_direct | fold_3 | 2025-01-01 | 2025-08-31 | 2025-09-01 | 2025-10-31 | 110.593839 | 634.722325 | 0.827007 | ablation_D_clean |
| catboost_fixed_E_clean_direct | fold_1 | 2025-01-01 | 2025-04-30 | 2025-05-01 | 2025-06-30 | 130.081108 | 640.400958 | 0.828647 | ablation_E_clean |
| catboost_fixed_E_clean_direct | fold_2 | 2025-01-01 | 2025-06-30 | 2025-07-01 | 2025-08-31 | 160.600093 | 626.582916 | 0.823304 | ablation_E_clean |
| catboost_fixed_E_clean_direct | fold_3 | 2025-01-01 | 2025-08-31 | 2025-09-01 | 2025-10-31 | 116.655746 | 636.683008 | 0.825937 | ablation_E_clean |
| catboost_fixed_F_clean_direct | fold_1 | 2025-01-01 | 2025-04-30 | 2025-05-01 | 2025-06-30 | 140.612623 | 642.631102 | 0.827451 | ablation_F_clean |
| catboost_fixed_F_clean_direct | fold_2 | 2025-01-01 | 2025-06-30 | 2025-07-01 | 2025-08-31 | 137.084228 | 624.927086 | 0.824237 | ablation_F_clean |
| catboost_fixed_F_clean_direct | fold_3 | 2025-01-01 | 2025-08-31 | 2025-09-01 | 2025-10-31 | 115.293720 | 635.830760 | 0.826402 | ablation_F_clean |
| catboost_fixed_F_clean_log1p | fold_1 | 2025-01-01 | 2025-04-30 | 2025-05-01 | 2025-06-30 | 142.561320 | 644.491670 | 0.826451 | ablation_F_clean |
| catboost_fixed_F_clean_log1p | fold_2 | 2025-01-01 | 2025-06-30 | 2025-07-01 | 2025-08-31 | 128.956789 | 623.894968 | 0.824817 | ablation_F_clean |
| catboost_fixed_F_clean_log1p | fold_3 | 2025-01-01 | 2025-08-31 | 2025-09-01 | 2025-10-31 | 116.707649 | 636.504505 | 0.826034 | ablation_F_clean |
| catboost_fixed_F_raw_direct | fold_1 | 2025-01-01 | 2025-04-30 | 2025-05-01 | 2025-06-30 | 140.654160 | 643.299146 | 0.827092 | ablation_F_raw |
| catboost_fixed_F_raw_direct | fold_2 | 2025-01-01 | 2025-06-30 | 2025-07-01 | 2025-08-31 | 129.141961 | 623.526610 | 0.825023 | ablation_F_raw |
| catboost_fixed_F_raw_direct | fold_3 | 2025-01-01 | 2025-08-31 | 2025-09-01 | 2025-10-31 | 124.748898 | 639.908349 | 0.824169 | ablation_F_raw |

MAE and RMSE are in posted-rate units (USD); R2 is dimensionless. CSV metrics retain normal floating-point precision. No row or high-rate outlier is removed. Predictions are evaluated directly without clipping; the CSV records any nonpositive local predictions. Production outputs will need their separate positivity checks.

## Unweighted mean across the three folds

| model | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| global_median | 1150.997962 | 1572.594471 | -0.068519 |
| distance_linear | 201.231984 | 658.826278 | 0.812447 |
| ridge | 252.811958 | 687.475618 | 0.795407 |
| catboost_d7_lr003_l2_3 | 136.353690 | 637.064128 | 0.824627 |
| catboost_d6_lr005_l2_5 | 129.661646 | 633.980861 | 0.826301 |
| catboost_d8_lr004_l2_7 | 136.297802 | 637.050283 | 0.824635 |
| catboost_ablation_A_clean_direct | 147.832460 | 639.745084 | 0.823128 |
| catboost_ablation_B_clean_direct | 144.021589 | 638.547036 | 0.823788 |
| catboost_ablation_C_clean_direct | 136.766525 | 636.735147 | 0.824764 |
| catboost_ablation_D_clean_direct | 138.475611 | 637.121197 | 0.824581 |
| catboost_ablation_E_clean_direct | 130.469460 | 633.912924 | 0.826323 |
| catboost_ablation_F_clean_direct | 129.661646 | 633.980861 | 0.826301 |
| catboost_ablation_C_raw_direct | 137.110003 | 636.088447 | 0.825129 |
| catboost_ablation_F_raw_direct | 130.151436 | 635.338031 | 0.825561 |
| catboost_ablation_C_clean_log1p | 141.406184 | 637.806966 | 0.824190 |
| catboost_ablation_F_clean_log1p | 127.877247 | 634.503153 | 0.826025 |
| catboost_fixed_C_clean_direct | 143.849955 | 637.140984 | 0.824534 |
| catboost_fixed_C_raw_direct | 144.109937 | 636.938452 | 0.824654 |
| catboost_fixed_D_clean_direct | 147.605337 | 638.593691 | 0.823747 |
| catboost_fixed_E_clean_direct | 135.778982 | 634.555627 | 0.825962 |
| catboost_fixed_F_clean_direct | 130.996857 | 634.462983 | 0.826030 |
| catboost_fixed_F_clean_log1p | 129.408586 | 634.963714 | 0.825767 |
| catboost_fixed_F_raw_direct | 131.515007 | 635.578035 | 0.825428 |

These are arithmetic means of per-fold metrics, not metrics recomputed on pooled predictions. Expanding training windows share historical observations; the folds are not independent samples.

## Primary September–October holdout

| model | MAE | RMSE | R2 |
| --- | --- | --- | --- |
| global_median | 1148.923726 | 1569.424179 | -0.057650 |
| distance_linear | 196.950147 | 654.415640 | 0.816106 |
| ridge | 228.675612 | 675.033817 | 0.804335 |
| catboost_d7_lr003_l2_3 | 117.956285 | 636.511881 | 0.826030 |
| catboost_d6_lr005_l2_5 | 115.175211 | 635.786705 | 0.826426 |
| catboost_d8_lr004_l2_7 | 119.220254 | 637.609460 | 0.825430 |
| catboost_ablation_A_clean_direct | 125.780808 | 638.512540 | 0.824935 |
| catboost_ablation_B_clean_direct | 117.151174 | 637.529210 | 0.825474 |
| catboost_ablation_C_clean_direct | 113.296080 | 635.286918 | 0.826699 |
| catboost_ablation_D_clean_direct | 110.471875 | 634.689882 | 0.827025 |
| catboost_ablation_E_clean_direct | 116.655746 | 636.683008 | 0.825937 |
| catboost_ablation_F_clean_direct | 115.175211 | 635.786705 | 0.826426 |
| catboost_ablation_C_raw_direct | 116.960665 | 636.925913 | 0.825804 |
| catboost_ablation_F_raw_direct | 123.852785 | 639.365688 | 0.824467 |
| catboost_ablation_C_clean_log1p | 123.130716 | 639.207075 | 0.824554 |
| catboost_ablation_F_clean_log1p | 116.067646 | 636.329647 | 0.826130 |
| catboost_fixed_C_clean_direct | 113.664719 | 635.312153 | 0.826685 |
| catboost_fixed_C_raw_direct | 116.960665 | 636.925913 | 0.825804 |
| catboost_fixed_D_clean_direct | 110.593839 | 634.722325 | 0.827007 |
| catboost_fixed_E_clean_direct | 116.655746 | 636.683008 | 0.825937 |
| catboost_fixed_F_clean_direct | 115.293720 | 635.830760 | 0.826402 |
| catboost_fixed_F_clean_log1p | 116.707649 | 636.504505 | 0.826034 |
| catboost_fixed_F_raw_direct | 124.748898 | 639.908349 | 0.824169 |

## Observations from these fixed experiments

Among these experiments, **catboost_ablation_F_clean_log1p** has the lowest mean chronological MAE (127.88). **catboost_ablation_D_clean_direct** has the lowest primary September–October MAE (110.47). The frozen production decision, inference constraints, and fixed-budget confirmation are documented in `docs/model_selection.md`; the feature/weight/target evidence is in `reports/feature_ablation.md`.

Nonpositive local predictions across all three validation windows:

| model | nonpositive_predictions |
| --- | --- |
| global_median | 0 |
| distance_linear | 0 |
| ridge | 20 |
| catboost_d7_lr003_l2_3 | 0 |
| catboost_d6_lr005_l2_5 | 0 |
| catboost_d8_lr004_l2_7 | 0 |
| catboost_ablation_A_clean_direct | 0 |
| catboost_ablation_B_clean_direct | 0 |
| catboost_ablation_C_clean_direct | 0 |
| catboost_ablation_D_clean_direct | 0 |
| catboost_ablation_E_clean_direct | 0 |
| catboost_ablation_F_clean_direct | 0 |
| catboost_ablation_C_raw_direct | 0 |
| catboost_ablation_F_raw_direct | 0 |
| catboost_ablation_C_clean_log1p | 0 |
| catboost_ablation_F_clean_log1p | 0 |
| catboost_fixed_C_clean_direct | 0 |
| catboost_fixed_C_raw_direct | 0 |
| catboost_fixed_D_clean_direct | 0 |
| catboost_fixed_E_clean_direct | 0 |
| catboost_fixed_F_clean_direct | 0 |
| catboost_fixed_F_clean_log1p | 0 |
| catboost_fixed_F_raw_direct | 0 |

These local predictions are retained in the metrics without clipping. They are not final submission outputs. A configuration that produces nonpositive predictions needs an explicitly validated treatment before production use.

## Feature sets and fixed configurations

- **training_fold_median:** no predictive inputs; median of training targets only.
- **distance_only:** distance.
- **all_26_candidates:** distance, equipment, weight_clean, weight_missing, weight_negative, pickup, delivery, pickup_lat, pickup_lon, delivery_lat, delivery_lon, route, month, day, day_of_week, day_of_year, iso_week, weekend, days_since_reference_date, sin_day_of_week, cos_day_of_week, sin_day_of_year, cos_day_of_year, market_index, market_index_missing, quote_signal.
- **ablation_A_clean:** distance, equipment, weight_clean, weight_missing, weight_negative, month, day, day_of_week, day_of_year, iso_week, weekend, days_since_reference_date, sin_day_of_week, cos_day_of_week, sin_day_of_year, cos_day_of_year.
- **ablation_B_clean:** distance, equipment, weight_clean, weight_missing, weight_negative, pickup, delivery, month, day, day_of_week, day_of_year, iso_week, weekend, days_since_reference_date, sin_day_of_week, cos_day_of_week, sin_day_of_year, cos_day_of_year.
- **ablation_C_clean:** distance, equipment, weight_clean, weight_missing, weight_negative, pickup, delivery, pickup_lat, pickup_lon, delivery_lat, delivery_lon, month, day, day_of_week, day_of_year, iso_week, weekend, days_since_reference_date, sin_day_of_week, cos_day_of_week, sin_day_of_year, cos_day_of_year.
- **ablation_D_clean:** distance, equipment, weight_clean, weight_missing, weight_negative, pickup, delivery, pickup_lat, pickup_lon, delivery_lat, delivery_lon, route, month, day, day_of_week, day_of_year, iso_week, weekend, days_since_reference_date, sin_day_of_week, cos_day_of_week, sin_day_of_year, cos_day_of_year.
- **ablation_E_clean:** distance, equipment, weight_clean, weight_missing, weight_negative, pickup, delivery, pickup_lat, pickup_lon, delivery_lat, delivery_lon, route, month, day, day_of_week, day_of_year, iso_week, weekend, days_since_reference_date, sin_day_of_week, cos_day_of_week, sin_day_of_year, cos_day_of_year, market_index, market_index_missing.
- **ablation_F_clean:** distance, equipment, weight_clean, weight_missing, weight_negative, pickup, delivery, pickup_lat, pickup_lon, delivery_lat, delivery_lon, route, month, day, day_of_week, day_of_year, iso_week, weekend, days_since_reference_date, sin_day_of_week, cos_day_of_week, sin_day_of_year, cos_day_of_year, market_index, market_index_missing, quote_signal.
- **ablation_C_raw:** distance, equipment, weight_raw, pickup, delivery, pickup_lat, pickup_lon, delivery_lat, delivery_lon, month, day, day_of_week, day_of_year, iso_week, weekend, days_since_reference_date, sin_day_of_week, cos_day_of_week, sin_day_of_year, cos_day_of_year.
- **ablation_F_raw:** distance, equipment, weight_raw, pickup, delivery, pickup_lat, pickup_lon, delivery_lat, delivery_lon, route, month, day, day_of_week, day_of_year, iso_week, weekend, days_since_reference_date, sin_day_of_week, cos_day_of_week, sin_day_of_year, cos_day_of_year, market_index, market_index_missing, quote_signal.

- **global_median:** `{"strategy": "median"}`.
- **distance_linear:** `{"fit_intercept": true}`.
- **ridge:** `{"alpha": 1.0, "max_iter": 10000, "random_state": 42, "solver": "lsqr", "tol": 1e-06}`.
- **catboost_d7_lr003_l2_3:** `{"allow_writing_files": false, "cat_features": ["equipment", "pickup", "delivery", "route"], "depth": 7, "early_stopping_rounds": 100, "eval_metric": "MAE", "iterations": 1500, "l2_leaf_reg": 3, "learning_rate": 0.03, "loss_function": "MAE", "nan_mode": "Min", "random_seed": 42, "thread_count": 4, "use_best_model": true, "verbose": false}`.
- **catboost_d6_lr005_l2_5:** `{"allow_writing_files": false, "cat_features": ["equipment", "pickup", "delivery", "route"], "depth": 6, "early_stopping_rounds": 100, "eval_metric": "MAE", "iterations": 1500, "l2_leaf_reg": 5, "learning_rate": 0.05, "loss_function": "MAE", "nan_mode": "Min", "random_seed": 42, "thread_count": 4, "use_best_model": true, "verbose": false}`.
- **catboost_d8_lr004_l2_7:** `{"allow_writing_files": false, "cat_features": ["equipment", "pickup", "delivery", "route"], "depth": 8, "early_stopping_rounds": 100, "eval_metric": "MAE", "iterations": 1500, "l2_leaf_reg": 7, "learning_rate": 0.04, "loss_function": "MAE", "nan_mode": "Min", "random_seed": 42, "thread_count": 4, "use_best_model": true, "verbose": false}`.
- **catboost_ablation_A_clean_direct:** `{"allow_writing_files": false, "cat_features": ["equipment"], "depth": 6, "early_stopping_rounds": 100, "eval_metric": "MAE", "iterations": 1500, "l2_leaf_reg": 5, "learning_rate": 0.05, "loss_function": "MAE", "nan_mode": "Min", "random_seed": 42, "thread_count": 4, "use_best_model": true, "verbose": false}`.
- **catboost_ablation_B_clean_direct:** `{"allow_writing_files": false, "cat_features": ["equipment", "pickup", "delivery"], "depth": 6, "early_stopping_rounds": 100, "eval_metric": "MAE", "iterations": 1500, "l2_leaf_reg": 5, "learning_rate": 0.05, "loss_function": "MAE", "nan_mode": "Min", "random_seed": 42, "thread_count": 4, "use_best_model": true, "verbose": false}`.
- **catboost_ablation_C_clean_direct:** `{"allow_writing_files": false, "cat_features": ["equipment", "pickup", "delivery"], "depth": 6, "early_stopping_rounds": 100, "eval_metric": "MAE", "iterations": 1500, "l2_leaf_reg": 5, "learning_rate": 0.05, "loss_function": "MAE", "nan_mode": "Min", "random_seed": 42, "thread_count": 4, "use_best_model": true, "verbose": false}`.
- **catboost_ablation_D_clean_direct:** `{"allow_writing_files": false, "cat_features": ["equipment", "pickup", "delivery", "route"], "depth": 6, "early_stopping_rounds": 100, "eval_metric": "MAE", "iterations": 1500, "l2_leaf_reg": 5, "learning_rate": 0.05, "loss_function": "MAE", "nan_mode": "Min", "random_seed": 42, "thread_count": 4, "use_best_model": true, "verbose": false}`.
- **catboost_ablation_E_clean_direct:** `{"allow_writing_files": false, "cat_features": ["equipment", "pickup", "delivery", "route"], "depth": 6, "early_stopping_rounds": 100, "eval_metric": "MAE", "iterations": 1500, "l2_leaf_reg": 5, "learning_rate": 0.05, "loss_function": "MAE", "nan_mode": "Min", "random_seed": 42, "thread_count": 4, "use_best_model": true, "verbose": false}`.
- **catboost_ablation_F_clean_direct:** `{"allow_writing_files": false, "cat_features": ["equipment", "pickup", "delivery", "route"], "depth": 6, "early_stopping_rounds": 100, "eval_metric": "MAE", "iterations": 1500, "l2_leaf_reg": 5, "learning_rate": 0.05, "loss_function": "MAE", "nan_mode": "Min", "random_seed": 42, "thread_count": 4, "use_best_model": true, "verbose": false}`.
- **catboost_ablation_C_raw_direct:** `{"allow_writing_files": false, "cat_features": ["equipment", "pickup", "delivery"], "depth": 6, "early_stopping_rounds": 100, "eval_metric": "MAE", "iterations": 1500, "l2_leaf_reg": 5, "learning_rate": 0.05, "loss_function": "MAE", "nan_mode": "Min", "random_seed": 42, "thread_count": 4, "use_best_model": true, "verbose": false}`.
- **catboost_ablation_F_raw_direct:** `{"allow_writing_files": false, "cat_features": ["equipment", "pickup", "delivery", "route"], "depth": 6, "early_stopping_rounds": 100, "eval_metric": "MAE", "iterations": 1500, "l2_leaf_reg": 5, "learning_rate": 0.05, "loss_function": "MAE", "nan_mode": "Min", "random_seed": 42, "thread_count": 4, "use_best_model": true, "verbose": false}`.
- **catboost_ablation_C_clean_log1p:** `{"allow_writing_files": false, "cat_features": ["equipment", "pickup", "delivery"], "depth": 6, "early_stopping_rounds": 100, "eval_metric": "DollarMAE", "iterations": 1500, "l2_leaf_reg": 5, "learning_rate": 0.05, "loss_function": "MAE", "nan_mode": "Min", "random_seed": 42, "thread_count": 4, "use_best_model": true, "verbose": false}`.
- **catboost_ablation_F_clean_log1p:** `{"allow_writing_files": false, "cat_features": ["equipment", "pickup", "delivery", "route"], "depth": 6, "early_stopping_rounds": 100, "eval_metric": "DollarMAE", "iterations": 1500, "l2_leaf_reg": 5, "learning_rate": 0.05, "loss_function": "MAE", "nan_mode": "Min", "random_seed": 42, "thread_count": 4, "use_best_model": true, "verbose": false}`.
- **catboost_fixed_C_clean_direct:** `{"allow_writing_files": false, "cat_features": ["equipment", "pickup", "delivery"], "depth": 6, "early_stopping_rounds": null, "eval_metric": "MAE", "iterations": 541, "l2_leaf_reg": 5, "learning_rate": 0.05, "loss_function": "MAE", "nan_mode": "Min", "random_seed": 42, "thread_count": 4, "use_best_model": false, "verbose": false}`.
- **catboost_fixed_C_raw_direct:** `{"allow_writing_files": false, "cat_features": ["equipment", "pickup", "delivery"], "depth": 6, "early_stopping_rounds": null, "eval_metric": "MAE", "iterations": 413, "l2_leaf_reg": 5, "learning_rate": 0.05, "loss_function": "MAE", "nan_mode": "Min", "random_seed": 42, "thread_count": 4, "use_best_model": false, "verbose": false}`.
- **catboost_fixed_D_clean_direct:** `{"allow_writing_files": false, "cat_features": ["equipment", "pickup", "delivery", "route"], "depth": 6, "early_stopping_rounds": null, "eval_metric": "MAE", "iterations": 897, "l2_leaf_reg": 5, "learning_rate": 0.05, "loss_function": "MAE", "nan_mode": "Min", "random_seed": 42, "thread_count": 4, "use_best_model": false, "verbose": false}`.
- **catboost_fixed_E_clean_direct:** `{"allow_writing_files": false, "cat_features": ["equipment", "pickup", "delivery", "route"], "depth": 6, "early_stopping_rounds": null, "eval_metric": "MAE", "iterations": 710, "l2_leaf_reg": 5, "learning_rate": 0.05, "loss_function": "MAE", "nan_mode": "Min", "random_seed": 42, "thread_count": 4, "use_best_model": false, "verbose": false}`.
- **catboost_fixed_F_clean_direct:** `{"allow_writing_files": false, "cat_features": ["equipment", "pickup", "delivery", "route"], "depth": 6, "early_stopping_rounds": null, "eval_metric": "MAE", "iterations": 284, "l2_leaf_reg": 5, "learning_rate": 0.05, "loss_function": "MAE", "nan_mode": "Min", "random_seed": 42, "thread_count": 4, "use_best_model": false, "verbose": false}`.
- **catboost_fixed_F_clean_log1p:** `{"allow_writing_files": false, "cat_features": ["equipment", "pickup", "delivery", "route"], "depth": 6, "early_stopping_rounds": null, "eval_metric": "DollarMAE", "iterations": 399, "l2_leaf_reg": 5, "learning_rate": 0.05, "loss_function": "MAE", "nan_mode": "Min", "random_seed": 42, "thread_count": 4, "use_best_model": false, "verbose": false}`.
- **catboost_fixed_F_raw_direct:** `{"allow_writing_files": false, "cat_features": ["equipment", "pickup", "delivery", "route"], "depth": 6, "early_stopping_rounds": null, "eval_metric": "MAE", "iterations": 205, "l2_leaf_reg": 5, "learning_rate": 0.05, "loss_function": "MAE", "nan_mode": "Min", "random_seed": 42, "thread_count": 4, "use_best_model": false, "verbose": false}`.

## CatBoost iterations and training time

| model | fold | best_iteration | tree_count | iterations_executed | training_seconds | prediction_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| catboost_d7_lr003_l2_3 | fold_1 | 764 | 765 | 865 | 13.888 | 0.015 |
| catboost_d6_lr005_l2_5 | fold_1 | 283 | 284 | 384 | 4.863 | 0.010 |
| catboost_d8_lr004_l2_7 | fold_1 | 612 | 613 | 713 | 13.048 | 0.016 |
| catboost_d7_lr003_l2_3 | fold_2 | 243 | 244 | 344 | 7.073 | 0.009 |
| catboost_d6_lr005_l2_5 | fold_2 | 161 | 162 | 262 | 4.443 | 0.009 |
| catboost_d8_lr004_l2_7 | fold_2 | 210 | 211 | 311 | 7.850 | 0.012 |
| catboost_d7_lr003_l2_3 | fold_3 | 509 | 510 | 610 | 15.815 | 0.011 |
| catboost_d6_lr005_l2_5 | fold_3 | 291 | 292 | 392 | 7.921 | 0.009 |
| catboost_d8_lr004_l2_7 | fold_3 | 423 | 424 | 524 | 16.036 | 0.011 |
| catboost_ablation_A_clean_direct | fold_1 | 508 | 509 | 609 | 3.667 | 0.004 |
| catboost_ablation_A_clean_direct | fold_2 | 89 | 90 | 190 | 1.799 | 0.003 |
| catboost_ablation_A_clean_direct | fold_3 | 288 | 289 | 389 | 4.680 | 0.003 |
| catboost_ablation_B_clean_direct | fold_1 | 905 | 906 | 1006 | 8.447 | 0.036 |
| catboost_ablation_B_clean_direct | fold_2 | 101 | 102 | 202 | 2.548 | 0.005 |
| catboost_ablation_B_clean_direct | fold_3 | 1496 | 1497 | 1500 | 23.946 | 0.042 |
| catboost_ablation_C_clean_direct | fold_1 | 540 | 541 | 641 | 5.868 | 0.008 |
| catboost_ablation_C_clean_direct | fold_2 | 83 | 84 | 184 | 2.414 | 0.005 |
| catboost_ablation_C_clean_direct | fold_3 | 759 | 760 | 860 | 15.431 | 0.010 |
| catboost_ablation_D_clean_direct | fold_1 | 896 | 897 | 997 | 12.021 | 0.019 |
| catboost_ablation_D_clean_direct | fold_2 | 93 | 94 | 194 | 3.294 | 0.010 |
| catboost_ablation_D_clean_direct | fold_3 | 925 | 926 | 1026 | 20.746 | 0.016 |
| catboost_ablation_E_clean_direct | fold_1 | 950 | 951 | 1051 | 12.106 | 0.045 |
| catboost_ablation_E_clean_direct | fold_2 | 104 | 105 | 205 | 3.448 | 0.008 |
| catboost_ablation_E_clean_direct | fold_3 | 709 | 710 | 810 | 16.782 | 0.014 |
| catboost_ablation_F_clean_direct | fold_1 | 283 | 284 | 384 | 4.863 | 0.010 |
| catboost_ablation_F_clean_direct | fold_2 | 161 | 162 | 262 | 4.443 | 0.009 |
| catboost_ablation_F_clean_direct | fold_3 | 291 | 292 | 392 | 7.921 | 0.009 |
| catboost_ablation_C_raw_direct | fold_1 | 1160 | 1161 | 1261 | 11.066 | 0.016 |
| catboost_ablation_C_raw_direct | fold_2 | 94 | 95 | 195 | 2.631 | 0.005 |
| catboost_ablation_C_raw_direct | fold_3 | 412 | 413 | 513 | 8.675 | 0.007 |
| catboost_ablation_F_raw_direct | fold_1 | 204 | 205 | 305 | 3.719 | 0.009 |
| catboost_ablation_F_raw_direct | fold_2 | 129 | 130 | 230 | 3.902 | 0.008 |
| catboost_ablation_F_raw_direct | fold_3 | 260 | 261 | 361 | 7.849 | 0.009 |
| catboost_ablation_C_clean_log1p | fold_1 | 611 | 612 | 712 | 12.197 | 0.009 |
| catboost_ablation_C_clean_log1p | fold_2 | 94 | 95 | 195 | 4.368 | 0.005 |
| catboost_ablation_C_clean_log1p | fold_3 | 746 | 747 | 847 | 24.694 | 0.011 |
| catboost_ablation_F_clean_log1p | fold_1 | 398 | 399 | 499 | 10.181 | 0.009 |
| catboost_ablation_F_clean_log1p | fold_2 | 115 | 116 | 216 | 5.658 | 0.008 |
| catboost_ablation_F_clean_log1p | fold_3 | 557 | 558 | 658 | 22.286 | 0.012 |
| catboost_fixed_C_clean_direct | fold_1 | — | 541 | 541 | 4.577 | 0.008 |
| catboost_fixed_C_clean_direct | fold_2 | — | 541 | 541 | 6.768 | 0.008 |
| catboost_fixed_C_clean_direct | fold_3 | — | 541 | 541 | 8.717 | 0.008 |
| catboost_fixed_C_raw_direct | fold_1 | — | 413 | 413 | 3.720 | 0.008 |
| catboost_fixed_C_raw_direct | fold_2 | — | 413 | 413 | 5.106 | 0.007 |
| catboost_fixed_C_raw_direct | fold_3 | — | 413 | 413 | 6.414 | 0.007 |
| catboost_fixed_D_clean_direct | fold_1 | — | 897 | 897 | 9.147 | 0.022 |
| catboost_fixed_D_clean_direct | fold_2 | — | 897 | 897 | 13.401 | 0.019 |
| catboost_fixed_D_clean_direct | fold_3 | — | 897 | 897 | 17.466 | 0.018 |
| catboost_fixed_E_clean_direct | fold_1 | — | 710 | 710 | 7.439 | 0.016 |
| catboost_fixed_E_clean_direct | fold_2 | — | 710 | 710 | 10.536 | 0.017 |
| catboost_fixed_E_clean_direct | fold_3 | — | 710 | 710 | 14.153 | 0.015 |
| catboost_fixed_F_clean_direct | fold_1 | — | 284 | 284 | 3.044 | 0.011 |
| catboost_fixed_F_clean_direct | fold_2 | — | 284 | 284 | 4.460 | 0.011 |
| catboost_fixed_F_clean_direct | fold_3 | — | 284 | 284 | 5.559 | 0.010 |
| catboost_fixed_F_clean_log1p | fold_1 | — | 399 | 399 | 6.455 | 0.011 |
| catboost_fixed_F_clean_log1p | fold_2 | — | 399 | 399 | 8.643 | 0.011 |
| catboost_fixed_F_clean_log1p | fold_3 | — | 399 | 399 | 11.280 | 0.011 |
| catboost_fixed_F_raw_direct | fold_1 | — | 205 | 205 | 2.396 | 0.011 |
| catboost_fixed_F_raw_direct | fold_2 | — | 205 | 205 | 3.244 | 0.012 |
| catboost_fixed_F_raw_direct | fold_3 | — | 205 | 205 | 4.237 | 0.009 |

best_iteration is zero-based; tree_count is the number of retained trees. iterations_executed includes rounds run before stopping, including rounds later discarded by use_best_model=True. Fixed-budget confirmation rows have no best_iteration because they do not receive an eval_set. Training time excludes prediction time. Baseline records from the earlier run contain combined fitting/prediction time only; missing CatBoost-specific fields are not applicable to them.

The initial hyperparameter comparison used three joint configurations: depth 7 / learning_rate 0.03 / l2_leaf_reg 3; depth 6 / learning_rate 0.05 / l2_leaf_reg 5; depth 8 / learning_rate 0.04 / l2_leaf_reg 7, each across three folds. Later feature/weight/target comparisons hold the strongest configuration (depth 6 / learning_rate 0.05 / l2_leaf_reg 5) fixed. Early-stopped comparisons use a 1,500-iteration cap and 100-round patience; fixed-budget confirmations use the frozen iteration counts without an eval_set. Full parameters are logged in every record.

**Early-stopping methodology:** for adaptive-budget comparisons, each chronological outer validation window is also CatBoost's eval_set. Gradient/tree fitting and native categorical statistics use training-fold observations; eval_set labels choose the best iteration. Fixed-budget confirmation fits do not use evaluation labels while fitting, but their features/configuration were chosen using the same earlier validation windows. Metrics remain model-selection estimates, not an untouched independent test. Direct-target stopping uses MAE; log1p-target stopping uses the custom DollarMAE metric, converting both prediction and truth back with expm1. All reported MAE/RMSE/R2 use original dollar-scale outcomes. September–October retains its role as the primary local selection holdout. No November–December final-validation observations are loaded or used.

CatBoost uses explicit string categories and native numeric NaN handling, with no sklearn one-hot encoding or full-data imputation. No external target encoding is computed. Selected feature subsets are listed below, and raw weight is distinguished from abs(weight) plus original missing/negative indicators. Coordinates are retained in eligible final models. Model serialization and production training are separate later steps.


The distance baseline predicts an intercept plus a distance coefficient. Its distance median imputer is fitted only on training rows. Ridge uses training-fold numeric medians, scaling, and one-hot categoricals with `handle_unknown="ignore"`, retaining coordinates for unfamiliar city labels. The global median is calculated independently from each fold's training targets. Every fold receives fresh estimators and fresh learned preprocessing. Feature engineering is deterministic, excludes load_id/posted_rate, preserves missing/negative-weight flags, and uses a fixed reference date.

Market/quote provenance, availability, missingness, and temporal behavior are documented in `reports/feature_ablation.md`. These local results do not establish accuracy for future unseen cities or the fixed December scenario. No production model, submission predictions, or Spotter hidden metric is created by these experiment commands.

The CSV also records row counts, fixed estimator parameters, elapsed fitting/prediction time, and nonpositive prediction counts. Timing varies by machine and run.
