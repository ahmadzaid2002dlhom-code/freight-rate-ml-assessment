"""Generate experiment, market-quality, and frozen-model decision documents."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_model_experiments import check_protected, selection_table
from src.evaluate import markdown_table, save_report
from src.features import FEATURE_GROUPS


def market_analysis(frame):
    data = frame.copy()
    data["month"] = pd.to_datetime(data.date).dt.strftime("%Y-%m")
    rows = []
    for month, block in data.groupby("month", sort=True):
        for feature in ("market_index", "quote_signal"):
            values = block[feature]
            rows.append({"month": month, "feature": feature, "rows": len(block),
                         "missing": int(values.isna().sum()), "missing_pct": 100*values.isna().mean(),
                         "mean": values.mean(), "std": values.std(), "median": values.median(),
                         "min": values.min(), "max": values.max()})
    monthly = pd.DataFrame(rows)
    monthly.to_csv(ROOT / "reports/market_feature_monthly.csv", index=False)
    overall = pd.DataFrame([{
        "feature": name, "missing": int(data[name].isna().sum()),
        "missing_pct": 100*data[name].isna().mean(), "mean": data[name].mean(),
        "std": data[name].std(), "min": data[name].min(), "max": data[name].max(),
    } for name in ("market_index", "quote_signal")])
    return overall, monthly


def market_deltas(results):
    clean = results.loc[results.experiment_phase.eq("feature_ablation")]
    rows = []
    for before, after, label in [("D", "E", "Add market_index + missing flag"), ("E", "F", "Add quote_signal")]:
        old = clean.loc[clean.feature_group.eq(before)].set_index("fold")
        new = clean.loc[clean.feature_group.eq(after)].set_index("fold")
        for fold in ("fold_1", "fold_2", "fold_3"):
            rows.append({"change": label, "fold": fold,
                         "MAE": new.loc[fold,"MAE"]-old.loc[fold,"MAE"],
                         "RMSE": new.loc[fold,"RMSE"]-old.loc[fold,"RMSE"],
                         "R2": new.loc[fold,"R2"]-old.loc[fold,"R2"]})
    return pd.DataFrame(rows)


def report_ablation(results, frame):
    overall, monthly = market_analysis(frame)
    group_rows = results.loc[results.experiment_phase.eq("feature_ablation")]
    means = group_rows.groupby("feature_group", sort=True)[["MAE","RMSE","R2"]].mean().reset_index()
    weight_rows = results.loc[results.experiment_phase.eq("weight_representation")]
    weight_groups = weight_rows.feature_group.unique()
    weights = results.loc[results.feature_group.isin(weight_groups) & results.target_strategy.eq("direct")
                          & ~results.experiment_phase.eq("fixed_budget_confirmation")]
    target_rows = results.loc[results.experiment_phase.eq("target_strategy")]
    comparisons = []
    for row in target_rows.drop_duplicates("model").itertuples():
        comparisons.append(results.loc[results.feature_group.eq(row.feature_group)
                                       & results.weight_representation.eq(row.weight_representation)
                                       & ~results.experiment_phase.eq("fixed_budget_confirmation")])
    targets = pd.concat(comparisons).drop_duplicates(["model","fold"])
    groups = pd.DataFrame([{"group": group, "count": len(columns), "features": ", ".join(columns)} for group,columns in FEATURE_GROUPS.items()])
    delta = market_deltas(results)
    per_fold_columns = ["model", "fold", "MAE", "RMSE", "R2", "best_iteration", "tree_count"]
    target_summary = frame.posted_rate.describe(percentiles=[0.01,0.05,0.5,0.95,0.99]).rename("posted_rate").reset_index().rename(columns={"index":"statistic"})
    market_means = monthly.loc[monthly.feature.eq("market_index")]
    quote_means = monthly.loc[monthly.feature.eq("quote_signal")]
    text = f"""# Controlled feature, weight, and target experiments

Reproduce experiments with `.venv/bin/python scripts/run_model_experiments.py --phase all`; regenerate this report with `.venv/bin/python scripts/report_model_selection.py`. Only labeled development rows are loaded. The fixed folds are January–April/May–June, January–June/July–August, and January–August/September–October (19,110/9,696; 28,806/9,671; 38,477/9,523 rows). No final-validation values are consulted for these comparisons.

## Controlled feature ablation

Use the strongest previously tested family/configuration: CatBoost, depth=6, learning_rate=0.05, l2_leaf_reg=5, MAE loss, random_seed=42, 1,500-iteration cap, and 100-round patience. Each ablation uses the same folds, canonical column order, and training-only native categorical statistics. Calendar includes the seven ordinary calendar features and four cyclic features. Cleaned weight includes abs(weight), original missingness, and original negative-sign flags. E includes both market_index and its missing flag.

{markdown_table(groups)}

Unweighted mean fold metrics:

{markdown_table(means)}

Per-fold metrics and retained iterations:

{markdown_table(group_rows[per_fold_columns])}

The F result reuses the identical prior depth-6/lr0.05/l2=5 full-feature experiment, including its metrics and timing. Five other groups required 15 new fits. All 48,000 rows are retained in the source; nothing is removed for missingness, negative weights, or target outliers. Coordinates remain mandatory in eligible final groups C–F; A/B are comparison baselines.

## Weight representation

Raw weight means the original signed value, preserving numeric NaN with native handling. Clean means abs(weight) plus weight_missing and weight_negative. Other feature groups and parameters remain identical. This is a comparison of complete representations; it does not isolate absolute-value correction from the contribution of the two flags. The strongest full-inference and December-feasible direct feature groups were chosen before these representation tests; the raw alternative is evaluated on both where different.

{markdown_table(selection_table(weights))}

{markdown_table(weights[per_fold_columns])}

Negative signs are consistent with the earlier sign-corruption investigation, not a proven correction. The winning representation is chosen by chronological performance and robustness, not that hypothesis alone. Numeric NaN is never filled from holdout/final inputs.

## Market feature quality and availability

Development-only overall summaries:

{markdown_table(overall)}

Monthly development-only summaries (full precision in `reports/market_feature_monthly.csv`):

{markdown_table(monthly[["month","feature","missing","missing_pct","mean","std","median"]])}

Development market_index monthly means range from {market_means['mean'].min():.6f} to {market_means['mean'].max():.6f}; quote_signal monthly means range from {quote_means['mean'].min():.6f} to {quote_means['mean'].max():.6f}. This shows calendar-associated distribution changes, not guaranteed repeatable annual seasonality. Missingness is tracked monthly; no missing observation is silently deleted.

Paired chronological feature additions (new minus old metric; negative MAE/RMSE is better, positive R2 is better):

{markdown_table(delta)}

Gains vary by fold; pooled marginal correlations do not decide usefulness. Identical folds/configurations isolate the addition's predictive effect, although adaptive stopping permits different retained tree counts. quote_signal's contribution must be assessed alongside its mixed fold behavior.

The earlier schema audit established that the assessment's supplied final-inference inputs include market_index and quote_signal. That is a schema-level availability assumption, not a source of model-selection row values. Both fields are absent from the fixed December scenario. The supplied instructions do not define units, provenance, publication lag, or when these signals become available relative to posting a rate. No executed analysis proves leakage, so neither is called leakage. A real deployment must verify timestamp alignment and availability before using them. For this assessment, provided inference values may be consumed; unavailable scenario values must never be forecast or invented. Native NaN handling and a separately validated feature-restricted December configuration address availability.

## Direct versus log1p target

The selected direct-model feature/weight choices for full inference and December form the controlled target-comparison scopes. The sequential experiment is intentionally limited, not a factorial search across every feature/weight/target interaction. MAE loss in log space changes the fitting emphasis; no smearing correction or post-hoc target calibration is learned. Predictions are transformed back with expm1. **Log models stop using custom DollarMAE**, which also transforms the evaluation truth/predictions back to dollars; stopping does not accidentally optimize relative/log-space error. Every reported MAE/RMSE/R2 is on the original posted-rate scale.

{markdown_table(selection_table(targets))}

{markdown_table(targets[per_fold_columns])}

Development target summary (all observations retained):

{markdown_table(target_summary)}

The previously generated target histogram and rate-per-mile figures are `reports/figures/eda_target_distribution.png` and `reports/figures/eda_rate_per_mile.png`. Large rates remain in all training and validation folds. RMSE remains sensitive to large errors; a log target is retained only if executed chronological metrics support it.

## Selection and methodological limits

Intermediate feature/weight scopes use an unweighted mean-MAE shortlist within 1% of the minimum, emphasizing September–October MAE, then mean MAE, RMSE, and MAE variability. Before final confirmation fits, shortlist adaptive primary and December-feasible contenders within 3% of their respective minimum mean MAE. For each finalist, derive a fixed iteration budget from its median retained tree count and evaluate that exact budget on all three folds without an eval_set. Choose among fixed-budget models using the 1% mean-MAE shortlist and the same September–October/MAE/RMSE/stability ordering. This separates the adaptive stopping winner from the configuration actually deployable without future labels. Missing-value/unseen-category compatibility and supplied-field availability are required. The frozen decision and actual fixed-budget metrics are in `docs/model_selection.md`; machine-readable settings are in `docs/final_model_config.json`.

Adaptive stopping, sequential feature/representation choice, and model selection reuse these historical validation windows. Fixed-budget confirmation removes evaluation labels from each fit but does not undo prior selection reuse. These are local model-selection estimates, not independent test scores or Spotter hidden metrics. Geographic compatibility is not evidence of accuracy for entirely new cities. No production training or final submission output occurs in these scripts.
"""
    (ROOT / "reports/feature_ablation.md").write_text(text)


def report_selection(results, all_results, config):
    primary, chart = config["primary"], config["december"]
    source = results.loc[results.model.eq(primary["source_experiment"])]
    fixed = results.loc[results.model.eq(primary["confirmation_model"])]
    chart_fixed = results.loc[results.model.eq(chart["confirmation_model"])]
    metrics = ["model","fold","MAE","RMSE","R2","tree_count"]
    current = all_results.groupby("model", sort=False)[["MAE","RMSE","R2"]].mean().reset_index()
    current = current.merge(all_results.loc[all_results.fold.eq("fold_3"),["model","MAE"]].rename(columns={"MAE":"Sep_Oct_MAE"}),on="model",validate="one_to_one")
    eligible = results.loc[~results.experiment_phase.eq("fixed_budget_confirmation")]
    diagnostic_path = ROOT / "reports/experiment_diagnostics.csv"
    diagnostic = pd.read_csv(diagnostic_path)
    diagnostic = diagnostic.loc[diagnostic.model.eq(primary["confirmation_model"])]
    robustness = pd.DataFrame(primary["robustness_checks"])
    fixed_candidates = results.loc[results.experiment_phase.eq("fixed_budget_confirmation")]
    direct_f = fixed_candidates.loc[fixed_candidates.model.eq("catboost_fixed_F_clean_direct")]
    log_f = fixed_candidates.loc[fixed_candidates.model.eq("catboost_fixed_F_clean_log1p")]
    weight_f = fixed_candidates.loc[fixed_candidates.model.eq("catboost_fixed_F_raw_direct")]
    target_note = (
        f"For the fixed full-feature comparison, clean/log MAE averages {log_f.MAE.mean():.6f} versus "
        f"{direct_f.MAE.mean():.6f} for clean/direct ({100*(1-log_f.MAE.mean()/direct_f.MAE.mean()):.3f}% lower). "
        f"Its mean RMSE is {log_f.RMSE.mean():.6f} versus {direct_f.RMSE.mean():.6f}, and "
        f"September–October MAE is {log_f.loc[log_f.fold.eq('fold_3'),'MAE'].iloc[0]:.6f} versus "
        f"{direct_f.loc[direct_f.fold.eq('fold_3'),'MAE'].iloc[0]:.6f}. The log target wins mean MAE "
        "but slightly worsens recent MAE and squared-error metrics; this is a modest tradeoff, not a win on every metric. "
        f"Full-feature raw/direct fixed-budget mean MAE is {weight_f.MAE.mean():.6f}. "
        "The cleaned representation preserves original missing/negative flags and performs better on the recent holdout."
    )
    params = json.dumps(primary["parameters"], indent=2)
    chart_params = json.dumps(chart["parameters"], indent=2)
    text = f"""# Frozen model selection before production training

**Decision: {primary['model_type']}, group {primary['feature_group']}, {primary['weight_representation']} weight representation, {primary['target_strategy']} target, {primary['parameters']['iterations']} fixed iterations.** The status is frozen_before_production_training. `docs/final_model_config.json` is the machine-readable configuration. No model has been fitted on all 48,000 development rows, saved for production, or used to generate final submission files at this checkpoint.

## Selection evidence and rule

All decisions use labeled development folds only. Intermediate comparisons use a 1% mean-MAE shortlist. Before fixed-budget results were observed, adaptive contenders within 3% of the respective primary/chart minimum mean MAE were chosen for confirmation. Each received a fixed iteration budget equal to its median retained trees. Final criterion is mean chronological MAE on these **fixed-budget** models: candidates within 1% of the minimum enter a shortlist ordered by September–October MAE, mean MAE, RMSE, and fold-MAE variability. Final candidates retain coordinates, handle missing measurements and unseen labels, and use realistically supplied fields. RMSE/R2 and instability are reported, not hidden by a single MAE. Source adaptive experiment: `{primary['source_experiment']}`; frozen confirmation model: `{primary['confirmation_model']}`.

Overall model comparison (unweighted fold means, plus primary holdout MAE):

{markdown_table(current)}

Eligible feature/weight/target candidates:

{markdown_table(selection_table(eligible))}

This configuration wins the stated fixed-budget development-only shortlist rule within the strongest model family. Earlier baselines establish the comparison floor; feature ablation and direct/log dollar-scale comparison support its representation. The early-stopping winner need not be the fixed-budget winner, because validation-adaptive tree counts cannot be used for production without future targets. All native CatBoost fits use fold-local categorical statistics; no global target encoding or learned preprocessing crosses fold boundaries. No final-validation row values, labels, distribution summaries, or prediction outcomes were consulted to choose the model.

{target_note}

The complete F bundle was the tested candidate, rather than an assertion that every component helps individually. The cumulative route addition worsened mean direct MAE before adding market signals; its marginal contribution conditional on the final log/market bundle was not separately isolated. quote_signal has mixed per-fold gains and uncertain upstream timing. The selected bundle wins among executed configurations, while these feature-specific limitations remain explicit.

## Exact final feature set and parameters

Feature group: **{primary['feature_group']}**. Exact ordered {len(primary['feature_columns'])}-feature list:

`{', '.join(primary['feature_columns'])}`

Categorical features: `{', '.join(primary['categorical_features'])}`. Target strategy: **{primary['target_strategy']}**. Reference date: **{primary['reference_date']}**. Both training and inference must use `build_model_features` with this frozen feature list. load_id and posted_rate are excluded. Missing categoricals are explicit strings; numeric NaN is native. The selected weight representation is intentional, not automatic row deletion or undocumented correction.

```json
{params}
```

Production fit must use **all 48,000 labeled rows with no eval_set and no early stopping**. The iteration count was fixed from the median of the three source model's retained tree counts; no unavailable future validation target is needed. Do not select a new iteration count during production. Direct predictions stay in dollars; log1p predictions use expm1. Frozen output policy: reject nonfinite outputs and enforce a $0.01 lower bound. That lower bound did not change any selected-model local evaluation or robustness prediction; original unmodified predictions were already positive.

Source adaptive-budget selection results:

{markdown_table(source[metrics])}

Actual frozen-budget confirmation results (trained on each historical training fold without eval_set):

{markdown_table(fixed[metrics])}

Unweighted mean fixed-budget MAE/RMSE/R2: **{fixed.MAE.mean():.6f} / {fixed.RMSE.mean():.6f} / {fixed.R2.mean():.6f}**. September–October fixed-budget MAE/RMSE/R2: **{primary['local_metrics_primary_holdout']['MAE']:.6f} / {primary['local_metrics_primary_holdout']['RMSE']:.6f} / {primary['local_metrics_primary_holdout']['R2']:.6f}**. These actual fixed-budget results are the recommended local metrics for subsequent training metadata and the report; source stopping-selected scores must be labeled separately.

## Missingness and unseen-category robustness

Actual local confirmation-subset metrics and coverage:

{markdown_table(diagnostic)}

Some chronological folds have no entirely new cities. Their zero-count subsets have no estimable accuracy; this does not demonstrate new-city accuracy. New-route subsets can contain familiar endpoint labels. Synthetic mechanical checks alter 128 September–October rows per case while preserving coordinates for new-name tests:

{markdown_table(robustness)}

All five cases produced finite strictly positive predictions using the selected full-size local holdout model. No accuracy is claimed for edited scenarios. Native CatBoost accepts unknown labels; sklearn's retained baseline preprocessing uses handle_unknown="ignore". Numeric latitude/longitude remain in the final model for geographic information beyond labels. No final unlabeled inference rows were used for these selection-stage checks.

## December inference configuration

Frozen December group: **{chart['feature_group']}**, weight **{chart['weight_representation']}**, target **{chart['target_strategy']}**, iterations **{chart['parameters']['iterations']}**. Its source adaptive experiment is `{chart['source_experiment']}`; frozen confirmation is `{chart['confirmation_model']}`. Exact features: `{', '.join(chart['feature_columns'])}`.

```json
{chart_params}
```

The chart configuration is restricted to freight/load/calendar/city/coordinate information. It does not require market_index or quote_signal. Lookup Lexington/Fort Wayne coordinates deterministically from the supplied development city mapping when the eventual chart pipeline is built; no future market or quote values may be invented. If primary and chart features/target match, the same final model can be reused; otherwise train the separately frozen chart configuration at the future production stage. Its development confirmation results are:

{markdown_table(chart_fixed[metrics])}

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
"""
    (ROOT / "docs/model_selection.md").write_text(text)


def main():
    check_protected()
    frame = pd.read_csv(ROOT / "data/train_test.csv")
    results = pd.read_csv(ROOT / "reports/ablation_results.csv", float_precision="round_trip")
    all_results = pd.read_csv(ROOT / "reports/model_comparison.csv", float_precision="round_trip")
    for name in ("best_iteration", "tree_count", "iterations_executed"):
        results[name] = results[name].astype("Int64")
        all_results[name] = all_results[name].astype("Int64")
    config = json.loads((ROOT / "docs/final_model_config.json").read_text())
    report_ablation(results, frame)
    report_selection(results, all_results, config)
    save_report(all_results, ROOT / "reports")
    check_protected()
    print("Updated feature ablation, model comparison, market summaries, and frozen model decision.")


if __name__ == "__main__":
    main()
