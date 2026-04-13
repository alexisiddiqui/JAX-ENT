from __future__ import annotations

import os
import traceback
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from ..config import PlotStyle
from ..plotting.mlm import (
    plot_coefficient_comparison,
    plot_correlations_bar_charts,
    plot_eta_and_ftest,
    plot_model_selection_performance,
    plot_partial_r2_comparison,
    plot_scatter_and_distributions,
    plot_stability_comparison,
)

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


_DEFAULT_SPLIT_COLORS = {
    "r": "fuchsia",
    "s": "black",
    "random": "fuchsia",
    "R3": "green",
    "sequence_cluster": "green",
    "Sp": "grey",
    "spatial": "grey",
    "_flat": "orange",
}
_DEFAULT_SPLIT_NAME_MAPPING = {
    "r": "Random",
    "s": "Sequence",
    "random": "Random",
    "R3": "Non-Redundant",
    "sequence_cluster": "Non-Redundant",
    "Sp": "Spatial",
    "spatial": "Spatial",
    "_flat": "Flat",
}
_DEFAULT_MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "H", "X", "d"]


def _resolve_style_maps(style: PlotStyle | None) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    ensemble_colors = dict(style.ensemble_colors) if style and style.ensemble_colors else {}
    split_colors = dict(_DEFAULT_SPLIT_COLORS)
    if style and style.split_type_colors:
        split_colors.update(style.split_type_colors)
    split_name_mapping = dict(_DEFAULT_SPLIT_NAME_MAPPING)
    if style and style.split_name_mapping:
        split_name_mapping.update(style.split_name_mapping)
    return ensemble_colors, split_colors, split_name_mapping


def prepare_metric_columns(df: pd.DataFrame, predictor_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    1. Create _transformed columns (log/exp/copy).
    2. Create _rank and _percentile columns (normalized within ensemble).
    """
    df = df.copy()
    stats_cols: list[str] = []

    for col in predictor_cols:
        trans_col = f"{col}_transformed"
        col_lower = col.lower()

        if "d_mse" in col_lower:
            mask = df[col].notna() & np.isfinite(df[col])
            df[trans_col] = np.nan
            if mask.any():
                df.loc[mask, trans_col] = np.exp(-df.loc[mask, col] * 2)
        elif "mse" in col_lower and "d_mse" not in col_lower:
            mask = (df[col] > 0) & df[col].notna() & np.isfinite(df[col])
            df[trans_col] = np.nan
            if mask.any():
                df.loc[mask, trans_col] = -np.log(df.loc[mask, col])
        else:
            df[trans_col] = df[col]

        rank_col = f"{col}_rank"
        pct_col = f"{col}_percentile"
        df[rank_col] = df.groupby("ensemble")[trans_col].transform(lambda x: x.rank(ascending=False))
        df[pct_col] = df.groupby("ensemble")[trans_col].transform(lambda x: x.rank(pct=True, ascending=True))
        stats_cols.append(pct_col)

    return df, stats_cols


def multiple_regression_analysis(
    data: pd.DataFrame,
    target_metric: str,
    predictor_cols: list[str],
    ensemble_colors: dict[str, str] | None = None,
) -> Optional[dict]:
    """Perform multiple linear regression with standardized predictors."""
    if len(data) < 5:
        return None

    if not HAS_STATSMODELS:
        raise RuntimeError("statsmodels is required for regression analysis")

    valid_predictors: list[str] = []
    for col in predictor_cols:
        if col not in data.columns:
            continue
        missing_pct = data[col].isna().mean()
        if missing_pct <= 0.2:
            valid_predictors.append(col)

    if not valid_predictors:
        return None

    df_reg = data[valid_predictors + [target_metric]].dropna()
    if len(df_reg) < 5:
        return None

    X = df_reg[valid_predictors].values
    y = df_reg[target_metric].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=valid_predictors, index=df_reg.index)
    X_scaled_with_intercept = sm.add_constant(X_scaled_df, has_constant="add")

    try:
        model_full = sm.OLS(y, X_scaled_with_intercept).fit()
    except Exception:
        return None

    results: dict[str, dict[str, float]] = {
        "model": {
            "r2_full": model_full.rsquared,
            "aic": model_full.aic,
            "bic": model_full.bic,
            "n_obs": len(y),
        }
    }

    for pred in valid_predictors:
        try:
            X_without = sm.add_constant(
                X_scaled_df.drop(columns=[pred]),
                has_constant="add",
            )
            model_without = sm.OLS(y, X_without).fit()
            partial_r2 = model_full.rsquared - model_without.rsquared
        except Exception:
            partial_r2 = np.nan

        try:
            X_univariate = sm.add_constant(X_scaled_df[[pred]], has_constant="add")
            model_univariate = sm.OLS(y, X_univariate).fit()
            univariate_r2 = model_univariate.rsquared
        except Exception:
            univariate_r2 = np.nan

        beta = float(model_full.params.get(pred, np.nan))
        se = float(model_full.bse.get(pred, np.nan))
        pvalue = float(model_full.pvalues.get(pred, np.nan))
        t_stat = float(model_full.tvalues.get(pred, np.nan))

        results[pred] = {
            "beta_standardized": beta,
            "se": se,
            "pvalue": pvalue,
            "partial_r2": partial_r2,
            "univariate_r2": univariate_r2,
            "t_statistic": t_stat,
        }

    return results


def _calculate_stability_metrics(
    data: pd.DataFrame,
    metric_name: str,
    grouping_col: str,
) -> Optional[dict[str, float]]:
    if metric_name not in data.columns or len(data) < 5:
        return None

    metric_data = data[metric_name].dropna()
    if len(metric_data) < 5:
        return None

    total_var = metric_data.var()

    if grouping_col in data.columns:
        split_groups = [
            data[data[grouping_col] == st][metric_name].dropna().values
            for st in data[grouping_col].dropna().unique()
            if len(data[data[grouping_col] == st][metric_name].dropna()) > 1
        ]

        if len(split_groups) > 1:
            split_means = np.array([g.mean() for g in split_groups])
            split_sizes = np.array([len(g) for g in split_groups])
            grand_mean = metric_data.mean()

            between_var = np.sum(split_sizes * (split_means - grand_mean) ** 2) / (len(split_means) - 1)
            within_var = np.mean([g.var() for g in split_groups])

            stability_index = 1.0 - (between_var / (between_var + within_var))
            cv_across_splits = split_means.std() / split_means.mean() if split_means.mean() != 0 else np.inf
            f_stat, p_value = stats.f_oneway(*split_groups)

            ss_between = np.sum(split_sizes * (split_means - grand_mean) ** 2)
            ss_total = np.sum((metric_data - grand_mean) ** 2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0

            return {
                "stability_index": stability_index,
                "var_ratio": between_var / (between_var + within_var),
                "cv_across_splits": cv_across_splits,
                "eta_squared": eta_squared,
                "f_statistic": f_stat,
                "p_value": p_value,
                "between_var": between_var,
                "within_var": within_var,
                "total_var": total_var,
                "n_obs": len(metric_data),
                "n_groups": len(split_groups),
            }

    return {
        "stability_index": 1.0,
        "var_ratio": 0.0,
        "cv_across_splits": 0.0,
        "eta_squared": 0.0,
        "f_statistic": np.nan,
        "p_value": np.nan,
        "between_var": np.nan,
        "within_var": np.nan,
        "total_var": total_var,
        "n_obs": len(metric_data),
        "n_groups": 1,
    }


def stability_analysis_by_ensemble(
    df: pd.DataFrame,
    target_metric: str,
    metric_cols: list[str],
) -> pd.DataFrame:
    """Compute stability metrics grouped by split_type within each ensemble."""
    rows: list[dict] = []
    for ensemble in sorted(df["ensemble"].dropna().unique()):
        df_ens = df[df["ensemble"] == ensemble].copy()
        if df_ens.empty:
            continue
        for metric in metric_cols:
            res = _calculate_stability_metrics(df_ens, metric, grouping_col="split_type")
            if not res:
                continue
            rows.append({"group": ensemble, "ensemble": ensemble, "metric": metric, **res})
    return pd.DataFrame(rows)


def stability_analysis_by_split(
    df: pd.DataFrame,
    target_metric: str,
    metric_cols: list[str],
    output_dir: str,
) -> pd.DataFrame:
    """Compute stability metrics grouped by ensemble within split_type and save CSVs."""
    rows: list[dict] = []
    if "split_type" not in df.columns:
        return pd.DataFrame(rows)

    for split_type in sorted(df["split_type"].dropna().unique()):
        df_split = df[df["split_type"] == split_type].copy()
        if df_split.empty:
            continue

        summary_data: list[dict] = []
        for metric in metric_cols:
            res = _calculate_stability_metrics(df_split, metric, grouping_col="ensemble")
            if not res:
                continue
            row = {"group": split_type, "split_type": split_type, "metric": metric, **res}
            rows.append(row)
            summary_data.append(
                {
                    "Metric": metric,
                    "Stability Index": res.get("stability_index"),
                    "CV": res.get("cv_across_splits"),
                    "F-statistic": res.get("f_statistic"),
                    "p-value": res.get("p_value"),
                    "Eta Squared": res.get("eta_squared"),
                    "Var Between": res.get("between_var"),
                    "Var Within": res.get("within_var"),
                }
            )

        if summary_data:
            df_sum = pd.DataFrame(summary_data)
            safe_name = "".join(c for c in str(split_type) if c.isalnum() or c in ("_", "-"))
            path = os.path.join(output_dir, f"summary_stability_swapped_{safe_name}.csv")
            df_sum.to_csv(path, index=False)

    return pd.DataFrame(rows)


def mixed_effects_analysis(
    df: pd.DataFrame,
    target_metric: str,
    metric_cols: list[str],
) -> pd.DataFrame | None:
    """Fit mixed-effects models by ensemble (if statsmodels available)."""
    if not HAS_STATSMODELS:
        warnings.warn("statsmodels not available. Mixed-effects models will be skipped.", stacklevel=2)
        return None

    grouping_var = None
    if "split_type" in df.columns and df["split_type"].nunique() > 1:
        grouping_var = "split_type"
    elif "ensemble" in df.columns and df["ensemble"].nunique() > 1:
        grouping_var = "ensemble"

    if grouping_var is None:
        return None

    rows: list[dict] = []
    for ensemble in sorted(df["ensemble"].dropna().unique()):
        df_ens = df[df["ensemble"] == ensemble].copy()
        if len(df_ens) < 10:
            continue

        for metric in metric_cols:
            if metric not in df_ens.columns:
                continue

            sub = df_ens[[target_metric, metric, grouping_var]].dropna().copy()
            if len(sub) < 5 or sub[grouping_var].nunique() < 2:
                continue

            sub[f"{metric}_scaled"] = (sub[metric] - sub[metric].mean()) / (sub[metric].std() + 1e-8)
            formula = f"{target_metric} ~ {metric}_scaled"

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    md = mixedlm(formula, sub, groups=sub[grouping_var])
                    mdf = md.fit(method="lbfgs", maxiter=100)
            except Exception:
                continue

            row = {
                "ensemble": ensemble,
                "metric": metric,
                "grouping_var": grouping_var,
                "fixed_effect": mdf.params.get(f"{metric}_scaled", np.nan),
                "fixed_effect_se": mdf.bse.get(f"{metric}_scaled", np.nan),
                "fixed_effect_pvalue": mdf.pvalues.get(f"{metric}_scaled", np.nan),
                "aic": mdf.aic,
                "bic": mdf.bic,
                "converged": mdf.converged,
                "n_obs": len(sub),
                "n_groups": sub[grouping_var].nunique(),
            }
            if mdf.converged:
                var_between = mdf.cov_re.values[0, 0] if mdf.cov_re.values[0, 0] > 0 else 0
                var_within = mdf.scale
                row["var_between"] = var_between
                row["var_within"] = var_within
                row["icc"] = var_between / (var_between + var_within) if (var_between + var_within) > 0 else 0
            rows.append(row)

    return pd.DataFrame(rows) if rows else None


def compute_model_selection_performance(
    df: pd.DataFrame,
    metric_cols: list[str],
    target_metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute model selection summary and by-split details."""
    df_plot = df.copy()
    group_cols = ["ensemble", "split_type", "split_idx"]

    if "loss_function" in df_plot.columns:
        group_cols.append("loss_function")
    if "bv_reg_function" in df_plot.columns:
        group_cols.append("bv_reg_function")

    if "loss_function" in df_plot.columns and "bv_reg_function" in df_plot.columns:
        df_plot["method_variant"] = (
            df_plot["loss_function"].astype(str) + "_" + df_plot["bv_reg_function"].astype(str)
        )
    elif "loss_function" in df_plot.columns:
        df_plot["method_variant"] = df_plot["loss_function"].astype(str)
    elif "bv_reg_function" in df_plot.columns:
        df_plot["method_variant"] = df_plot["bv_reg_function"].astype(str)
    else:
        df_plot["method_variant"] = "All"

    df_clean = df_plot.dropna(subset=[target_metric]).copy()

    selection_stats_list: list[pd.DataFrame] = []
    detailed_rows: list[pd.DataFrame] = []

    for metric in metric_cols:
        if metric not in df_clean.columns:
            continue

        is_loss = any(x in metric.lower() for x in ["loss", "mse", "mae", "error", "scale"])
        direction = "min" if is_loss else "max"

        try:
            if direction == "min":
                idx = df_clean.groupby(group_cols)[metric].idxmin()
            else:
                idx = df_clean.groupby(group_cols)[metric].idxmax()
            idx = idx.dropna()
        except Exception:
            continue

        selected_models = df_clean.loc[idx].copy()
        if selected_models.empty:
            continue

        agg_cols = ["ensemble", "split_type"]
        if "loss_function" in selected_models.columns:
            agg_cols.append("loss_function")
        if "bv_reg_function" in selected_models.columns:
            agg_cols.append("bv_reg_function")

        agg_dict: dict[str, list[str]] = {target_metric: ["mean", "std", "min", "count"]}
        for pred in metric_cols:
            agg_dict[pred] = ["mean", "std"]
            for suffix in ("_transformed", "_rank", "_percentile"):
                col = f"{pred}{suffix}"
                if col in selected_models.columns:
                    agg_dict[col] = ["mean", "std"]

        stats_df = selected_models.groupby(agg_cols).agg(agg_dict).reset_index()
        flat_cols: list[str] = []
        for c in stats_df.columns:
            if isinstance(c, tuple):
                flat_cols.append(f"{c[0]}_{c[1]}" if c[1] else c[0])
            else:
                flat_cols.append(c)
        stats_df.columns = flat_cols
        stats_df["score_metric"] = metric
        stats_df["direction"] = direction
        selection_stats_list.append(stats_df)

        selected_models["score_metric"] = metric
        selected_models["direction"] = direction
        keep_cols = group_cols + ["method_variant", target_metric, "score_metric", "direction"]
        for pred in metric_cols:
            keep_cols.append(pred)
            for suffix in ("_transformed", "_rank", "_percentile"):
                col = f"{pred}{suffix}"
                if col in selected_models.columns:
                    keep_cols.append(col)
        keep_cols = list(dict.fromkeys([c for c in keep_cols if c in selected_models.columns]))
        detailed_rows.append(selected_models[keep_cols])

    summary_df = pd.concat(selection_stats_list, ignore_index=True) if selection_stats_list else pd.DataFrame()
    by_split_df = pd.concat(detailed_rows, ignore_index=True) if detailed_rows else pd.DataFrame()
    return summary_df, by_split_df


def compute_correlations(
    df: pd.DataFrame,
    metric_cols: list[str],
    target_metric: str,
) -> pd.DataFrame:
    """Compute Pearson correlations grouped by split_type and ensemble."""
    rows: list[dict] = []

    group_cols = [c for c in ["split_type", "ensemble"] if c in df.columns]
    grouped = [((), df)] if not group_cols else df.groupby(group_cols, dropna=False)

    for keys, grp in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_cols, keys)) if group_cols else {}

        for col in metric_cols:
            if col not in grp.columns:
                continue
            if not np.issubdtype(grp[col].dtype, np.number):
                continue
            valid = grp[[col, target_metric]].dropna()
            if len(valid) < 2:
                continue
            corr, p_value = stats.pearsonr(valid[col], valid[target_metric])
            rows.append(
                {
                    "metric": col,
                    "correlation": corr,
                    "p_value": p_value,
                    "split_type": meta.get("split_type", "N/A"),
                    "ensemble": meta.get("ensemble", "N/A"),
                }
            )

    return pd.DataFrame(rows)


def _create_summary_table(
    regression_results: dict,
    stability_df: pd.DataFrame,
    ensemble: str,
    predictor_cols: list[str],
) -> pd.DataFrame:
    summary_data: list[dict] = []
    ens_stability = stability_df[stability_df["ensemble"] == ensemble] if not stability_df.empty else pd.DataFrame()

    for metric in predictor_cols:
        reg_res = regression_results.get(ensemble, {}).get(metric, {})
        if ens_stability.empty:
            stab_row = {}
        else:
            mdf = ens_stability[ens_stability["metric"] == metric]
            stab_row = mdf.iloc[0].to_dict() if not mdf.empty else {}

        summary_data.append(
            {
                "Metric": metric,
                "β (std)": f"{reg_res.get('beta_standardized', np.nan):.4f}",
                "p-value": f"{reg_res.get('pvalue', np.nan):.4e}",
                "Partial R²": f"{reg_res.get('partial_r2', np.nan):.4f}",
                "Univariate R²": f"{reg_res.get('univariate_r2', np.nan):.4f}",
                "Stability Idx": f"{stab_row.get('stability_index', np.nan):.4f}",
                "CV": f"{stab_row.get('cv_across_splits', np.nan):.4f}",
                "η²": f"{stab_row.get('eta_squared', np.nan):.4f}",
            }
        )

    return pd.DataFrame(summary_data)


def run_analysis_on_subset(
    df_clean: pd.DataFrame,
    target_metric: str,
    output_dir: str,
    style: PlotStyle | None,
    marker_list: list[str] | None = None,
) -> None:
    """Run the complete mixed linear model analysis pipeline for one subset."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Rows with valid target metric: {len(df_clean)}")
    print(f"Ensembles: {df_clean['ensemble'].unique()}")
    if "split_type" in df_clean.columns:
        print(f"Split types: {df_clean['split_type'].unique()}")
    if "bv_reg_function" in df_clean.columns:
        print(f"BV Reg functions: {df_clean['bv_reg_function'].unique()}")
    if "loss_function" in df_clean.columns:
        print(f"Loss functions: {df_clean['loss_function'].unique()}")

    error_metrics = [col for col in df_clean.columns if "mse" in col.lower()]
    work_metrics = [col for col in df_clean.columns if "work_" in col.lower()]
    cluster_metrics = [col for col in df_clean.columns if col.startswith("cluster_")]
    print("\nIdentified metric groups:")
    print(f"  Error metrics: {error_metrics}")
    print(f"  Work metrics: {work_metrics}")
    print(f"  Cluster metrics: {cluster_metrics}")

    exclude_cols = {target_metric, "convergence_value", "maxent_value", "split_idx", "bv_reg_value"}
    predictor_cols = [
        col
        for col in df_clean.select_dtypes(include=np.number).columns
        if col not in exclude_cols and df_clean[col].nunique() > 1
    ]
    print(f"\nSelected predictor metrics: {predictor_cols}")
    if not predictor_cols:
        print("\nNo predictors found, skipping analysis.")
        return

    print("\nApplying transformations and normalization...")
    df_clean, stats_predictors = prepare_metric_columns(df_clean, predictor_cols)

    print("\n2. Multiple Regression Analysis (on Percentile Data)")
    print("-" * 80)

    df_reg = df_clean.copy()
    reg_predictors = list(stats_predictors)
    if "split_type" in df_reg.columns and df_reg["split_type"].nunique() > 1:
        dummies = pd.get_dummies(df_reg["split_type"], prefix="split", drop_first=True).astype(int)
        df_reg = pd.concat([df_reg, dummies], axis=1)
        reg_predictors.extend(dummies.columns.tolist())
        print(f"Included split_type dummies in regression: {dummies.columns.tolist()}")

    regression_results: dict[str, dict] = {}
    for ensemble in sorted(df_clean["ensemble"].dropna().unique()):
        df_ens = df_reg[df_reg["ensemble"] == ensemble].copy()
        if df_ens.empty:
            continue
        print(f"\n{ensemble} (n={len(df_ens)}):")
        results = multiple_regression_analysis(df_ens, target_metric, reg_predictors)
        if results:
            regression_results[ensemble] = results
            print(f"  Full model R\u00b2: {results['model']['r2_full']:.4f}")
            for pred in reg_predictors:
                if pred in results:
                    print(
                        f"    {pred}: \u03b2={results[pred]['beta_standardized']:.4f}, "
                        f"partial R\u00b2={results[pred]['partial_r2']:.4f}, "
                        f"p={results[pred]['pvalue']:.4e}"
                    )

    print("\n3. Stability Analysis (Effect of Split Type within Ensembles)")
    print("-" * 80)
    stability_ensemble_df = stability_analysis_by_ensemble(df_clean, target_metric, stats_predictors)
    for ensemble in sorted(df_clean["ensemble"].dropna().unique()):
        ens_rows = stability_ensemble_df[stability_ensemble_df["ensemble"] == ensemble] if not stability_ensemble_df.empty else pd.DataFrame()
        if ens_rows.empty:
            continue
        print(f"\n{ensemble}:")
        for _, row in ens_rows.iterrows():
            print(
                f"  {row['metric']}: Stability Index={row.get('stability_index', float('nan')):.4f}, "
                f"CV={row.get('cv_across_splits', float('nan')):.4f}"
            )

    print("\n3b. Stability Analysis (Effect of Ensemble within Split Types)")
    print("-" * 80)
    stability_split_df = stability_analysis_by_split(df_clean, target_metric, stats_predictors, output_dir)
    if not stability_split_df.empty:
        for split_type in sorted(stability_split_df["split_type"].dropna().unique()):
            split_rows = stability_split_df[stability_split_df["split_type"] == split_type]
            print(f"\n{split_type}:")
            for _, row in split_rows.iterrows():
                print(
                    f"  {row['metric']}: Stability Index={row.get('stability_index', float('nan')):.4f}, "
                    f"CV={row.get('cv_across_splits', float('nan')):.4f}, "
                    f"F={row.get('f_statistic', float('nan')):.2f}"
                )
        print("\nSaving Swapped Stability Summaries...")
        for split_type in sorted(stability_split_df["split_type"].dropna().unique()):
            safe_name = "".join(c for c in str(split_type) if c.isalnum() or c in ("_", "-"))
            path = os.path.join(output_dir, f"summary_stability_swapped_{safe_name}.csv")
            if os.path.exists(path):
                print(f"  Saved {path}")

    grouping_var = None
    if "split_type" in df_clean.columns and df_clean["split_type"].nunique() > 1:
        grouping_var = "split_type"
    elif "ensemble" in df_clean.columns and df_clean["ensemble"].nunique() > 1:
        grouping_var = "ensemble"

    mixed_df = mixed_effects_analysis(df_clean, target_metric, stats_predictors)
    if grouping_var:
        print(f"\n4. Mixed-Effects Models (grouping by {grouping_var})")
        print("-" * 80)
        if mixed_df is not None and not mixed_df.empty:
            for ensemble in sorted(df_clean["ensemble"].dropna().unique()):
                ens_rows = mixed_df[mixed_df["ensemble"] == ensemble]
                if ens_rows.empty:
                    continue
                print(f"\n{ensemble}:")
                for _, row in ens_rows.iterrows():
                    print(
                        f"  {row['metric']}: ICC={row.get('icc', float('nan')):.4f}, "
                        f"fixed effect \u03b2={row.get('fixed_effect', float('nan')):.4f}"
                    )

    print("\n5. Creating Summary Tables")
    print("-" * 80)
    for ensemble in sorted(regression_results.keys()):
        summary_table = _create_summary_table(regression_results, stability_ensemble_df, ensemble, reg_predictors)
        print(f"\n{ensemble}:")
        print(summary_table.to_string(index=False))
        table_path = os.path.join(output_dir, f"summary_table_{ensemble}.csv")
        summary_table.to_csv(table_path, index=False)
        print(f"Saved to {table_path}")

    print("\n6. Creating Visualizations")
    print("-" * 80)

    ensemble_colors, split_colors, split_name_mapping = _resolve_style_maps(style)

    try:
        plot_coefficient_comparison(
            regression_results,
            output_dir,
            ensemble_colors=ensemble_colors,
            style=style,
            predictor_cols=reg_predictors,
        )
        plot_partial_r2_comparison(
            regression_results,
            output_dir,
            ensemble_colors=ensemble_colors,
            style=style,
            predictor_cols=reg_predictors,
        )

        plot_stability_comparison(
            stability_ensemble_df,
            output_dir,
            ensemble_colors=ensemble_colors,
            split_colors=split_colors,
            split_name_mapping=split_name_mapping,
            suffix="",
            style=style,
        )
        plot_eta_and_ftest(
            stability_ensemble_df,
            output_dir,
            ensemble_colors=ensemble_colors,
            split_colors=split_colors,
            split_name_mapping=split_name_mapping,
            suffix="",
            style=style,
        )

        if not stability_split_df.empty:
            plot_stability_comparison(
                stability_split_df,
                output_dir,
                ensemble_colors=ensemble_colors,
                split_colors=split_colors,
                split_name_mapping=split_name_mapping,
                suffix="_swapped",
                style=style,
            )
            plot_eta_and_ftest(
                stability_split_df,
                output_dir,
                ensemble_colors=ensemble_colors,
                split_colors=split_colors,
                split_name_mapping=split_name_mapping,
                suffix="_swapped",
                style=style,
            )

        markers = marker_list or _DEFAULT_MARKERS
        for metric in predictor_cols:
            plot_scatter_and_distributions(
                df_clean,
                metric,
                output_dir,
                ensemble_colors=ensemble_colors,
                style=style,
                marker_list=markers,
                target_metric=target_metric,
            )

        selection_summary_df, selection_by_split_df = compute_model_selection_performance(
            df_clean,
            predictor_cols,
            target_metric,
        )
        if not selection_summary_df.empty:
            selection_summary_df.to_csv(
                os.path.join(output_dir, "model_selection_performance_summary.csv"),
                index=False,
            )
        if not selection_by_split_df.empty:
            selection_by_split_df.to_csv(
                os.path.join(output_dir, "model_selection_performance_by_split.csv"),
                index=False,
            )
            for metric in predictor_cols:
                plot_model_selection_performance(
                    selection_by_split_df,
                    metric,
                    output_dir,
                    ensemble_colors=ensemble_colors,
                    split_colors=split_colors,
                    split_name_mapping=split_name_mapping,
                    style=style,
                    target_metric=target_metric,
                )

        corr_df = compute_correlations(df_clean, stats_predictors, target_metric)
        if not corr_df.empty:
            corr_df.to_csv(os.path.join(output_dir, "correlations_summary.csv"), index=False)
            for split_type in corr_df["split_type"].dropna().unique():
                plot_correlations_bar_charts(
                    corr_df,
                    split_type,
                    output_dir,
                    ensemble_colors=ensemble_colors,
                    style=style,
                )

    except Exception as e:
        print(f"Error creating visualizations: {e}")
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for ensemble in sorted(regression_results.keys()):
        print(f"\n{ensemble}:")
        print("  Predictive Metrics (Partial R\u00b2):")
        for metric in reg_predictors:
            if ensemble in regression_results and metric in regression_results[ensemble]:
                partial_r2 = regression_results[ensemble][metric].get("partial_r2", float("nan"))
                beta = regression_results[ensemble][metric].get("beta_standardized", float("nan"))
                pval = regression_results[ensemble][metric].get("pvalue", float("nan"))
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                print(f"    {metric}: R\u00b2={partial_r2:.4f}, \u03b2={beta:.4f} {sig}")
        print("  Stability (across splits):")
        if not stability_ensemble_df.empty:
            for metric in stats_predictors:
                ens_rows = stability_ensemble_df[
                    (stability_ensemble_df["ensemble"] == ensemble)
                    & (stability_ensemble_df["metric"] == metric)
                ]
                if not ens_rows.empty:
                    row = ens_rows.iloc[0]
                    stab_idx = row.get("stability_index", float("nan"))
                    cv = row.get("cv_across_splits", float("nan"))
                    print(f"    {metric}: Stability Index={stab_idx:.4f}, CV={cv:.4f}")

    print("\n" + "=" * 80)
    print(f"All results saved to: {output_dir}")
    print("=" * 80)


__all__ = [
    "prepare_metric_columns",
    "multiple_regression_analysis",
    "stability_analysis_by_ensemble",
    "stability_analysis_by_split",
    "mixed_effects_analysis",
    "compute_model_selection_performance",
    "compute_correlations",
    "run_analysis_on_subset",
]
