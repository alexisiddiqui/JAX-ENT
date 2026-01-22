"""
MoPrP Linear Effects Modelling Analysis
========================================

Comprehensive linear modelling framework for analyzing optimization metrics:
- Multiple regression: recovery = β0 + β1*MSE_metrics + β2*Work_metrics + ε
- Stability analysis across splits and replicates
- Mixed-effects models with random effects for grouping variables
- ICC (intra-class correlation) quantification
- Standardized coefficient plots and comparative visualizations

This script processes output from MoPrP_Score_Models.py and provides
insights into metric predictive utility and stability across experimental conditions.
"""

import argparse
import os
import warnings
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# For mixed-effects models
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels not available. Mixed-effects models will be skipped.")

# Set publication-ready style
sns.set_style("ticks")
sns.set_context(
    "paper",
    rc={
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    },
)

# Color schemes - Updated for MoPrP ensembles
ensemble_colors = {
    "AF2_MSAss": "RoyalBlue",      # Blue
    "AF2_filtered": "Cyan",         # Cyan
}

split_colors = {
    "r": "fuchsia",
    "s": "black",
    "random": "fuchsia",
    "R3": "green",
    "sequence_cluster": "green",
    "Sp": "grey",
    "spatial": "grey",
    "_flat": "orange",
}

split_name_mapping = {
    "r": "Random",
    "s": "Sequence",
    "random": "Random",
    "R3": "Non-Redundant",
    "sequence_cluster": "Non-Redundant",
    "Sp": "Spatial",
    "spatial": "Spatial",
    "_flat": "Flat",
}

# Marker list for split indices
MARKER_LIST = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'X', 'd']

# Metric visualization styles (MSE vs Work)
MSE_HATCH = "///"
WORK_HATCH = "\\\\\\\\"
MSE_COLOR = "white"
MSE_TEXT = "grey"
WORK_COLOR = "xkcd:neon yellow"
WORK_TEXT = "black"
MSE_ALPHA = 0.9
WORK_ALPHA = 0.7


def prepare_metric_columns(df: pd.DataFrame, predictor_cols: list) -> Tuple[pd.DataFrame, list]:
    """
    1. Create _transformed columns (log/exp/copy).
    2. Create _rank and _percentile columns (normalized within ensemble).
    Returns updated df and list of percentile column names for stats analysis.
    """
    df = df.copy()
    stats_cols = []
    
    for col in predictor_cols:
        # 1. Transformation
        trans_col = f"{col}_transformed"
        col_lower = col.lower()
        
        if 'd_mse' in col_lower:
             # exp(-d_mse * 2)
             mask = df[col].notna() & np.isfinite(df[col])
             df[trans_col] = np.nan
             if mask.any():
                 df.loc[mask, trans_col] = np.exp(-df.loc[mask, col] * 2)
                 print(f"  Transformed {col} -> {trans_col} using exp(-x*2)")
        elif 'mse' in col_lower and 'd_mse' not in col_lower:
             # -log(mse)
             mask = (df[col] > 0) & df[col].notna() & np.isfinite(df[col])
             df[trans_col] = np.nan
             if mask.any():
                 df.loc[mask, trans_col] = -np.log(df.loc[mask, col])
                 print(f"  Transformed {col} -> {trans_col} using -log(x)")
        else:
             df[trans_col] = df[col]
             print(f"  Copied {col} -> {trans_col}")
             
        # 2. Rank and Percentile (Grouped by ensemble for normalization)
        rank_col = f"{col}_rank"
        pct_col = f"{col}_percentile"
        
        # Rank: 1 = Highest Transformed Value (Best if transform aligns with quality)
        df[rank_col] = df.groupby('ensemble')[trans_col].transform(lambda x: x.rank(ascending=False))
        
        # Percentile: 1.0 = Highest Transformed Value
        df[pct_col] = df.groupby('ensemble')[trans_col].transform(lambda x: x.rank(pct=True, ascending=True))
        
        stats_cols.append(pct_col)
        
    return df, stats_cols


def run_analysis_on_subset(
    df_clean: pd.DataFrame,
    output_dir: str,
    target_metric: str,
):
    """
    Run the full suite of analyses on a provided subset of data.
    """
    print(f"Rows with valid target metric: {len(df_clean)}")
    print(f"Ensembles: {df_clean['ensemble'].unique()}")
    print(f"Split types: {df_clean['split_type'].unique()}")
    if 'bv_reg_function' in df_clean.columns:
        print(f"BV Reg functions: {df_clean['bv_reg_function'].unique()}")
    print(f"Loss functions: {df_clean['loss_function'].unique()}")

    # Identify metric groups
    error_metrics = [col for col in df_clean.columns if "mse" in col.lower()]
    work_metrics = [col for col in df_clean.columns if "work_" in col.lower()]
    cluster_metrics = [col for col in df_clean.columns if col.startswith("cluster_")]

    print(f"\nIdentified metric groups:")
    print(f"  Error metrics: {error_metrics}")
    print(f"  Work metrics: {work_metrics}")
    print(f"  Cluster metrics: {cluster_metrics}")

    # Select predictors
    exclude_cols = {target_metric, "convergence_value", "maxent_value", "split_idx", "bv_reg_value"}
    available_numeric = df_clean.select_dtypes(include=np.number).columns
    predictor_cols = [
        col for col in available_numeric
        if col not in exclude_cols and df_clean[col].nunique() > 1
    ]

    print(f"\nSelected predictor metrics: {predictor_cols}")

    if not predictor_cols:
        print("\nNo predictors found, skipping analysis.")
        return

    # Apply Transformations and Normalization
    print("\nApplying transformations and normalization...")
    df_clean, stats_predictors = prepare_metric_columns(df_clean, predictor_cols)

    # Regression analysis (Using Percentile Transformed Data)
    print("\n2. Multiple Regression Analysis (on Percentile Data)")
    print("-" * 80)

    # Prepare data for regression with split_type dummies
    df_reg = df_clean.copy()
    reg_predictors = list(stats_predictors) # Use percentiles
    
    if "split_type" in df_reg.columns and df_reg["split_type"].nunique() > 1:
        dummies = pd.get_dummies(df_reg["split_type"], prefix="split", drop_first=True)
        # Ensure boolean/int
        dummies = dummies.astype(int) 
        df_reg = pd.concat([df_reg, dummies], axis=1)
        reg_predictors.extend(dummies.columns.tolist())
        print(f"Included split_type dummies in regression: {dummies.columns.tolist()}")

    regression_results = {}
    for ensemble in sorted(df_clean["ensemble"].unique()):
        df_ens = df_reg[df_reg["ensemble"] == ensemble].copy()
        if df_ens.empty:
            continue
        print(f"\n{ensemble} (n={len(df_ens)}):")

        results = multiple_regression_analysis(df_ens, target_metric, reg_predictors)
        if results:
            regression_results[ensemble] = results
            print(f"  Full model R²: {results['model']['r2_full']:.4f}")
            for pred in reg_predictors:
                if pred in results:
                    print(
                        f"    {pred}: β={results[pred]['beta_standardized']:.4f}, "
                        f"partial R²={results[pred]['partial_r2']:.4f}, "
                        f"p={results[pred]['pvalue']:.4e}"
                    )

    # Stability analysis (Using Percentile Data)
    print("\n3. Stability Analysis (Effect of Split Type within Ensembles)")
    print("-" * 80)

    stability_results = {}
    for ensemble in sorted(df_clean["ensemble"].unique()):
        df_ens = df_clean[df_clean["ensemble"] == ensemble].copy()
        if df_ens.empty:
            continue
        print(f"\n{ensemble}:")

        stability_results[ensemble] = {}
        for metric in stats_predictors:
            if metric not in df_ens.columns:
                continue

            stab_res = calculate_stability_metrics(df_ens, metric, grouping_col="split_type")
            if stab_res:
                stability_results[ensemble][metric] = stab_res
                print(
                    f"  {metric}: Stability Index={stab_res['stability_index']:.4f}, "
                    f"CV={stab_res['cv_across_splits']:.4f}"
                )

    # Swapped Stability analysis (Effect of Ensemble within Split Types)
    print("\n3b. Stability Analysis (Effect of Ensemble within Split Types)")
    print("-" * 80)

    stability_results_swapped = {}
    if "split_type" in df_clean.columns:
        for split_type in sorted(df_clean["split_type"].unique()):
            df_split = df_clean[df_clean["split_type"] == split_type].copy()
            if df_split.empty: continue
            
            print(f"\n{split_type}:")
            stability_results_swapped[split_type] = {}
            
            for metric in stats_predictors:
                if metric not in df_split.columns: continue
                
                # Group by ensemble now
                stab_res = calculate_stability_metrics(df_split, metric, grouping_col="ensemble")
                if stab_res:
                    stability_results_swapped[split_type][metric] = stab_res
                    print(
                        f"  {metric}: Stability Index={stab_res['stability_index']:.4f}, "
                        f"CV={stab_res['cv_across_splits']:.4f}, "
                        f"F={stab_res['f_statistic']:.2f}"
                    )

    # Save swapped stability summaries
    if stability_results_swapped:
        print("\nSaving Swapped Stability Summaries...")
        for split_type, metrics_res in stability_results_swapped.items():
            summary_data = []
            for metric, res in metrics_res.items():
                row = {
                    "Metric": metric,
                    "Stability Index": res.get('stability_index'),
                    "CV": res.get('cv_across_splits'),
                    "F-statistic": res.get('f_statistic'),
                    "p-value": res.get('p_value'),
                    "Eta Squared": res.get('eta_squared'),
                    "Var Between": res.get('between_var'),
                    "Var Within": res.get('within_var')
                }
                summary_data.append(row)
            
            if summary_data:
                df_sum = pd.DataFrame(summary_data)
                safe_name = "".join(c for c in str(split_type) if c.isalnum() or c in ('_', '-'))
                path = os.path.join(output_dir, f"summary_stability_swapped_{safe_name}.csv")
                df_sum.to_csv(path, index=False)
                print(f"  Saved {path}")

    # Determine grouping variable for mixed-effects models
    grouping_var = None
    if "split_type" in df_clean.columns and df_clean["split_type"].nunique() > 1:
        grouping_var = "split_type"
    elif "ensemble" in df_clean.columns and df_clean["ensemble"].nunique() > 1:
        grouping_var = "ensemble"

    if grouping_var:
        print(f"\n4. Mixed-Effects Models (grouping by {grouping_var})")
        print("-" * 80)

        mixed_results = {}
        for ensemble in sorted(df_clean["ensemble"].unique()):
            df_ens = df_clean[df_clean["ensemble"] == ensemble].copy()
            if len(df_ens) < 10:
                continue

            print(f"\n{ensemble}:")
            mixed_results[ensemble] = {}

            for metric in stats_predictors:
                if metric not in df_ens.columns:
                    continue

                try:
                    mixed_res = fit_mixed_effects_model(
                        df_ens, target_metric, metric, grouping_var
                    )
                    if mixed_res:
                        mixed_results[ensemble][metric] = mixed_res
                        print(
                            f"  {metric}: ICC={mixed_res.get('icc', np.nan):.4f}, "
                            f"fixed effect β={mixed_res.get('fixed_effect', np.nan):.4f}"
                        )
                except Exception as e:
                    print(f"  {metric}: Mixed model failed ({str(e)[:50]})")

    # Create summary tables
    print("\n5. Creating Summary Tables")
    print("-" * 80)

    for ensemble in sorted(regression_results.keys()):
        summary_table = create_summary_table(
            regression_results, stability_results, ensemble, reg_predictors
        )
        print(f"\n{ensemble}:")
        print(summary_table.to_string(index=False))

        table_path = os.path.join(output_dir, f"summary_table_{ensemble}.csv")
        summary_table.to_csv(table_path, index=False)
        print(f"Saved to {table_path}")

    # Create visualizations
    print("\n6. Creating Visualizations")
    print("-" * 80)

    try:
        # Regression plots use reg_predictors (includes dummies)
        plot_coefficient_comparison(regression_results, reg_predictors, output_dir)
        plot_partial_r2_comparison(regression_results, reg_predictors, output_dir)
        
        # Stability plots use stats_predictors (percentiles)
        plot_stability_comparison(stability_results, stats_predictors, output_dir, 
                                  xlabel="Ensemble", title_suffix="Across Splits")
        plot_eta_and_ftest(stability_results, stats_predictors, output_dir,
                           xlabel="Ensemble", title_suffix="Split Type Effect")
        
        # Swapped stability plots
        if stability_results_swapped:
            plot_stability_comparison(stability_results_swapped, stats_predictors, output_dir, 
                                      xlabel="Split Type", title_suffix="Across Ensembles", filename_suffix="_swapped")
            plot_eta_and_ftest(stability_results_swapped, stats_predictors, output_dir,
                               xlabel="Split Type", title_suffix="Ensemble Effect", filename_suffix="_swapped")

        # Scatter plots use original predictor_cols for interpretability
        plot_scatter_and_distributions(df_clean, target_metric, predictor_cols, output_dir)
        
        # Model selection uses original predictor_cols to find best, but reports transformed/ranked
        plot_model_selection_performance(df_clean, target_metric, predictor_cols, output_dir)
        
        # Calculate and plot correlations (using percentiles for stats consistency?)
        # Usually correlations are better on the transformed/percentile data if relationships are non-linear
        corr_df = calculate_and_save_correlations(df_clean, target_metric, stats_predictors, output_dir)
        plot_correlations_bar_charts(corr_df, output_dir)

    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for ensemble in sorted(regression_results.keys()):
        print(f"\n{ensemble}:")
        print("  Predictive Metrics (Partial R²):")
        for metric in reg_predictors:
            if ensemble in regression_results and metric in regression_results[ensemble]:
                partial_r2 = regression_results[ensemble][metric].get("partial_r2", np.nan)
                beta = regression_results[ensemble][metric].get("beta_standardized", np.nan)
                pval = regression_results[ensemble][metric].get("pvalue", np.nan)
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                print(f"    {metric}: R²={partial_r2:.4f}, β={beta:.4f} {sig}")

        print("  Stability (across splits):")
        for metric in stats_predictors:
            if ensemble in stability_results and metric in stability_results[ensemble]:
                stab_idx = stability_results[ensemble][metric].get("stability_index", np.nan)
                cv = stability_results[ensemble][metric].get("cv_across_splits", np.nan)
                print(f"    {metric}: Stability Index={stab_idx:.4f}, CV={cv:.4f}")

    print("\n" + "=" * 80)
    print(f"All results saved to: {output_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Perform comprehensive linear effects modelling on MoPrP optimization scores."
    )
    parser.add_argument(
        "--scores-csv-path",
        default="../fitting/jaxENT/_scores_processed_optimise_quick_test__20251117_144611/model_scores.csv",
        help="Path to the CSV file containing model scores (output from MoPrP_Score_Models.py).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for analysis results. If omitted, creates a subdirectory alongside the scores CSV file.",
    )
    parser.add_argument(
        "--target-metric",
        default="recovery_percent",
        help="The metric to use as the dependent variable in the linear model.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        default=False,
        help="Interpret provided paths as absolute paths.",
    )
    parser.add_argument(
        "--filter-mode",
        choices=["both", "unfiltered", "filtered"],
        default="both",
        help="Analysis mode: 'unfiltered' (all data), 'filtered' (best convergence only), or 'both' (default).",
    )
    parser.add_argument(
        "--analyze-subsets",
        action="store_true",
        default=False,
        help="Analyze subsets (e.g. per loss function) in addition to whole dataset.",
    )
    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.absolute_paths:
        scores_csv_path = args.scores_csv_path
    else:
        scores_csv_path = os.path.abspath(os.path.join(script_dir, args.scores_csv_path))

    if args.output_dir:
        if args.absolute_paths:
            base_output_dir = args.output_dir
        else:
            base_output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    else:
        # Create output directory as a sibling to the scores CSV file's parent directory
        scores_parent_dir = os.path.dirname(scores_csv_path)
        scores_parent_basename = os.path.basename(scores_parent_dir)
        # If the scores are in a _scores_* directory, create analysis dir as sibling
        if scores_parent_basename.startswith("_scores_"):
            grandparent_dir = os.path.dirname(scores_parent_dir)
            base_output_dir = os.path.join(grandparent_dir, f"_analysis_{scores_parent_basename}")
        else:
            # Otherwise, create it inside the same directory
            base_name = os.path.basename(scores_csv_path).replace(".csv", "")
            base_output_dir = os.path.join(scores_parent_dir, f"_analysis_{base_name}")

    print("=" * 80)
    print("MoPrP Linear Effects Modelling Analysis")
    print("=" * 80)
    print(f"Resolved scores_csv_path: {scores_csv_path}")
    print(f"Base output dir: {base_output_dir}")
    print(f"Target metric: {args.target_metric}")
    print(f"Filter mode: {args.filter_mode}")
    print("-" * 80)

    # Load scores data
    if not os.path.exists(scores_csv_path):
        raise FileNotFoundError(f"Scores CSV not found: {scores_csv_path}")

    df_master = pd.read_csv(scores_csv_path)
    print(f"\nLoaded {len(df_master)} rows from {scores_csv_path}")
    print("\nDataFrame info:")
    print(df_master.info())
    print("\nDataFrame head:")
    print(df_master.head())
    print("-" * 80)

    # Determine modes to run
    modes = []
    if args.filter_mode == "both":
        modes = [False, True]
    elif args.filter_mode == "filtered":
        modes = [True]
    else:
        modes = [False]

    for filter_best_convergence in modes:
        print("\n" + "=" * 80)
        print(f"RUNNING ANALYSIS: Filter Best Convergence = {filter_best_convergence}")
        print("=" * 80)

        # Setup output directory for this run
        run_output_dir = base_output_dir
        if filter_best_convergence:
            run_output_dir = run_output_dir.rstrip(os.sep) + "_filtered"
        
        os.makedirs(run_output_dir, exist_ok=True)
        print(f"Output directory: {run_output_dir}")

        df = df_master.copy()

        if filter_best_convergence:
            print("\nFiltering for best convergence threshold (lowest val_loss)...")
            # Identify grouping columns
            group_cols = ['ensemble', 'split_type', 'split_idx', 'maxent_value']
            if 'bv_reg_value' in df.columns:
                group_cols.append('bv_reg_value')
            if 'loss_function' in df.columns:
                group_cols.append('loss_function')
            if 'bv_reg_function' in df.columns:
                group_cols.append('bv_reg_function')
            
            # Check if we have all necessary columns
            missing_cols = [c for c in group_cols + ['val_loss'] if c not in df.columns]
            
            if missing_cols:
                print(f"  WARNING: Cannot filter. Missing columns: {missing_cols}")
            else:
                original_len = len(df)
                
                # Ensure val_loss is numeric to guarantee correct sorting
                df['val_loss'] = pd.to_numeric(df['val_loss'], errors='coerce')
                
                # Sort by val_loss ascending
                df = df.sort_values("val_loss", ascending=True, na_position='last')
                
                # Keep first (lowest val_loss) for each group
                df = df.drop_duplicates(subset=group_cols, keep="first")
                
                # Restore index order
                df = df.sort_index()
                print(f"  Retained {len(df)} rows (dropped {original_len - len(df)}).")
                print("-" * 80)

        # Data preparation
        print("\n1. Data Preparation")
        print("-" * 80)

        df_clean_base = df.dropna(subset=[args.target_metric]).copy()
        if df_clean_base.empty:
            print(f"ERROR: No data remaining after dropping NaNs in '{args.target_metric}'")
            continue

        # Define datasets to analyze
        datasets_to_analyze = {"whole_dataset": df_clean_base}
        
        if args.analyze_subsets:
            # Determine columns to split by
            split_cols = []
            if 'loss_function' in df_clean_base.columns:
                split_cols.append('loss_function')
            if 'bv_reg_function' in df_clean_base.columns:
                split_cols.append('bv_reg_function')
                
            if split_cols:
                print(f"Splitting dataset by: {split_cols}")
                # Get unique combinations
                combinations = df_clean_base[split_cols].drop_duplicates()
                
                for _, row in combinations.iterrows():
                    # Create a mask for this combination
                    mask = pd.Series(True, index=df_clean_base.index)
                    name_parts = []
                    
                    for col in split_cols:
                        val = row[col]
                        mask &= (df_clean_base[col] == val)
                        name_parts.append(str(val))
                    
                    subset_name = "_".join(name_parts)
                    datasets_to_analyze[subset_name] = df_clean_base[mask]

        for name, df_subset in datasets_to_analyze.items():
            print("\n" + "#" * 80)
            print(f"# Analyzing subset: {name}")
            print("#" * 80)

            # Sanitize name for directory
            safe_name = "".join(c for c in name if c.isalnum() or c in ('_', '-')).rstrip()
            subset_output_dir = os.path.join(run_output_dir, safe_name)
            os.makedirs(subset_output_dir, exist_ok=True)

            if df_subset.empty:
                print("Subset is empty, skipping.")
                continue

            run_analysis_on_subset(df_subset.copy(), subset_output_dir, args.target_metric)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print(f"All results saved in subdirectories of: {base_output_dir}")
    print("=" * 80)


def multiple_regression_analysis(
    data: pd.DataFrame,
    target_metric: str,
    predictor_cols: list,
) -> Optional[Dict]:
    """
    Perform multiple linear regression with standardized predictors.

    Returns:
    --------
    results_dict : dict
        Dictionary containing regression statistics for each predictor
    """
    if len(data) < 5:
        print("Insufficient data for regression")
        return None

    # Filter predictors to those with sufficient data
    # Drop predictors that have > 20% missing values to preserve rows
    valid_predictors = []
    for col in predictor_cols:
        if col not in data.columns:
            continue
        missing_pct = data[col].isna().mean()
        if missing_pct > 0.2:
            print(f"  Dropping predictor '{col}' (missing {missing_pct:.1%})")
        else:
            valid_predictors.append(col)
            
    if not valid_predictors:
        print("No valid predictors remaining")
        return None

    # Prepare data with valid predictors
    df_reg = data[valid_predictors + [target_metric]].dropna()
    
    if len(df_reg) < 5:
        print(f"Insufficient valid data for regression (n={len(df_reg)})")
        # Diagnostic
        for col in valid_predictors:
            print(f"  {col}: {data[col].count()} valid")
        return None

    X = df_reg[valid_predictors].values
    y = df_reg[target_metric].values

    # Standardize predictors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_with_intercept = sm.add_constant(X_scaled)

    # Fit full model
    try:
        model_full = sm.OLS(y, X_scaled_with_intercept).fit()
    except Exception as e:
        print(f"Regression failed: {e}")
        return None

    # Calculate results for each predictor
    results = {"model": {"r2_full": model_full.rsquared, "aic": model_full.aic, "bic": model_full.bic, "n_obs": len(y)}}

    for i, pred in enumerate(valid_predictors):
        # Partial R²: contribution of this predictor beyond all others
        try:
            model_without = sm.OLS(y, sm.add_constant(np.delete(X_scaled, i, axis=1))).fit()
            partial_r2 = model_full.rsquared - model_without.rsquared
        except:
            partial_r2 = np.nan

        # Univariate R² for comparison
        try:
            model_univariate = sm.OLS(y, sm.add_constant(X_scaled[:, i])).fit()
            univariate_r2 = model_univariate.rsquared
        except:
            univariate_r2 = np.nan

        results[pred] = {
            "beta_standardized": model_full.params[i + 1],
            "se": model_full.bse[i + 1],
            "pvalue": model_full.pvalues[i + 1],
            "partial_r2": partial_r2,
            "univariate_r2": univariate_r2,
            "t_statistic": model_full.tvalues[i + 1],
        }

    return results


def calculate_stability_metrics(
    data: pd.DataFrame, 
    metric_name: str, 
    grouping_col: str = "split_type"
) -> Optional[Dict]:
    """
    Calculate stability metrics using variance decomposition across groups.

    Parameters:
    -----------
    data : pd.DataFrame
        Analysis data
    metric_name : str
        Name of metric to analyze
    grouping_col : str
        Column to use for grouping (e.g., 'split_type' or 'ensemble')

    Returns:
    --------
    results_dict : dict
        Dictionary containing stability statistics
    """
    if metric_name not in data.columns or len(data) < 5:
        return None

    metric_data = data[metric_name].dropna()
    if len(metric_data) < 5:
        return None

    # Calculate variance components
    total_var = metric_data.var()

    # Variance by group (if available)
    if grouping_col in data.columns:
        split_groups = [
            data[data[grouping_col] == st][metric_name].dropna().values
            for st in data[grouping_col].unique()
            if len(data[data[grouping_col] == st][metric_name].dropna()) > 1
        ]

        if len(split_groups) > 1:
            # Between-group variance
            split_means = np.array([g.mean() for g in split_groups])
            split_sizes = np.array([len(g) for g in split_groups])
            grand_mean = metric_data.mean()

            between_var = np.sum(split_sizes * (split_means - grand_mean) ** 2) / (
                len(split_means) - 1
            )
            within_var = np.mean([g.var() for g in split_groups])

            # Stability index: 1 - (between_var / total_var)
            stability_index = 1.0 - (between_var / (between_var + within_var))

            # Coefficient of variation across groups
            cv_across_splits = split_means.std() / split_means.mean() if split_means.mean() != 0 else np.inf

            # ANOVA F-test
            f_stat, p_value = stats.f_oneway(*split_groups)

            # Eta-squared
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

    # Fallback for no grouping information
    return {
        "stability_index": 1.0,
        "var_ratio": 0.0,
        "cv_across_splits": 0.0,
        "eta_squared": 0.0,
        "f_statistic": np.nan,
        "p_value": np.nan,
        "total_var": total_var,
        "n_obs": len(metric_data),
        "n_groups": 1,
    }


def fit_mixed_effects_model(
    data: pd.DataFrame,
    target_metric: str,
    predictor_metric: str,
    grouping_var: str,
) -> Optional[Dict]:
    """
    Fit mixed-effects model with random intercepts.

    Parameters:
    -----------
    data : pd.DataFrame
        Data for one ensemble
    target_metric : str
        Dependent variable
    predictor_metric : str
        Independent variable
    grouping_var : str
        Variable for random intercepts

    Returns:
    --------
    results_dict : dict or None
        Dictionary containing model statistics
    """
    if not HAS_STATSMODELS:
        return None

    df = data[[target_metric, predictor_metric, grouping_var]].dropna().copy()
    if len(df) < 5 or df[grouping_var].nunique() < 2:
        return None

    # Standardize predictor
    df[f"{predictor_metric}_scaled"] = (
        df[predictor_metric] - df[predictor_metric].mean()
    ) / (df[predictor_metric].std() + 1e-8)

    try:
        formula = f"{target_metric} ~ {predictor_metric}_scaled"

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            md = mixedlm(formula, df, groups=df[grouping_var])
            mdf = md.fit(method="lbfgs", maxiter=100)

        # Extract results
        results = {
            "fixed_effect": mdf.params[f"{predictor_metric}_scaled"],
            "fixed_effect_se": mdf.bse[f"{predictor_metric}_scaled"],
            "fixed_effect_pvalue": mdf.pvalues[f"{predictor_metric}_scaled"],
            "aic": mdf.aic,
            "bic": mdf.bic,
            "converged": mdf.converged,
            "n_obs": len(df),
            "n_groups": df[grouping_var].nunique(),
        }

        # Calculate ICC if model converged
        if mdf.converged:
            var_between = mdf.cov_re.values[0, 0] if mdf.cov_re.values[0, 0] > 0 else 0
            var_within = mdf.scale
            icc = var_between / (var_between + var_within) if (var_between + var_within) > 0 else 0
            results["icc"] = icc
            results["var_between"] = var_between
            results["var_within"] = var_within

        return results

    except Exception as e:
        return None


def create_summary_table(
    regression_results: Dict,
    stability_results: Dict,
    ensemble: str,
    predictor_cols: list,
) -> pd.DataFrame:
    """
    Create summary table for an ensemble.
    """
    summary_data = []

    for metric in predictor_cols:
        reg_res = regression_results.get(ensemble, {}).get(metric, {})
        stab_res = stability_results.get(ensemble, {}).get(metric, {})

        row = {
            "Metric": metric,
            "β (std)": f"{reg_res.get('beta_standardized', np.nan):.4f}",
            "p-value": f"{reg_res.get('pvalue', np.nan):.4e}",
            "Partial R²": f"{reg_res.get('partial_r2', np.nan):.4f}",
            "Univariate R²": f"{reg_res.get('univariate_r2', np.nan):.4f}",
            "Stability Idx": f"{stab_res.get('stability_index', np.nan):.4f}",
            "CV": f"{stab_res.get('cv_across_splits', np.nan):.4f}",
            "η²": f"{stab_res.get('eta_squared', np.nan):.4f}",
        }
        summary_data.append(row)

    return pd.DataFrame(summary_data)


def plot_coefficient_comparison(
    regression_results: Dict, predictor_cols: list, output_dir: str
) -> None:
    """Create coefficient comparison plot across ensembles."""
    ensembles = sorted(regression_results.keys())
    if not ensembles:
        print("No regression results to plot")
        return

    fig, axes = plt.subplots(
        1, len(ensembles), figsize=(7 * len(ensembles), 6), sharey=True
    )
    if len(ensembles) == 1:
        axes = [axes]

    for idx, ensemble in enumerate(ensembles):
        ax = axes[idx]
        results = regression_results.get(ensemble, {})

        betas = [
            results.get(m, {}).get("beta_standardized", 0) for m in predictor_cols
        ]
        ses = [results.get(m, {}).get("se", 0) for m in predictor_cols]

        y_pos = np.arange(len(predictor_cols))

        bars = ax.barh(
            y_pos,
            betas,
            xerr=ses,
            color="steelblue",
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )

        ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(predictor_cols, fontsize=12)
        ax.set_xlabel("Standardized β", fontsize=14)
        ax.set_title(f"{ensemble}", fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # Add value labels
        for i, (beta, se) in enumerate(zip(betas, ses)):
            label_x = beta + se + 0.01 if beta > 0 else beta - se - 0.01
            ha = "left" if beta > 0 else "right"
            ax.text(
                label_x,
                i,
                f"{beta:.4f}",
                va="center",
                ha=ha,
                fontsize=10,
                fontweight="bold",
            )

    plt.tight_layout()
    output_path = os.path.join(output_dir, "01_coefficient_comparison.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_partial_r2_comparison(
    regression_results: Dict, predictor_cols: list, output_dir: str
) -> None:
    """Create partial R² comparison plot."""
    ensembles = sorted(regression_results.keys())
    if not ensembles:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(ensembles))
    width = 0.8 / len(predictor_cols)

    for i, metric in enumerate(predictor_cols):
        r2_vals = [
            regression_results.get(ens, {})
            .get(metric, {})
            .get("partial_r2", 0)
            for ens in ensembles
        ]

        ax.bar(
            x + i * width,
            r2_vals,
            width,
            label=metric,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.0,
        )

    ax.set_ylabel("Partial R²", fontsize=14)
    ax.set_xlabel("Ensemble", fontsize=14)
    ax.set_title("Partial R²: Metric Predictive Utility", fontsize=16, fontweight="bold")
    ax.set_xticks(x + width * (len(predictor_cols) - 1) / 2)
    ax.set_xticklabels(ensembles, fontsize=12)
    ax.legend(fontsize=11, loc="best")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "02_partial_r2_comparison.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_stability_comparison(
    stability_results: Dict, 
    predictor_cols: list, 
    output_dir: str,
    xlabel: str = "Ensemble",
    title_suffix: str = "Across Splits",
    filename_suffix: str = ""
) -> None:
    """Create stability metrics comparison plot."""
    ensembles = sorted(stability_results.keys())
    if not ensembles:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Stability Index
    ax1 = axes[0]
    x = np.arange(len(ensembles))
    width = 0.8 / len(predictor_cols)

    for i, metric in enumerate(predictor_cols):
        stab_vals = [
            stability_results.get(ens, {})
            .get(metric, {})
            .get("stability_index", 0)
            for ens in ensembles
        ]
        ax1.bar(
            x + i * width,
            stab_vals,
            width,
            label=metric,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.0,
        )

    ax1.set_ylabel("Stability Index", fontsize=14)
    ax1.set_xlabel(xlabel, fontsize=14)
    ax1.set_title(f"Stability {title_suffix}", fontsize=16, fontweight="bold")
    ax1.set_xticks(x + width * (len(predictor_cols) - 1) / 2)
    ax1.set_xticklabels(ensembles, fontsize=12)
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=11, loc="best")
    ax1.grid(axis="y", alpha=0.3)

    # Coefficient of Variation
    ax2 = axes[1]
    for i, metric in enumerate(predictor_cols):
        cv_vals = [
            stability_results.get(ens, {})
            .get(metric, {})
            .get("cv_across_splits", 0)
            for ens in ensembles
        ]
        # Clip infinite values
        cv_vals = [min(v, 1.0) if np.isfinite(v) else 0 for v in cv_vals]
        ax2.bar(
            x + i * width,
            cv_vals,
            width,
            label=metric,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.0,
        )

    ax2.set_ylabel("Coefficient of Variation", fontsize=14)
    ax2.set_xlabel(xlabel, fontsize=14)
    ax2.set_title(f"Metric Variability {title_suffix}", fontsize=16, fontweight="bold")
    ax2.set_xticks(x + width * (len(predictor_cols) - 1) / 2)
    ax2.set_xticklabels(ensembles, fontsize=12)
    ax2.legend(fontsize=11, loc="best")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"03_stability_comparison{filename_suffix}.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_eta_and_ftest(
    stability_results: Dict, 
    predictor_cols: list, 
    output_dir: str,
    xlabel: str = "Ensemble",
    title_suffix: str = "Split Type Effect",
    filename_suffix: str = ""
) -> None:
    """Create eta-squared and F-test comparison plots."""
    ensembles = sorted(stability_results.keys())
    if not ensembles:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(ensembles))
    width = 0.8 / len(predictor_cols)

    # Eta-squared (effect size)
    ax1 = axes[0]
    for i, metric in enumerate(predictor_cols):
        eta_vals = [
            stability_results.get(ens, {})
            .get(metric, {})
            .get("eta_squared", 0)
            for ens in ensembles
        ]
        ax1.bar(
            x + i * width,
            eta_vals,
            width,
            label=metric,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.0,
        )

    ax1.set_ylabel("η² (Effect Size)", fontsize=14)
    ax1.set_xlabel(xlabel, fontsize=14)
    ax1.set_title(f"Effect Size: {title_suffix}", fontsize=16, fontweight="bold")
    ax1.set_xticks(x + width * (len(predictor_cols) - 1) / 2)
    ax1.set_xticklabels(ensembles, fontsize=12)
    ax1.legend(fontsize=11, loc="best")
    ax1.grid(axis="y", alpha=0.3)

    # F-statistic
    ax2 = axes[1]
    for i, metric in enumerate(predictor_cols):
        f_vals = [
            stability_results.get(ens, {})
            .get(metric, {})
            .get("f_statistic", 0)
            for ens in ensembles
        ]
        ax2.bar(
            x + i * width,
            f_vals,
            width,
            label=metric,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.0,
        )

    ax2.set_ylabel("F-statistic", fontsize=14)
    ax2.set_xlabel(xlabel, fontsize=14)
    ax2.set_title(f"ANOVA F-test: {title_suffix}", fontsize=16, fontweight="bold")
    ax2.set_xticks(x + width * (len(predictor_cols) - 1) / 2)
    ax2.set_xticklabels(ensembles, fontsize=12)
    ax2.legend(fontsize=11, loc="best")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"04_eta_ftest_comparison{filename_suffix}.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_scatter_and_distributions(
    data: pd.DataFrame, target_metric: str, predictor_cols: list, output_dir: str
) -> None:
    """Create combined scatter and distribution plots for each metric and ensemble."""
    ensembles = sorted(data["ensemble"].unique())
    
    has_bv_reg = "bv_reg_value" in data.columns
    n_cols = 6 if has_bv_reg else 5

    for metric in predictor_cols:
        if metric not in data.columns:
            continue

        fig, axes = plt.subplots(
            len(ensembles), n_cols, figsize=(6 * n_cols, 5 * len(ensembles))
        )
        if len(ensembles) == 1:
            axes = axes.reshape(1, -1)

        for idx, ensemble in enumerate(ensembles):
            df_ens = data[data["ensemble"] == ensemble].copy()

            # Left plot: Distribution
            ax_dist = axes[idx, 0]
            if "split_type" in df_ens.columns:
                split_types = sorted(df_ens["split_type"].unique())
                for split_type in split_types:
                    df_split = df_ens[df_ens["split_type"] == split_type]
                    ax_dist.hist(
                        df_split[metric].dropna(),
                        bins=15,
                        alpha=0.6,
                        label=split_name_mapping.get(split_type, split_type),
                        color=split_colors.get(split_type, "gray"),
                    )
            else:
                ax_dist.hist(df_ens[metric].dropna(), bins=20, alpha=0.7, color="steelblue")

            ax_dist.set_xlabel(metric, fontsize=12)
            ax_dist.set_ylabel("Frequency", fontsize=12)
            ax_dist.set_title(f"{ensemble} - Distribution", fontsize=13, fontweight="bold")
            ax_dist.legend(fontsize=10)
            ax_dist.grid(alpha=0.3)

            # Second plot: Scatter with regression fit
            ax_scatter = axes[idx, 1]

            if "split_type" in df_ens.columns:
                split_types = sorted(df_ens["split_type"].unique())
                for split_type in split_types:
                    df_split = df_ens[df_ens["split_type"] == split_type]
                    
                    if "split_idx" in df_split.columns:
                        unique_idxs = sorted(df_split["split_idx"].unique())
                        for i, s_idx in enumerate(unique_idxs):
                            df_sub = df_split[df_split["split_idx"] == s_idx]
                            valid = df_sub[[metric, target_metric]].dropna()
                            
                            # Determine marker based on split_idx
                            try:
                                m_idx = int(s_idx)
                            except (ValueError, TypeError):
                                m_idx = i
                            marker = MARKER_LIST[m_idx % len(MARKER_LIST)]
                            
                            # Only label the first occurrence of the split_type
                            label = split_name_mapping.get(split_type, split_type) if i == 0 else None

                            ax_scatter.scatter(
                                valid[metric],
                                valid[target_metric],
                                alpha=0.6,
                                s=60,
                                color=split_colors.get(split_type, "gray"),
                                marker=marker,
                                label=label,
                                edgecolor="black",
                                linewidth=0.5,
                            )
                    else:
                        valid = df_split[[metric, target_metric]].dropna()
                        ax_scatter.scatter(
                            valid[metric],
                            valid[target_metric],
                            alpha=0.6,
                            s=60,
                            color=split_colors.get(split_type, "gray"),
                            label=split_name_mapping.get(split_type, split_type),
                            edgecolor="black",
                            linewidth=0.5,
                        )
            else:
                valid = df_ens[[metric, target_metric]].dropna()
                ax_scatter.scatter(
                    valid[metric],
                    valid[target_metric],
                    alpha=0.6,
                    s=60,
                    color="steelblue",
                    edgecolor="black",
                    linewidth=0.5,
                )

            # Add regression line
            valid = df_ens[[metric, target_metric]].dropna()
            if len(valid) > 2:
                X = valid[metric].values.reshape(-1, 1)
                y = valid[target_metric].values
                reg = LinearRegression().fit(X, y)
                x_line = np.linspace(X.min(), X.max(), 100)
                y_line = reg.predict(x_line.reshape(-1, 1))
                r2 = r2_score(y, reg.predict(X))
                ax_scatter.plot(
                    x_line,
                    y_line,
                    "k--",
                    linewidth=2.5,
                    alpha=0.7,
                    label=f"Linear fit (R²={r2:.3f})",
                )

            ax_scatter.set_xlabel(metric, fontsize=12)
            ax_scatter.set_ylabel(target_metric, fontsize=12)
            ax_scatter.set_title(f"{ensemble} - Relationship", fontsize=13, fontweight="bold")
            ax_scatter.legend(fontsize=10)
            ax_scatter.grid(alpha=0.3)

            # Third plot: Scatter (Hue: MaxEnt Value)
            ax_maxent = axes[idx, 2]
            if "maxent_value" in df_ens.columns:
                vals = df_ens["maxent_value"].dropna()
                if not vals.empty:
                    # Use LogNorm if range is large
                    if vals.max() > 0 and vals.min() > 0 and (vals.max() / vals.min() > 50):
                        norm = matplotlib.colors.LogNorm(vmin=vals.min(), vmax=vals.max())
                    else:
                        norm = plt.Normalize(vmin=vals.min(), vmax=vals.max())
                    cmap = plt.cm.viridis

                    if "split_type" in df_ens.columns and "split_idx" in df_ens.columns:
                        split_types = sorted(df_ens["split_type"].unique())
                        for split_type in split_types:
                            df_split = df_ens[df_ens["split_type"] == split_type]
                            unique_idxs = sorted(df_split["split_idx"].unique())
                            for i, s_idx in enumerate(unique_idxs):
                                df_sub = df_split[df_split["split_idx"] == s_idx]
                                valid = df_sub[[metric, target_metric, "maxent_value"]].dropna()
                                if valid.empty: continue
                                
                                try:
                                    m_idx = int(s_idx)
                                except (ValueError, TypeError):
                                    m_idx = i
                                marker = MARKER_LIST[m_idx % len(MARKER_LIST)]
                                
                                ax_maxent.scatter(
                                    valid[metric],
                                    valid[target_metric],
                                    c=valid["maxent_value"],
                                    norm=norm,
                                    cmap=cmap,
                                    marker=marker,
                                    s=60,
                                    alpha=0.7,
                                    edgecolor="black",
                                    linewidth=0.5,
                                )
                    else:
                        valid = df_ens[[metric, target_metric, "maxent_value"]].dropna()
                        ax_maxent.scatter(
                            valid[metric],
                            valid[target_metric],
                            c=valid["maxent_value"],
                            norm=norm,
                            cmap=cmap,
                            s=60,
                            alpha=0.7,
                            edgecolor="black",
                            linewidth=0.5,
                        )
                    
                    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_maxent, label="MaxEnt Value")
            
            ax_maxent.set_xlabel(metric, fontsize=12)
            ax_maxent.set_ylabel(target_metric, fontsize=12)
            ax_maxent.set_title(f"{ensemble} - Hue: MaxEnt", fontsize=13, fontweight="bold")
            ax_maxent.grid(alpha=0.3)

            # Fourth plot: Scatter (Hue: Convergence Value)
            ax_conv = axes[idx, 3]
            if "convergence_value" in df_ens.columns:
                vals = df_ens["convergence_value"].dropna()
                if not vals.empty:
                    if vals.max() > 0 and vals.min() > 0 and (vals.max() / vals.min() > 50):
                        norm = matplotlib.colors.LogNorm(vmin=vals.min(), vmax=vals.max())
                    else:
                        norm = plt.Normalize(vmin=vals.min(), vmax=vals.max())
                    cmap = plt.cm.plasma

                    if "split_type" in df_ens.columns and "split_idx" in df_ens.columns:
                        split_types = sorted(df_ens["split_type"].unique())
                        for split_type in split_types:
                            df_split = df_ens[df_ens["split_type"] == split_type]
                            unique_idxs = sorted(df_split["split_idx"].unique())
                            for i, s_idx in enumerate(unique_idxs):
                                df_sub = df_split[df_split["split_idx"] == s_idx]
                                valid = df_sub[[metric, target_metric, "convergence_value"]].dropna()
                                if valid.empty: continue
                                
                                try:
                                    m_idx = int(s_idx)
                                except (ValueError, TypeError):
                                    m_idx = i
                                marker = MARKER_LIST[m_idx % len(MARKER_LIST)]
                                
                                ax_conv.scatter(
                                    valid[metric],
                                    valid[target_metric],
                                    c=valid["convergence_value"],
                                    norm=norm,
                                    cmap=cmap,
                                    marker=marker,
                                    s=60,
                                    alpha=0.7,
                                    edgecolor="black",
                                    linewidth=0.5,
                                )
                    else:
                        valid = df_ens[[metric, target_metric, "convergence_value"]].dropna()
                        ax_conv.scatter(
                            valid[metric],
                            valid[target_metric],
                            c=valid["convergence_value"],
                            norm=norm,
                            cmap=cmap,
                            s=60,
                            alpha=0.7,
                            edgecolor="black",
                            linewidth=0.5,
                        )
                    
                    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_conv, label="Convergence Value")

            ax_conv.set_xlabel(metric, fontsize=12)
            ax_conv.set_ylabel(target_metric, fontsize=12)
            ax_conv.set_title(f"{ensemble} - Hue: Convergence", fontsize=13, fontweight="bold")
            ax_conv.grid(alpha=0.3)

            # Fifth plot: Scatter (Hue: -log(Val Loss))
            ax_loss = axes[idx, 4]
            if "val_loss" in df_ens.columns:
                # Calculate -log(val_loss)
                mask = (df_ens["val_loss"] > 0) & df_ens["val_loss"].notna()
                if mask.any():
                    neg_log_loss_col = "_neg_log_val_loss"
                    df_ens[neg_log_loss_col] = np.nan
                    df_ens.loc[mask, neg_log_loss_col] = -np.log(df_ens.loc[mask, "val_loss"])
                    
                    vals = df_ens[neg_log_loss_col].dropna()
                    if not vals.empty:
                        norm = plt.Normalize(vmin=vals.min(), vmax=vals.max())
                        cmap = plt.cm.ocean_r

                        if "split_type" in df_ens.columns and "split_idx" in df_ens.columns:
                            split_types = sorted(df_ens["split_type"].unique())
                            for split_type in split_types:
                                df_split = df_ens[df_ens["split_type"] == split_type]
                                unique_idxs = sorted(df_split["split_idx"].unique())
                                for i, s_idx in enumerate(unique_idxs):
                                    df_sub = df_split[df_split["split_idx"] == s_idx]
                                    valid = df_sub[[metric, target_metric, neg_log_loss_col]].dropna()
                                    if valid.empty: continue
                                    
                                    try:
                                        m_idx = int(s_idx)
                                    except (ValueError, TypeError):
                                        m_idx = i
                                    marker = MARKER_LIST[m_idx % len(MARKER_LIST)]
                                    
                                    ax_loss.scatter(
                                        valid[metric],
                                        valid[target_metric],
                                        c=valid[neg_log_loss_col],
                                        norm=norm,
                                        cmap=cmap,
                                        marker=marker,
                                        s=60,
                                        alpha=0.7,
                                        edgecolor="black",
                                        linewidth=0.5,
                                    )
                        else:
                            valid = df_ens[[metric, target_metric, neg_log_loss_col]].dropna()
                            ax_loss.scatter(
                                valid[metric],
                                valid[target_metric],
                                c=valid[neg_log_loss_col],
                                norm=norm,
                                cmap=cmap,
                                s=60,
                                alpha=0.7,
                                edgecolor="black",
                                linewidth=0.5,
                            )
                        
                        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_loss, label="-log(Val Loss)")

            ax_loss.set_xlabel(metric, fontsize=12)
            ax_loss.set_ylabel(target_metric, fontsize=12)
            ax_loss.set_title(f"{ensemble} - Hue: -log(Val Loss)", fontsize=13, fontweight="bold")
            ax_loss.grid(alpha=0.3)

            # Sixth plot: Scatter (Hue: BV Reg Value)
            if has_bv_reg:
                ax_bv = axes[idx, 5]
                vals = df_ens["bv_reg_value"].dropna()
                if not vals.empty:
                    if vals.max() > 0 and vals.min() > 0 and (vals.max() / vals.min() > 50):
                        norm = matplotlib.colors.LogNorm(vmin=vals.min(), vmax=vals.max())
                    else:
                        norm = plt.Normalize(vmin=vals.min(), vmax=vals.max())
                    cmap = plt.cm.cividis

                    if "split_type" in df_ens.columns and "split_idx" in df_ens.columns:
                        split_types = sorted(df_ens["split_type"].unique())
                        for split_type in split_types:
                            df_split = df_ens[df_ens["split_type"] == split_type]
                            unique_idxs = sorted(df_split["split_idx"].unique())
                            for i, s_idx in enumerate(unique_idxs):
                                df_sub = df_split[df_split["split_idx"] == s_idx]
                                valid = df_sub[[metric, target_metric, "bv_reg_value"]].dropna()
                                if valid.empty: continue
                                
                                try:
                                    m_idx = int(s_idx)
                                except (ValueError, TypeError):
                                    m_idx = i
                                marker = MARKER_LIST[m_idx % len(MARKER_LIST)]
                                
                                ax_bv.scatter(
                                    valid[metric],
                                    valid[target_metric],
                                    c=valid["bv_reg_value"],
                                    norm=norm,
                                    cmap=cmap,
                                    marker=marker,
                                    s=60,
                                    alpha=0.7,
                                    edgecolor="black",
                                    linewidth=0.5,
                                )
                    else:
                        valid = df_ens[[metric, target_metric, "bv_reg_value"]].dropna()
                        ax_bv.scatter(
                            valid[metric],
                            valid[target_metric],
                            c=valid["bv_reg_value"],
                            norm=norm,
                            cmap=cmap,
                            s=60,
                            alpha=0.7,
                            edgecolor="black",
                            linewidth=0.5,
                        )
                    
                    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_bv, label="BV Reg Value")

                ax_bv.set_xlabel(metric, fontsize=12)
                ax_bv.set_ylabel(target_metric, fontsize=12)
                ax_bv.set_title(f"{ensemble} - Hue: BV Reg", fontsize=13, fontweight="bold")
                ax_bv.grid(alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"05_scatter_and_distribution_{metric}.png")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close()


def plot_model_selection_performance(
    df: pd.DataFrame,
    target_metric: str,
    predictor_cols: list,
    output_dir: str,
) -> None:
    """
    Plots the recovery performance when selecting models based on different scores.
    Also saves detailed CSVs with original, transformed, and ranked scores.
    """
    print("\nGenerating model selection performance plots...")
    
    df_plot = df.copy()
    
    # Construct grouping columns and x-axis
    group_cols = ['ensemble', 'split_type', 'split_idx']
    
    if 'loss_function' in df_plot.columns:
        group_cols.append('loss_function')
        
    if 'bv_reg_function' in df_plot.columns:
        group_cols.append('bv_reg_function')

    # Create a display column for x-axis
    if 'loss_function' in df_plot.columns and 'bv_reg_function' in df_plot.columns:
        df_plot['method_variant'] = df_plot['loss_function'].astype(str) + "_" + df_plot['bv_reg_function'].astype(str)
        x_axis = 'method_variant'
    elif 'loss_function' in df_plot.columns:
        x_axis = 'loss_function'
    elif 'bv_reg_function' in df_plot.columns:
        x_axis = 'bv_reg_function'
    else:
        df_plot['method_variant'] = 'All'
        group_cols.append('method_variant')
        x_axis = 'method_variant'
    
    # Filter out groups that don't have the target metric
    df_clean = df_plot.dropna(subset=[target_metric]).copy()

    selection_stats_list = []
    detailed_rows = []

    for metric in predictor_cols:
        if metric not in df_clean.columns:
            continue
            
        # Determine optimization direction
        # Heuristic: if 'loss', 'mse', 'mae', 'error' in name -> minimize
        # Else -> maximize (as per user instruction "maximising the scores", with exception for loss)
        is_loss = any(x in metric.lower() for x in ['loss', 'mse', 'mae', 'error','scale'])
        direction = "min" if is_loss else "max"
        
        try:
            # Select best model per group
            if direction == "min":
                idx = df_clean.groupby(group_cols)[metric].idxmin()
            else:
                idx = df_clean.groupby(group_cols)[metric].idxmax()
            
            idx = idx.dropna()
            selected_models = df_clean.loc[idx].copy()
            
            if selected_models.empty:
                continue

            # Collect stats for summary CSV
            agg_cols = ['ensemble', 'split_type']
            if 'loss_function' in selected_models.columns:
                agg_cols.append('loss_function')
            if 'bv_reg_function' in selected_models.columns:
                agg_cols.append('bv_reg_function')
            
            # Calculate stats for Target and ALL predictor metrics (original, transformed, rank, percentile)
            agg_dict = {target_metric: ['mean', 'std', 'min', 'count']}
            for pred in predictor_cols:
                agg_dict[pred] = ['mean', 'std']
                trans_col = f"{pred}_transformed"
                rank_col = f"{pred}_rank"
                pct_col = f"{pred}_percentile"
                if trans_col in selected_models.columns:
                    agg_dict[trans_col] = ['mean', 'std']
                if rank_col in selected_models.columns:
                    agg_dict[rank_col] = ['mean', 'std']
                if pct_col in selected_models.columns:
                    agg_dict[pct_col] = ['mean', 'std']
                
            stats = selected_models.groupby(agg_cols).agg(agg_dict).reset_index()
            
            # Flatten MultiIndex columns
            new_cols = []
            for c in stats.columns:
                if isinstance(c, tuple):
                    if c[1]:
                        new_cols.append(f"{c[0]}_{c[1]}")
                    else:
                        new_cols.append(c[0])
                else:
                    new_cols.append(c)
            stats.columns = new_cols
            
            stats['score_metric'] = metric
            stats['direction'] = direction
            selection_stats_list.append(stats)
            
            # Collect detailed rows for by-split CSV: include ALL predictor metrics and their transforms
            selected_models['score_metric'] = metric
            selected_models['direction'] = direction
            # Keep relevant columns: group_cols + target + ALL predictors and their transforms
            keep_cols = group_cols + [target_metric, 'score_metric', 'direction']
            target_rank_col = f"{target_metric}_rank"
            target_pct_col = f"{target_metric}_percentile"
            for col in (target_rank_col, target_pct_col):
                if col in selected_models.columns:
                    keep_cols.append(col)
            for pred in predictor_cols:
                keep_cols.append(pred)
                trans_col = f"{pred}_transformed"
                rank_col = f"{pred}_rank"
                pct_col = f"{pred}_percentile"
                if trans_col in selected_models.columns:
                    keep_cols.append(trans_col)
                if rank_col in selected_models.columns:
                    keep_cols.append(rank_col)
                if pct_col in selected_models.columns:
                    keep_cols.append(pct_col)
            
            # Remove duplicates while preserving order
            keep_cols = list(dict.fromkeys(keep_cols))
            
            detailed_rows.append(selected_models[keep_cols])

            # Plotting
            g = sns.catplot(
                data=selected_models,
                x=x_axis,
                y=target_metric,
                hue="split_type",
                col="ensemble",
                kind="bar",
                height=5,
                aspect=1,
                sharey=True,
                palette=split_colors,
                errorbar="sd",
                capsize=0.1,
                edgecolor="black",
                linewidth=1.0
            )
            
            g.fig.subplots_adjust(top=0.85)
            g.fig.suptitle(f"Selection by {metric} ({direction}) -> {target_metric}", fontsize=16, fontweight='bold')
            
            g.set_axis_labels("Method Variant", target_metric)
            g.set_titles("{col_name}")
            
            safe_metric = "".join(c for c in metric if c.isalnum() or c in ('_', '-'))
            output_path = os.path.join(output_dir, f"06_selection_performance_{safe_metric}.png")
            g.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(g.fig)
            print(f"  Saved selection plot for {metric}")

        except Exception as e:
            print(f"  Could not plot selection for {metric}: {e}")
            import traceback
            traceback.print_exc()

    # Save summary CSV
    if selection_stats_list:
        full_stats = pd.concat(selection_stats_list, ignore_index=True)
        out_path = os.path.join(output_dir, "model_selection_performance_summary.csv")
        full_stats.to_csv(out_path, index=False)
        print(f"Saved model selection performance summary to {out_path}")
        
    # Save detailed CSV
    if detailed_rows:
        full_detailed = pd.concat(detailed_rows, ignore_index=True)
        out_path_detailed = os.path.join(output_dir, "model_selection_performance_by_split.csv")
        full_detailed.to_csv(out_path_detailed, index=False)
        print(f"Saved detailed model selection performance to {out_path_detailed}")


def calculate_and_save_correlations(
    df: pd.DataFrame,
    target_metric: str,
    predictor_cols: list,
    output_dir: str,
) -> pd.DataFrame:
    """
    Calculate Pearson correlations between target metric and predictor columns.
    Saves a CSV summary and returns a DataFrame with detailed results.
    """
    print("\nCalculating correlations...")
    
    results = []
    for col in predictor_cols:
        if col not in df.columns:
            continue
            
        # Skip non-numeric columns
        if not np.issubdtype(df[col].dtype, np.number):
            continue

        # Calculate correlation
        valid = df[[col, target_metric]].dropna()
        if len(valid) < 2:
            continue
        corr, p_value = stats.pearsonr(valid[col], valid[target_metric])
        
        results.append({
            "metric": col,
            "correlation": corr,
            "p_value": p_value,
            "split_type": df["split_type"].iloc[0] if "split_type" in df else "N/A",
            "ensemble": df["ensemble"].iloc[0] if "ensemble" in df else "N/A",
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        output_path = os.path.join(output_dir, "correlations_summary.csv")
        results_df.to_csv(output_path, index=False)
        print(f"Saved correlation summary to {output_path}")
        
    return results_df


def plot_correlations_bar_charts(
    corr_df: pd.DataFrame,
    output_dir: str
):
    """
    Plot correlations as bar charts.
    Separate figures for each split type.
    X-axis: Scores (metrics).
    Hue: Ensemble.
    """
    if corr_df.empty:
        return

    split_types = corr_df['split_type'].unique()
    
    for split_type in split_types:
        df_subset = corr_df[corr_df['split_type'] == split_type]
        
        if df_subset.empty:
            continue
            
        plt.figure(figsize=(max(10, len(df_subset['metric'].unique()) * 1.5), 6))
        
        # Use seaborn for easy grouping
        g = sns.barplot(
            data=df_subset,
            x='metric',
            y='correlation',
            hue='ensemble',
            palette=ensemble_colors,
            edgecolor='black'
        )
        
        plt.title(f"Correlation with Target (Split Type: {split_name_mapping.get(split_type, split_type)})", fontsize=16)
        plt.xlabel("Score Metric", fontsize=14)
        plt.ylabel("Pearson Correlation", fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.legend(title='Ensemble', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        safe_name = "".join(c for c in str(split_type) if c.isalnum() or c in ('_', '-'))
        output_path = os.path.join(output_dir, f"07_correlation_bars_{safe_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved correlation plot for {split_type}")


if __name__ == "__main__":
    main()