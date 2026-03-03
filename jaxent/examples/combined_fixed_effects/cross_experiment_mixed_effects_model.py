"""
Cross-Experiment Mixed Effects Model Analysis

Function:
Integrates results across multiple JAX-ENT experiments (IsoValidation, MoPrP-Reweighting, MoPrP-RW+BV) to:
- Assess the robustness and performance of various metrics (MSE, Work, etc.).
- Quantify variance components using linear mixed-effects modeling.
- Identify the most stable and predictive indicators across diverse systems.
- Generate performance vs. robustness scatter plots and 3D summaries.

Requirements:
- Paths to long-format GT scores (`gt_scores_long.csv`) from each experiment.
"""

import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Set publication-ready style
sns.set_style("ticks")
sns.set_context(
    "paper",
    rc={
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    },
)

# Default experiment paths
DEFAULT_EXPERIMENTS = {
    "TeaA-IsoValidation": "jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/_processed__optimise_test_SIGMA_500__20260223_215731/_analysis__scores__processed__optimise_test_SIGMA_500__20260223_215731/plots_selection",
    "MoPrP-Reweighting": "jaxent/examples/2_CrossValidation/fitting/jaxENT/_processed__optimise_quick_test_FIGURE_SIGMA_500__20260223_212645/_analysis__scores__processed__optimise_quick_test_FIGURE_SIGMA_500__20260223_212645/plots_selection",
    "MoPrP-RW+BV": "jaxent/examples/2_CrossValidation/fitting/jaxENT/_processed__optimise_quick_test_FIGURE_SIGMA_500__20260223_212645/_analysis__scores__processed__optimise_quick_test_FIGURE_SIGMA_500__20260223_212645/plots_selection"
}

ENSEMBLE_CLASS_MAPPING = {
    "ISO_BI": "GT-only",
    "AF2_filtered": "GT-only",
    "ISO_TRI": "GT+alt",
    "AF2_MSAss": "GT+alt"
}

METRIC_HASHTYPE_MAPPING = {
    'mse': 'xx',
    'loss': '++',
}
# Updated color schemes
metric_colors = sns.color_palette("tab20")
class_colors = {"GT-only": "#3498db", "GT+alt": "#e74c3c"}
exp_colors = {"TeaA-IsoValidation": "#2ecc71", "MoPrP-Reweighting": "#9b59b6", "MoPrP-RW+BV": "#f1c40f"}

# --- Helper Functions ---

def sanitize_string(val):
    """Sanitize string for statsmodels formula compatibility."""
    return str(val).replace('+', '_').replace('-', '_')

def load_data(experiments_dict):
    all_data = []
    for exp_name, path in experiments_dict.items():
        csv_path = os.path.join(path, "gt_scores_long.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping {exp_name}.")
            continue
            
        df = pd.read_csv(csv_path)
        df['experiment'] = exp_name
        
        # Map classes
        df['mclass'] = df['ensemble'].map(ENSEMBLE_CLASS_MAPPING)
        
        if df['mclass'].isnull().any():
            unknowns = df[df['mclass'].isnull()]['ensemble'].unique()
            print(f"Warning: Unknown ensembles in {exp_name}: {unknowns}")
            df = df.dropna(subset=['mclass'])
            
        all_data.append(df)
        
    if not all_data:
        raise ValueError("No data loaded!")
        
    return pd.concat(all_data, ignore_index=True)

def preprocess_scores(df):
    """
    Transform scores and calculate percentiles.
    """
    df['transformed_value'] = df['value']
    
    # CV: Lower is better -> -log(CV)
    mask_cv = df['gt_score_type'] == 'CV'
    df.loc[mask_cv, 'transformed_value'] = -np.log(df.loc[mask_cv, 'value'] + 1e-9)
    
    # Regret: Lower is better -> -Regret
    mask_regret = df['gt_score_type'] == 'Regret'
    df.loc[mask_regret, 'transformed_value'] = -df.loc[mask_regret, 'value']
    
    # Calculate Percentiles
    def calc_percentile(g):
        return g.rank(pct=True) * 100
        
    df['percentile'] = df.groupby(['experiment', 'ensemble', 'gt_score_type'])['transformed_value'].transform(calc_percentile)
    
    return df

def extract_variance_components(model_result):
    """Safely extract variance components (diagonal elements) from MixedLM result."""
    vcomps = model_result.cov_re
    
    if hasattr(vcomps, 'values'): # DataFrame or Series
        if vcomps.ndim == 2:
            vals = vcomps.values.diagonal()
        else:
            vals = vcomps.values
        keys = vcomps.index
    else: # Dict
        vals = list(vcomps.values())
        keys = list(vcomps.keys())
        
    return keys, vals

def get_model_coefficient(term, level, param_index, param_df):
    """
    Robustly lookup coefficient for a given term and level.
    Handles C(term)[level] and C(term)[T.level] formats.
    Returns (coef, lower, upper, found_bool)
    """
    sanitized = sanitize_string(level)
    
    # Try formats
    candidates = [
        f"C({term})[{sanitized}]",
        f"C({term})[T.{sanitized}]"
    ]
    
    for name in candidates:
        if name in param_index:
            row = param_df.loc[name]
            return row['coef'], row[0], row[1], True
            
    return 0.0, 0.0, 0.0, False

def calculate_robustness_and_performance(model_result, metrics):
    """
    Calculate performance (main effect) and robustness (inverse sum of interaction magnitudes)
    for each metric.
    """
    params = model_result.params
    robustness_data = []
    
    for metric in metrics:
        sanitized_metric = sanitize_string(metric)
        
        # 1. Performance: Main effect coefficient
        # Since we use "0 + C(metric)", the coefficient IS the main effect
        # We look for exactly C(metric)[level]
        main_effect_name = f"C(metric)[{sanitized_metric}]"
        if main_effect_name in params:
            performance = params[main_effect_name]
        else:
            # Should not happen if we use 0 + C(metric)
            performance = 0.0
            
        # 2. Robustness: Sum of absolute interaction coefficients involving this metric
        interaction_sum = 0.0
        
        # Iterate over all params to find interactions involving this metric
        # Interaction terms will look like "C(metric)[T.level]:C(other)[...]" or "C(other)[...]:C(metric)[T.level]"
        # Or "C(metric)[level]:C(other)[...]" if no intercept (but interactions usually have T.)
        
        target_1 = f"C(metric)[{sanitized_metric}]"
        target_2 = f"C(metric)[T.{sanitized_metric}]"
        
        for param_name, val in params.items():
            # Must be an interaction term (contain ':')
            if ":" not in param_name:
                continue
                
            # Check if it involves our metric level
            if target_1 in param_name or target_2 in param_name:
                interaction_sum += abs(val)
                
        # Robustness = Inverse of total interaction magnitude
        # Add small epsilon to avoid division by zero
        robustness = 1.0 / (interaction_sum + 1e-5)
        
        robustness_data.append({
            'Metric': metric,
            'Performance': performance,
            'Robustness': robustness,
            'Interaction_Sum': interaction_sum
        })
        
    return pd.DataFrame(robustness_data)

def get_metric_order(metrics):
    """
    Sort metrics based on groups:
    1. Loss
    2. MSE and d_MSE
    3. KL and Work
    4. Other/Control
    Within groups, sort alphabetically.
    """
    g1 = [] # Loss
    g2 = [] # MSE
    g3 = [] # KL/Work
    g4 = [] # Other
    
    for m in metrics:
        m_lower = str(m).lower()
        if 'loss' in m_lower:
            g1.append(m)
        elif 'mse' in m_lower:
            g2.append(m)
        elif 'kl' in m_lower or 'work' in m_lower:
            g3.append(m)
        else:
            g4.append(m)
            
    return sorted(g1) + sorted(g2) + sorted(g3) + sorted(g4)

# --- Modeling ---

def fit_mixed_model(df, gt_score_type, output_dir):
    print(f"\n--- Fitting Mixed Effects Model for {gt_score_type} ---")
    # Work on a copy
    df_work = df[df['gt_score_type'] == gt_score_type].copy()
    
    df_work = df_work.rename(columns={
        'score_metric': 'metric',
        'loss_function': 'loss',
        'split_type': 'split',
        'experiment': 'exp'
    })
    
    # Filter sparse clusters (Cluster 0 and 1 only)
    is_cluster = df_work['metric'].str.startswith('cluster_')
    allowed_clusters = df_work['metric'].isin(['cluster_0', 'cluster_1'])
    df_work = df_work[(~is_cluster) | allowed_clusters]

    # Drop NaNs
    df_work = df_work.dropna(subset=['percentile', 'metric', 'mclass', 'exp', 'ensemble', 'loss', 'split'])
    df_work['group'] = 1
    
    # Create sanitized version for fitting
    fit_df = df_work.copy()
    for col in ['mclass', 'metric', 'exp', 'loss', 'split']:
        fit_df[col] = fit_df[col].apply(sanitize_string)
    
    # Fixed Effects Formula
    # Main effects + Interactions specified in plan
    # Interactions: Metric x Class, Metric x Exp, Metric x Loss, Metric x Split
    #               Class x Exp, Class x Loss, Class x Split
    #               Exp x Loss, Exp x Split
    #               Loss x Split
    
    fixed_formula = (
        "percentile ~ 0 + C(metric) + C(mclass) + C(exp) + C(loss) + C(split) + "
        "C(metric):C(mclass) + C(metric):C(exp) + C(metric):C(loss) + C(metric):C(split) + "
        "C(mclass):C(exp) + C(mclass):C(loss) + C(mclass):C(split) + "
        "C(exp):C(loss) + C(exp):C(split) + "
        "C(loss):C(split)"
    )
    
    # Variance Components: Ensemble nested in Class
    # We use 'ensemble' as a random intercept. 
    # Since 'ensemble' names might be unique or not, usually 'ensemble' nested in 'mclass' 
    # is handled by treating 'ensemble' as the grouping factor if unique.
    # Here we use a single variance component for 'ensemble'.
    vc_formula = {
        "ensemble": "0 + C(ensemble)"
    }

    print("Fitting full mixed model...")
    try:
        model = smf.mixedlm(fixed_formula, fit_df, groups="group", vc_formula=vc_formula)
        # Using 'powell' as it's more robust
        result = model.fit(reml=True, method='powell', maxiter=5000)
        
        if not result.converged:
            print("Mixed model did not converge.")
        
        print(result.summary())
        
        with open(os.path.join(output_dir, f"mixed_model_summary_{gt_score_type}.txt"), "w") as f:
            f.write(result.summary().as_text())
            
        return result, df_work
        
    except Exception as e:
        print(f"Error fitting mixed model for {gt_score_type}: {e}")

    # Fallback to OLS
    print(f"Falling back to OLS (Fixed Effects Only) for {gt_score_type}...")
    try:
        model = smf.ols(fixed_formula, data=fit_df)
        result = model.fit()
        print(result.summary())
        with open(os.path.join(output_dir, f"ols_model_summary_{gt_score_type}.txt"), "w") as f:
            f.write(result.summary().as_text())
        return result, df_work
    except Exception as e:
        print(f"OLS failed for {gt_score_type}: {e}")
            
    return None, None

# --- Plotting Functions ---

def plot_variance_decomposition(model_result, df, gt_score_type, output_dir):
    """
    Plot 1: Variance Decomposition per Metric
    Splits variance into:
    - Fixed Main Effects (excluding Metric main effect, which is constant per metric)
    - Interactions (Metric-specific effects of other factors)
    - Nuisance Interactions
    - Ensemble (Random)
    - Residuals
    """
    
    # 1. Determine Model Type and Params
    is_mixed = hasattr(model_result, 'cov_re')
    
    if is_mixed:
        params = model_result.fe_params
        exog = model_result.model.exog
        exog_names = model_result.model.exog_names
        
        # Random Effects (Global estimate)
        keys, vals = extract_variance_components(model_result)
        var_random = sum(vals)
        
        # Residual (Global estimate)
        var_residual = model_result.scale
    else:
        # OLS
        params = model_result.params
        exog = model_result.model.exog
        exog_names = model_result.model.exog_names
        
        var_random = 0.0
        var_residual = model_result.scale 
        
    # 2. Decompose Fixed Effects Prediction Vectors
    n_obs = exog.shape[0]
    pred_main = np.zeros(n_obs)
    pred_inter = np.zeros(n_obs)
    pred_nuisance = np.zeros(n_obs)
    
    # Access params by name
    use_loc = hasattr(params, 'loc')
    
    for i, name in enumerate(exog_names):
        if use_loc:
            if name not in params.index: continue
            coef = params[name]
        else:
            coef = params[i]
            
        term_contribution = exog[:, i] * coef
        
        if ':' in name:
            # Interaction
            if 'metric' in name: 
                 pred_inter += term_contribution
            else:
                 pred_nuisance += term_contribution
        else:
            # Main Effect
            # We include metric main effect here, but its variance within a metric group will be 0
            pred_main += term_contribution
            
    # 3. Calculate Variances per Metric
    # Ensure df has same length/index
    if len(df) != n_obs:
        print(f"Warning: Dataframe length ({len(df)}) matches model ({n_obs})?")
        # Proceed assuming indices match if filtered correctly before
    
    # Add predictions to a temporary DF for grouping
    # We use the index from model.data.row_labels if available, or assume df index alignment
    # fit_mixed_model returns df_work which was used to create fit_df.
    
    res_df = df.copy()
    res_df['pred_main'] = pred_main
    res_df['pred_inter'] = pred_inter
    res_df['pred_nuisance'] = pred_nuisance
    
    metrics = res_df['metric'].unique()
    plot_data = []
    
    for m in metrics:
        sub = res_df[res_df['metric'] == m]
        if len(sub) < 2:
            continue
            
        v_main = np.var(sub['pred_main'])
        v_inter = np.var(sub['pred_inter'])
        v_nuis = np.var(sub['pred_nuisance'])
        
        total = v_main + v_inter + v_nuis + var_random + var_residual
        if total == 0: total = 1.0
        
        plot_data.append({
            'Metric': m,
            'Fixed Effects': v_main,
            'Metric Interactions': v_inter,
            'Nuisance Interactions': v_nuis,
            'Ensemble (Random)': var_random,
            'Residuals': var_residual,
            'Total': total
        })
        
    pct_df = pd.DataFrame(plot_data)
    if pct_df.empty:
        return

    # Normalize to percentages
    cols = ['Fixed Effects', 'Metric Interactions', 'Nuisance Interactions', 'Ensemble (Random)', 'Residuals']
    for c in cols:
        pct_df[c] = (pct_df[c] / pct_df['Total']) * 100
        
    pct_df = pct_df.set_index('Metric')
    
    # Sort metrics for consistent plotting
    ordered_metrics = get_metric_order(pct_df.index.unique())
    pct_df = pct_df.reindex(ordered_metrics)
    
    # 4. Plotting
    plt.figure(figsize=(12, 8))
    
    # Colors
    colors = ['#3498db', '#e74c3c', '#95a5a6', '#f1c40f', '#ecf0f1']
    
    pct_df[cols].plot(kind='bar', stacked=True, color=colors, width=0.8, edgecolor='black', linewidth=0.5, ax=plt.gca())
    
    plt.title(f"Variance Decomposition per Metric ({gt_score_type})", fontsize=16, fontweight='bold')
    plt.ylabel("% Variance Explained", fontsize=14)
    plt.xlabel("Metric", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Component")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_variance_decomposition.png"), dpi=300)
    plt.close()

def plot_fixed_effects_forest(model_result, df, gt_score_type, output_dir):
    """Plot 2: Fixed Effects Forest Plots"""
    params = model_result.params
    conf_int = model_result.conf_int()
    conf_int['coef'] = params
    
    if hasattr(model_result, 'cov_re'):
        fixed_params = conf_int[~conf_int.index.str.contains("Group|Var")]
    else:
        fixed_params = conf_int.copy()
        
    full_coefs_list = []
    
    # 5 panels: Metric, Class, Exp, Loss, Split
    fig, axes = plt.subplots(1, 5, figsize=(25, 8))
    terms = ['metric', 'mclass', 'exp', 'loss', 'split']
    
    for i, (term, ax) in enumerate(zip(terms, axes)):
        if term not in df.columns:
            continue
            
        if term == 'metric':
            all_levels = get_metric_order(df[term].unique())
        else:
            all_levels = sorted(df[term].unique())
        plot_data = []
        
        for level in all_levels:
            coef, lower, upper, found = get_model_coefficient(term, level, fixed_params.index, fixed_params)
            
            if not found and term == 'metric':
                 # For metric, we expect all levels to be found if using "0 + C(metric)"
                 pass

            plot_data.append({
                'level': str(level),
                'coef': coef,
                'lower': lower,
                'upper': upper
            })
            
            full_coefs_list.append({
                'Term': term,
                'Level': str(level),
                'Coefficient': coef,
                'Lower_CI': lower,
                'Upper_CI': upper,
                'Is_Reference': not found 
            })
        
        p_df = pd.DataFrame(plot_data)
        if not p_df.empty:
            p_df['y'] = range(len(p_df))
            ax.set_yticks(p_df['y'])
            ax.set_yticklabels(p_df['level'])
            ax.axvline(0, linestyle='--', color='grey')
            
            xerr = np.array([p_df['coef'] - p_df['lower'], p_df['upper'] - p_df['coef']])
            ax.errorbar(p_df['coef'], p_df['y'], xerr=xerr, 
                        fmt='o', color='black', capsize=5, markerfacecolor='none')
            
            ax.set_title(f"{term.capitalize()}")
            ax.set_xlabel("Coef Impact")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_fixed_effects_forest.png"))
    plt.close()
    
    coef_df = pd.DataFrame(full_coefs_list)
    coef_df.to_csv(os.path.join(output_dir, f"../full_coefficients_{gt_score_type}.csv"), index=False)

def plot_interaction_heatmaps(df, gt_score_type, output_dir):
    """Plot 3: Interaction Heatmaps"""
    # Metric x Class
    metrics_ordered = get_metric_order(df['metric'].unique())
    mclass_ordered = sorted(df['mclass'].unique())
    pivot_mc = df.pivot_table(index='metric', columns='mclass', values='percentile', aggfunc='mean').reindex(index=metrics_ordered, columns=mclass_ordered)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_mc, annot=True, fmt=".1f", cmap="viridis")
    plt.title(f"Interaction: Metric x Class ({gt_score_type})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_interaction_metric_class.png"))
    plt.close()
    
    # Metric x Experiment
    exp_ordered = sorted(df['exp'].unique())
    pivot_me = df.pivot_table(index='metric', columns='exp', values='percentile', aggfunc='mean').reindex(index=metrics_ordered, columns=exp_ordered)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_me, annot=True, fmt=".1f", cmap="viridis")
    plt.title(f"Interaction: Metric x Experiment ({gt_score_type})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_interaction_metric_exp.png"))
    plt.close()

    # Metric x Loss
    if 'loss' in df.columns:
        loss_ordered = sorted(df['loss'].unique())
        pivot_ml = df.pivot_table(index='metric', columns='loss', values='percentile', aggfunc='mean').reindex(index=metrics_ordered, columns=loss_ordered)
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_ml, annot=True, fmt=".1f", cmap="viridis")
        plt.title(f"Interaction: Metric x Loss ({gt_score_type})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "3_interaction_metric_loss.png"))
        plt.close()

    # Metric x Split
    if 'split' in df.columns:
        split_ordered = sorted(df['split'].unique())
        pivot_ms = df.pivot_table(index='metric', columns='split', values='percentile', aggfunc='mean').reindex(index=metrics_ordered, columns=split_ordered)
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_ms, annot=True, fmt=".1f", cmap="viridis")
        plt.title(f"Interaction: Metric x Split ({gt_score_type})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "3_interaction_metric_split.png"))
        plt.close()

def plot_random_intercepts(model_result, gt_score_type, output_dir):
    """Plot 4: Random Intercepts (Ensemble)"""
    if not hasattr(model_result, 'random_effects'):
        return
        
    random_effects = model_result.random_effects[1] # Group 1
    re_df = pd.DataFrame({'value': random_effects})
    re_df['term'] = re_df.index
    
    def parse_term(t):
        if t.startswith('ensemble['): return 'ensemble', t.split('[')[1][:-1]
        return 'other', t
        
    re_df[['type', 'level']] = re_df['term'].apply(lambda x: pd.Series(parse_term(x)))
    re_df.to_csv(os.path.join(output_dir, "random_effects_blups.csv"), index=False)
    
    # Plot only ensemble
    subset = re_df[re_df['type'] == 'ensemble'].sort_values('value')
    if subset.empty: return
    
    plt.figure(figsize=(8, max(4, len(subset)*0.2)))
    plt.scatter(subset['value'], range(len(subset)))
    plt.yticks(range(len(subset)), subset['level'], fontsize=8)
    plt.axvline(0, linestyle='--', color='grey')
    plt.title(f"Random Intercepts: Ensemble ({gt_score_type})")
    plt.xlabel("BLUP Value")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4_random_intercepts_ensemble.png"))
    plt.close()

def plot_metric_robustness_summary(rob_df, gt_score_type, output_dir):
    """Plot 5: Metric Robustness Summary (Bar Chart of Interaction Magnitudes)"""
    plt.figure(figsize=(12, 6))
    
    # Sort metrics for consistent plotting
    ordered_metrics = get_metric_order(rob_df['Metric'].unique())
    
    # We plot Interaction Sum (Lower is better/more robust)
    sns.barplot(data=rob_df, x='Metric', y='Interaction_Sum', order=ordered_metrics, palette='viridis')
    
    plt.title(f"Metric Interaction Magnitude (Lower = More Robust) - {gt_score_type}")
    plt.ylabel("Sum of Absolute Interaction Coefs")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "5_metric_robustness_summary.png"))
    plt.close()

def plot_performance_vs_robustness(rob_df, gt_score_type, output_dir):
    """Plot 6: Performance vs Robustness"""
    plt.figure(figsize=(10, 8))
    
    # Calculate percentiles
    rob_df['Performance_Percentile'] = rob_df['Performance'].rank(pct=True) * 100
    rob_df['Robustness_Percentile'] = rob_df['Robustness'].rank(pct=True) * 100
    
    # Sort metrics for consistent plotting in text labels
    ordered_metrics = get_metric_order(rob_df['Metric'].unique())
    # Reindex rob_df to ensure consistent color mapping if hue is used, and ordering for text labels
    rob_df = rob_df.set_index('Metric').reindex(ordered_metrics).reset_index()
    
    sns.scatterplot(data=rob_df, x='Performance_Percentile', y='Robustness_Percentile', hue='Metric', s=100)
    
    for i, row in rob_df.iterrows():
        plt.text(row['Performance_Percentile'] + 1, row['Robustness_Percentile'], row['Metric'], fontsize=9)
        
    plt.title(f"Performance vs Robustness ({gt_score_type})")
    plt.xlabel("Performance (Percentile Rank)")
    plt.ylabel("Robustness (Percentile Rank)")
    plt.xlim(0, 105)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "6_performance_vs_robustness.png"))
    plt.close()

def plot_3d_summary(results_dict, rob_dfs, output_dir):
    """
    Plot 7: 3D Summary Panels
    Panel 1: Performance (Mean Recovery vs CV vs Minimax)
    Panel 2: Robustness (Mean Recovery vs CV vs Minimax)
    """
    print("\nGenerating 3D Summary Panels (Performance & Robustness)...")
    
    # Check if we have the required scores
    required_scores = ['Mean_Recovery', 'CV', 'Minimax']
    for s in required_scores:
        if s not in rob_dfs:
            print(f"Skipping 3D summary: missing {s}")
            return

    # Merge dataframes
    # Start with the first one
    merged = rob_dfs[required_scores[0]][['Metric', 'Performance', 'Robustness']].copy()
    merged.columns = ['Metric', f'Perf_{required_scores[0]}', f'Rob_{required_scores[0]}']
    
    for s in required_scores[1:]:
        temp = rob_dfs[s][['Metric', 'Performance', 'Robustness']].copy()
        temp.columns = ['Metric', f'Perf_{s}', f'Rob_{s}']
        merged = pd.merge(merged, temp, on='Metric', how='inner')
        
    if merged.empty:
        print("No common metrics found across GT scores.")
        return

    # Sort metrics
    ordered_metrics = get_metric_order(merged['Metric'].unique())
    merged = merged.set_index('Metric').reindex(ordered_metrics).reset_index()
    
    # Colors
    unique_metrics = merged['Metric'].unique()
    metric_colors = {m: plt.cm.tab20(i % 20) for i, m in enumerate(unique_metrics)}
    colors = [metric_colors[m] for m in merged['Metric']]
    
    fig = plt.figure(figsize=(20, 10))
    
    # --- Panel 1: Performance ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    xs = merged[f'Perf_{required_scores[0]}']
    ys = merged[f'Perf_{required_scores[1]}']
    zs = merged[f'Perf_{required_scores[2]}']
    
    ax1.scatter(xs, ys, zs, c=colors, s=100, depthshade=True, edgecolor='k')
    
    # Labels
    for i, row in merged.iterrows():
        ax1.text(row[f'Perf_{required_scores[0]}'], 
                 row[f'Perf_{required_scores[1]}'], 
                 row[f'Perf_{required_scores[2]}'], 
                 row['Metric'], fontsize=8)
                 
    ax1.set_xlabel(f'{required_scores[0]} (Perf)')
    ax1.set_ylabel(f'{required_scores[1]} (Perf)')
    ax1.set_zlabel(f'{required_scores[2]} (Perf)')
    ax1.set_title('Performance Comparison', fontsize=16, fontweight='bold')
    
    # --- Panel 2: Robustness ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Use Log10 for robustness
    xs = np.log10(merged[f'Rob_{required_scores[0]}'] + 1e-9)
    ys = np.log10(merged[f'Rob_{required_scores[1]}'] + 1e-9)
    zs = np.log10(merged[f'Rob_{required_scores[2]}'] + 1e-9)
    
    ax2.scatter(xs, ys, zs, c=colors, s=100, depthshade=True, edgecolor='k')
    
    # Labels
    for i, row in merged.iterrows():
        ax2.text(xs[i], ys[i], zs[i], row['Metric'], fontsize=8)
        
    ax2.set_xlabel(f'{required_scores[0]} (Log Rob)')
    ax2.set_ylabel(f'{required_scores[1]} (Log Rob)')
    ax2.set_zlabel(f'{required_scores[2]} (Log Rob)')
    ax2.set_title('Robustness Comparison', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "7_3d_summary_panels.png"))
    plt.close()

def get_metric_category(metric):
    m_lower = str(metric).lower()
    if 'mse' in m_lower:
        return 'Error'
    elif 'kl' in m_lower or m_lower.startswith('work_'):
        return 'Work Done'
    elif 'cluster_' in m_lower:
        return 'Controls'
    else:
        return 'Other'

def plot_metric_fixed_effects_summary(results_dict, rob_dfs, output_dir):
    """
    Plot 8: Forest plot of Metric Fixed Effects for Mean_Recovery, CV, and Minimax.
    Grouped by Metric Category (Error, Work Done, Controls).
    """
    print("\nGenerating Metric Fixed Effects Summary Forest Plot...")
    
    target_scores = ['Mean_Recovery', 'CV', 'Minimax']
    categories = ['Error', 'Work Done', 'Controls']
    
    # Gather all metrics from the rob_dfs
    all_metrics_set = set()
    for s in target_scores:
        if s in rob_dfs:
            all_metrics_set.update(rob_dfs[s]['Metric'].unique())
            
    if not all_metrics_set:
        print("No metrics found.")
        return
        
    ordered_metrics = get_metric_order(list(all_metrics_set))
    
    # Create figure with 3 rows (Scores) x 3 columns (Categories)
    fig, axes = plt.subplots(len(target_scores), len(categories), 
                             figsize=(18, len(target_scores) * 6), 
                             sharey=False, sharex=False) # Don't share y as metrics differ per cat
    
    if len(target_scores) == 1:
        axes = np.array([axes])
        
    for i, score in enumerate(target_scores):
        if score not in results_dict:
            for j in range(len(categories)):
                axes[i, j].axis('off')
            continue
            
        res = results_dict[score]
        params = res.params
        conf_int = res.conf_int()
        conf_int['coef'] = params
        
        for j, cat in enumerate(categories):
            ax = axes[i, j]
            
            # Filter metrics for this category
            cat_metrics = [m for m in ordered_metrics if get_metric_category(m) == cat]
            
            if not cat_metrics:
                ax.axis('off')
                continue
                
            y = []
            x = []
            xerr_low = []
            xerr_high = []
            labels = []
            
            for k, metric in enumerate(cat_metrics):
                coef, lower, upper, found = get_model_coefficient('metric', metric, params.index, conf_int)
                
                if found:
                    y.append(k)
                    x.append(coef)
                    xerr_low.append(coef - lower)
                    xerr_high.append(upper - coef)
                    labels.append(metric)
            
            if not x:
                ax.axis('off')
                continue
                
            ax.errorbar(x, y, xerr=[xerr_low, xerr_high], fmt='o', color='black', capsize=5, markerfacecolor='blue')
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            
            # Title only on top row
            if i == 0:
                ax.set_title(cat, fontweight='bold', fontsize=14)
                
            # Y-label (Score Name) only on left column
            if j == 0:
                ax.set_ylabel(score, fontweight='bold', fontsize=14)
                
            ax.set_xlabel('Fixed Effect Coef (Percentile)')
            ax.grid(axis='x', linestyle='--', alpha=0.5)
            ax.axvline(50, color='red', linestyle='--', alpha=0.5) # Median rank reference
            
            ax.set_ylim(-0.5, len(labels)-0.5)
            ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "8_metric_fixed_effects_forest_grouped.png"))
    plt.close()

def plot_supp_class_preference(df, gt_score_type, output_dir, ordered_metrics):
    """Supplemental Plot 1: Class Preference (GT+alt - GT-only)"""
    if 'mclass' not in df.columns: return
    
    # Check if we have both classes
    classes = df['mclass'].unique()
    if 'GT-only' in classes and 'GT+alt' in classes:
        means = df.pivot_table(index='metric', columns='mclass', values='percentile', aggfunc='mean')
        means = means.reindex(ordered_metrics)
        
        if 'GT-only' in means.columns and 'GT+alt' in means.columns:
            diff = means['GT+alt'] - means['GT-only']
            
            plt.figure(figsize=(12, 6))
            # Color bars: Green if >0 (Prefers GT+alt), Red if <0 (Prefers GT-only)
            colors = ["#3ce78f" if x >= 0 else "#c4cc2e" for x in diff]
            
            diff.plot(kind='bar', color=colors, edgecolor='black', width=0.8)
            plt.title(f"Preference for GT+alt Ensembles ({gt_score_type})", fontsize=14, fontweight='bold')
            plt.ylabel("Mean Percentile Difference (GT+alt - GT-only)", fontsize=12)
            plt.xlabel("Metric", fontsize=12)
            plt.axhline(0, color='black', linewidth=1)
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "3_supp_class_preference.png"))
            plt.close()

def plot_supp_experiment_performance(df, gt_score_type, output_dir, ordered_metrics):
    """Supplemental Plot 2: Experiment Breakdown (Mean)"""
    if 'exp' in df.columns:
        plt.figure(figsize=(14, 7))
        sns.barplot(data=df, x='metric', y='percentile', hue='exp', 
                    order=ordered_metrics, palette='Paired', edgecolor='black', errorbar=None)
        plt.title(f"Performance by Experiment ({gt_score_type})", fontsize=14, fontweight='bold')
        plt.ylabel("Mean Percentile", fontsize=12)
        plt.xlabel("Metric", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title="Experiment", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "3_supp_experiment_performance.png"))
        plt.close()

def plot_supp_variance_comparison(df, gt_score_type, output_dir, ordered_metrics):
    """
    Supplemental Plot 3 & 4 (Merged): Variance Comparison (Loss vs Split)
    Calculates:
    - Variance of mean percentile across Losses (for each metric)
    - Variance of mean percentile across Splits (for each metric)
    Plots them as grouped bars on separate panels.
    """
    loss_var = None
    split_var = None
    
    # 1. Variance across Losses
    if 'loss' in df.columns:
        # Mean for each (metric, loss) pair
        loss_means = df.groupby(['metric', 'loss'])['percentile'].mean().reset_index()
        # Variance of those means across losses
        loss_var = loss_means.groupby('metric')['percentile'].var().reset_index()
        
    # 2. Variance across Splits
    if 'split' in df.columns:
        # Mean for each (metric, split) pair
        split_means = df.groupby(['metric', 'split'])['percentile'].mean().reset_index()
        # Variance of those means across splits
        split_var = split_means.groupby('metric')['percentile'].var().reset_index()
        
    if loss_var is None and split_var is None:
        return
        
    # Determine number of subplots
    n_plots = 0
    if loss_var is not None and not loss_var.empty: n_plots += 1
    if split_var is not None and not split_var.empty: n_plots += 1
    
    if n_plots == 0: return
    
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 7), sharey=True)
    if n_plots == 1:
        axes = [axes]
        
    curr_ax = 0
    
    # Plot Loss Variance
    if loss_var is not None and not loss_var.empty:
        ax = axes[curr_ax]
        sns.barplot(data=loss_var, x='metric', y='percentile', 
                    order=ordered_metrics, color="#e73cbf", edgecolor='black', ax=ax)
        ax.set_title(f"Variance by Loss Function ({gt_score_type})", fontsize=14, fontweight='bold')
        ax.set_ylabel("Variance of Percentile", fontsize=12)
        ax.set_xlabel("Metric", fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        curr_ax += 1
        
    # Plot Split Variance
    if split_var is not None and not split_var.empty:
        ax = axes[curr_ax]
        sns.barplot(data=split_var, x='metric', y='percentile', 
                    order=ordered_metrics, color="#37db34", edgecolor='black', ax=ax)
        ax.set_title(f"Variance by Split Type ({gt_score_type})", fontsize=14, fontweight='bold')
        if curr_ax > 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Variance of Percentile", fontsize=12)
        ax.set_xlabel("Metric", fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_supp_variance_comparison.png"))
    plt.close()

def plot_variance_decomposition_grouped(model_result, df, gt_score_type, output_dir):
    """
    Plot 1: Variance Decomposition per Metric
    Grouped by Metric Category (Error, Work Done, Controls).
    """
    
    # 1. Determine Model Type and Params
    is_mixed = hasattr(model_result, 'cov_re')
    
    if is_mixed:
        params = model_result.fe_params
        exog = model_result.model.exog
        exog_names = model_result.model.exog_names
        
        # Random Effects (Global estimate)
        keys, vals = extract_variance_components(model_result)
        var_random = sum(vals)
        
        # Residual (Global estimate)
        var_residual = model_result.scale
    else:
        # OLS
        params = model_result.params
        exog = model_result.model.exog
        exog_names = model_result.model.exog_names
        
        var_random = 0.0
        var_residual = model_result.scale 
        
    # 2. Decompose Fixed Effects Prediction Vectors
    n_obs = exog.shape[0]
    pred_main = np.zeros(n_obs)
    pred_inter = np.zeros(n_obs)
    pred_nuisance = np.zeros(n_obs)
    
    # Access params by name
    use_loc = hasattr(params, 'loc')
    
    for i, name in enumerate(exog_names):
        if use_loc:
            if name not in params.index: continue
            coef = params[name]
        else:
            coef = params[i]
            
        term_contribution = exog[:, i] * coef
        
        if ':' in name:
            # Interaction
            if 'metric' in name: 
                 pred_inter += term_contribution
            else:
                 pred_nuisance += term_contribution
        else:
            # Main Effect
            # We include metric main effect here, but its variance within a metric group will be 0
            pred_main += term_contribution
            
    # 3. Calculate Variances per Metric
    if len(df) != n_obs:
        print(f"Warning: Dataframe length ({len(df)}) matches model ({n_obs})?")
    
    res_df = df.copy()
    res_df['pred_main'] = pred_main
    res_df['pred_inter'] = pred_inter
    res_df['pred_nuisance'] = pred_nuisance
    
    metrics = res_df['metric'].unique()
    plot_data = []
    
    for m in metrics:
        sub = res_df[res_df['metric'] == m]
        if len(sub) < 2:
            continue
            
        v_main = np.var(sub['pred_main'])
        v_inter = np.var(sub['pred_inter'])
        v_nuis = np.var(sub['pred_nuisance'])
        
        total = v_main + v_inter + v_nuis + var_random + var_residual
        if total == 0: total = 1.0
        
        plot_data.append({
            'Metric': m,
            'Fixed Effects': v_main,
            'Metric Interactions': v_inter,
            'Nuisance Interactions': v_nuis,
            'Ensemble (Random)': var_random,
            'Residuals': var_residual,
            'Total': total
        })
        
    pct_df = pd.DataFrame(plot_data)
    if pct_df.empty:
        return

    # Normalize to percentages
    cols = ['Fixed Effects', 'Metric Interactions', 'Nuisance Interactions', 'Ensemble (Random)', 'Residuals']
    for c in cols:
        pct_df[c] = (pct_df[c] / pct_df['Total']) * 100
        
    # Grouping
    pct_df['Category'] = pct_df['Metric'].apply(get_metric_category)
    pct_df = pct_df[pct_df['Category'] != 'Other']
    
    categories = ['Error', 'Work Done', 'Controls']
    colors = ['#3498db', '#e74c3c', '#95a5a6', '#f1c40f', '#ecf0f1']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    for ax, cat in zip(axes, categories):
        cat_data = pct_df[pct_df['Category'] == cat].set_index('Metric')
        if cat_data.empty:
            ax.axis('off')
            continue
            
        # Sort metrics
        cat_data = cat_data.reindex(get_metric_order(cat_data.index))
        
        cat_data[cols].plot(kind='bar', stacked=True, color=colors, width=0.8, edgecolor='black', linewidth=0.5, ax=ax, legend=False)
        
        ax.set_title(cat, fontweight='bold', fontsize=14)
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        if cat == 'Error':
            ax.set_ylabel("% Variance Explained", fontsize=14)
        else:
            ax.set_ylabel("")

    # Legend on the last plot or separate
    handles, labels = axes[0].get_legend_handles_labels()
    # Semi-transparent legend inside the first plot
    axes[0].legend(handles, labels, loc='best', title="Component", framealpha=0.5)
    
    plt.suptitle(f"Variance Decomposition per Metric ({gt_score_type})", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_variance_decomposition_grouped.png"), dpi=300)
    plt.close()

def plot_supp_class_preference_grouped(df, gt_score_type, output_dir, ordered_metrics):
    """
    Supplemental Plot 1: Class Preference (GT+alt - GT-only).
    Grouped by Metric Category.
    """
    if 'mclass' not in df.columns: return
    
    # Check if we have both classes
    classes = df['mclass'].unique()
    if 'GT-only' in classes and 'GT+alt' in classes:
        means = df.pivot_table(index='metric', columns='mclass', values='percentile', aggfunc='mean')
        
        if 'GT-only' in means.columns and 'GT+alt' in means.columns:
            diff_series = means['GT+alt'] - means['GT-only']
            diff_df = diff_series.reset_index(name='diff')
            diff_df['Category'] = diff_df['metric'].apply(get_metric_category)
            diff_df = diff_df[diff_df['Category'] != 'Other']
            
            categories = ['Error', 'Work Done', 'Controls']
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
            
            for ax, cat in zip(axes, categories):
                cat_data = diff_df[diff_df['Category'] == cat].set_index('metric')
                
                if cat_data.empty:
                    ax.axis('off')
                    continue
                    
                # Sort metrics
                cat_data = cat_data.reindex(get_metric_order(cat_data.index))
                
                colors = ["#3ce78f" if x >= 0 else "#c4cc2e" for x in cat_data['diff']]
                
                cat_data['diff'].plot(kind='bar', color=colors, edgecolor='black', width=0.8, ax=ax)
                
                ax.set_title(cat, fontweight='bold', fontsize=14)
                ax.set_xlabel("")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.grid(axis='y', linestyle='--', alpha=0.5)
                ax.axhline(0, color='black', linewidth=1)
                
                if cat == 'Error':
                    ax.set_ylabel("Mean Percentile Difference (GT+alt - GT-only)", fontsize=12)
                else:
                    ax.set_ylabel("")
            
            plt.suptitle(f"Preference for Ground Truth+Alternate Ensembles ({gt_score_type})", fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "3_supp_class_preference_grouped.png"))
            plt.close()

def plot_supp_experiment_performance_grouped(df, gt_score_type, output_dir, ordered_metrics):
    """
    Supplemental Plot 2: Experiment Breakdown (Mean).
    Grouped by Metric Category.
    """
    if 'exp' not in df.columns: return
    
    # Add category to df
    df = df.copy()
    df['Category'] = df['metric'].apply(get_metric_category)
    df = df[df['Category'] != 'Other']
    
    categories = ['Error', 'Work Done', 'Controls']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    for ax, cat in zip(axes, categories):
        cat_data = df[df['Category'] == cat]
        
        if cat_data.empty:
            ax.axis('off')
            continue
            
        metrics = get_metric_order(cat_data['metric'].unique())
        
        sns.barplot(data=cat_data, x='metric', y='percentile', hue='exp', 
                    order=metrics, palette='Paired', edgecolor='black', errorbar=None, ax=ax)
        
        ax.set_title(cat, fontweight='bold', fontsize=14)
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        if cat == 'Error':
            ax.set_ylabel("Mean Percentile", fontsize=12)
        else:
            ax.set_ylabel("")
            
        if ax.get_legend():
            ax.get_legend().remove()
            
    # Common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Experiment", bbox_to_anchor=(1.02, 1), loc='upper left')
            
    plt.suptitle(f"Performance by Experiment ({gt_score_type})", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_supp_experiment_performance_grouped.png"))
    plt.close()

def plot_supp_experiment_performance_top10(df, gt_score_type, output_dir):
    """
    Supplemental Plot: Top 10 Metrics per Experiment (excluding controls).
    Displayed in descending order of performance for each experiment split as panels.
    """
    if 'exp' not in df.columns: return

    # Filter controls
    df_filtered = df.copy()
    df_filtered['Category'] = df_filtered['metric'].apply(get_metric_category)
    df_filtered = df_filtered[df_filtered['Category'] != 'Controls']
    
    experiments = sorted(df_filtered['exp'].unique(), reverse=True)
    n_exp = len(experiments)
    
    if n_exp == 0: return

    # Match colors to plot_supp_experiment_performance (Paired palette)
    # Map colors to the experiments in the order they appear (reversed)
    # so the first panel gets the first color, etc.
    palette = sns.color_palette("Paired", n_colors=len(experiments))
    exp_palette = dict(zip(experiments, palette))

    fig, axes = plt.subplots(1, n_exp, figsize=(6 * n_exp, 8), sharey=True)
    if n_exp == 1:
        axes = [axes]

    for i, exp in enumerate(experiments):
        ax = axes[i]
        exp_data = df_filtered[df_filtered['exp'] == exp]
        
        # Calculate mean percentile per metric
        metric_means = exp_data.groupby('metric')['percentile'].mean().sort_values(ascending=False)
        
        # Get top 10 metrics
        top10_metrics = metric_means.head(10).index.tolist()
        
        # Filter data for these top 10 metrics to plot
        plot_data = exp_data[exp_data['metric'].isin(top10_metrics)]
        
        # Plot
        sns.barplot(data=plot_data, x='metric', y='percentile', 
                    order=top10_metrics, color=exp_palette[exp], 
                    edgecolor='black', errorbar=None, ax=ax)
        
        # Apply hatching
        for bar, metric in zip(ax.patches, top10_metrics):
             m_lower = metric.lower()
             for key, hatch in METRIC_HASHTYPE_MAPPING.items():
                 if key in m_lower:
                     bar.set_hatch(hatch)
                     break
        
        ax.set_title(f"{exp}", fontsize=14, fontweight='bold')
        if i == 0:
            ax.set_ylabel("Mean Percentile", fontsize=12)
        else:
            ax.set_ylabel("")
            
        ax.set_xlabel("")
        ax.set_ylim(0, 105) 
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle(f"Top 10 Metrics per Experiment ({gt_score_type})", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_supp_experiment_performance_top10.png"))
    plt.close()

def plot_results_wrapper(model_result, df, gt_score_type, output_dir):
    """Orchestrate all plots for a single score."""
    if model_result is None:
        return
        
    print(f"Generating plots for {gt_score_type}...")
    plot_dir = os.path.join(output_dir, gt_score_type)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Calculate Robustness & Performance
    metrics = df['metric'].unique()
    rob_df = calculate_robustness_and_performance(model_result, metrics)
    
    # Calculate percentiles for Performance and Robustness
    rob_df['Performance_Percentile'] = rob_df['Performance'].rank(pct=True) * 100
    rob_df['Robustness_Percentile'] = rob_df['Robustness'].rank(pct=True) * 100
    
    ordered_metrics = get_metric_order(metrics)
    
    plot_variance_decomposition(model_result, df, gt_score_type, plot_dir)
    plot_variance_decomposition_grouped(model_result, df, gt_score_type, plot_dir)
    
    plot_fixed_effects_forest(model_result, df, gt_score_type, plot_dir)
    plot_interaction_heatmaps(df, gt_score_type, plot_dir)
    
    # Supplemental plots
    plot_supp_class_preference(df, gt_score_type, plot_dir, ordered_metrics)
    plot_supp_class_preference_grouped(df, gt_score_type, plot_dir, ordered_metrics)
    
    plot_supp_experiment_performance(df, gt_score_type, plot_dir, ordered_metrics)
    plot_supp_experiment_performance_grouped(df, gt_score_type, plot_dir, ordered_metrics)
    
    plot_supp_experiment_performance_top10(df, gt_score_type, plot_dir)
    
    plot_supp_variance_comparison(df, gt_score_type, plot_dir, ordered_metrics)
    
    plot_random_intercepts(model_result, gt_score_type, plot_dir)
    plot_metric_robustness_summary(rob_df, gt_score_type, plot_dir)
    plot_performance_vs_robustness(rob_df, gt_score_type, plot_dir)
    
    return rob_df

def main():
    parser = argparse.ArgumentParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output_dir = os.path.join(script_dir, "_cross_experiment_analysis")
    parser.add_argument("--output-dir", default=default_output_dir, help="Directory to save outputs")
    parser.add_argument("--exp-teaa", default=DEFAULT_EXPERIMENTS["TeaA-IsoValidation"], help="Path to TeaA-IsoValidation results")
    parser.add_argument("--exp-moprp-rw", default=DEFAULT_EXPERIMENTS["MoPrP-Reweighting"], help="Path to MoPrP-Reweighting results")
    parser.add_argument("--exp-moprp-rwbv", default=DEFAULT_EXPERIMENTS["MoPrP-RW+BV"], help="Path to MoPrP-RW+BV results")
    args = parser.parse_args()
    
    experiments = {
        "TeaA-IsoValidation": args.exp_teaa,
        "MoPrP-Reweighting": args.exp_moprp_rw,
        "MoPrP-RW+BV": args.exp_moprp_rwbv
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading data...")
    try:
        df = load_data(experiments)
    except Exception as e:
        print(f"Error: {e}")
        return
        
    print("Preprocessing...")
    df = preprocess_scores(df)
    
    df.to_csv(os.path.join(args.output_dir, "combined_processed_data.csv"), index=False)
    
    gt_scores = ['Mean_Recovery', 'CV', 'Regret', 'Minimax']
    
    results = {}
    rob_dfs = {} 
    
    for score in gt_scores:
        res, sub_df = fit_mixed_model(df, score, args.output_dir)
        if res:
            results[score] = res
            rob_df = plot_results_wrapper(res, sub_df, score, args.output_dir)
            rob_dfs[score] = rob_df
            
    plot_3d_summary(results, rob_dfs, args.output_dir)
    plot_metric_fixed_effects_summary(results, rob_dfs, args.output_dir)

if __name__ == "__main__":
    main()