"""
This script plots the recovery scores from the selection analysis performed on each of the metrics provided to analyse_scores_mixed_linear_model.py

The two sets of plots should be of publication quality suitable for a high-impact journal.

Loads in the model_selection_performance_summary.csv  before and after convergence filtering.

Plots Recovery Score (y) vs Metric (x) with ensembles hued and split types on separate panels and each score as a different figure.

Plot before, after convergence filtering and the different (Before-After) convergence filtering. using a welchs t-test for each ensembles and annotate the plots with significance stars. 
Ensure that reccovery axis is always 0-100.


Then plot the log complement of the p-values -log(1-p) between ensembles for each metric - hue by split with metrics on the x-axis and bayes factor on the y-axis.

Plot both before and after convergence filtering and the difference (Before-After) convergence filtering.   


    # Save summary CSV
    if selection_stats_list:
        full_stats = pd.concat(selection_stats_list, ignore_index=True)
        # Reorder columns for readability
        cols = ['score_metric', 'direction'] + [c for c in full_stats.columns if c not in ['score_metric', 'direction', 'mean', 'std', 'count']] + ['mean', 'std', 'count']
        full_stats = full_stats[cols]
        
        out_path = os.path.join(output_dir, "model_selection_performance_summary.csv")
        full_stats.to_csv(out_path, index=False)
        print(f"Saved model selection performance summary to {out_path}")

        
"""

import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import statsmodels.formula.api as smf
from itertools import cycle

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
    "ISO_TRI": "indigo",
    "ISO_BI": "saddlebrown",
}

split_colors = {
    "Random": "fuchsia",
    "Sequence": "black",
    "Non-Redundant": "green",
    "Spatial": "grey",
    "Flat": "orange",
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

score_metric_colors = {
	"loss": "#c0392b",
	"mse": "#2980b9",
	"d_mse": "#1abc9c",
	"kl": "#8e44ad",
	"work": "#d35400",
	"recovery": "#27ae60",
	"regret": "#f1c40f",
	"spearman": "#16a085",
}
_metric_color_cache = {}
_metric_palette_cycle = cycle(sns.color_palette("tab20"))

def get_metric_color(metric):
	key = str(metric)
	color = score_metric_colors.get(key) or score_metric_colors.get(key.lower())
	if color:
		return color
	if key not in _metric_color_cache:
		_metric_color_cache[key] = next(_metric_palette_cycle)
	return _metric_color_cache[key]


def p_to_stars(p_val):
    """Convert p-value to significance stars."""
    if pd.isna(p_val):
        return 'n.s.'
    elif p_val < 0.0001:
        return '****'
    elif p_val < 0.001:
        return '***'
    elif p_val < 0.01:
        return '**'
    elif p_val < 0.05:
        return '*'
    elif p_val < 0.1:
        return '‡'  # Unicode double dagger (U+2021)
    elif p_val < 0.25:
        return '†'  # Unicode single dagger (U+2020)
    else:
        return 'n.s.'


def ttest_from_stats(m1, s1, n1, m2, s2, n2):
    """
    Calculate Welch's t-test from summary statistics.
    """
    # Welch's t-test
    vn1 = s1**2 / n1
    vn2 = s2**2 / n2
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (m1 - m2) / np.sqrt(vn1 + vn2)
        df = (vn1 + vn2)**2 / (vn1**2 / (n1 - 1) + vn2**2 / (n2 - 1))
    
    # Two-sided p-value
    p = 2 * (1 - stats.t.cdf(np.abs(t), df))
    return t, p

def calculate_kendalls_w(data_matrix):
    """
    data_matrix: m raters x n subjects
    Returns Kendall's W (0 to 1).
    """
    data_matrix = np.array(data_matrix)
    m, n = data_matrix.shape
    if m <= 1 or n <= 1:
        return 0.0
    
    # Convert to ranks (row-wise)
    ranks = np.apply_along_axis(stats.rankdata, 1, data_matrix)
    
    sum_ranks = np.sum(ranks, axis=0)
    # S = sum of squared deviations of sum_ranks from mean
    mean_sum_ranks = m * (n + 1) / 2
    S = np.sum((sum_ranks - mean_sum_ranks)**2)
    
    denom = m**2 * (n**3 - n) / 12.0
    if denom == 0: return 0.0
    W = S / denom
    return W

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


def load_and_process_data(before_path, after_path, include_transformed=False):
    """
    Load before and after CSVs and compute difference stats.
    """
    print(f"Loading Before: {before_path}")
    df_before = pd.read_csv(before_path)
    print(f"Loading After: {after_path}")
    df_after = pd.read_csv(after_path)
    
    # Standardize columns: rename recovery_percent_mean -> mean, etc.
    def standardize_df(df):
        target_base = None
        if 'recovery_percent_mean' in df.columns:
            target_base = 'recovery_percent'
        elif 'recovery_score_mean' in df.columns:
            target_base = 'recovery_score'
        else:
            # Fallback: look for any 'recovery' related mean
            for c in df.columns:
                if 'recovery' in c and c.endswith('_mean'):
                    target_base = c[:-5]
                    break
        
        if target_base:
            print(f"Identified target metric: {target_base}")
            rename_map = {
                f"{target_base}_mean": "mean",
                f"{target_base}_std": "std",
                f"{target_base}_count": "count",
                f"{target_base}_min": "min"
            }
            df = df.rename(columns=rename_map)

        # Check for spearman columns and standardize them to spearman_mean/std/count
        spearman_cols = [c for c in df.columns if 'spearman' in c.lower()]
        if spearman_cols:
            print(f"Found spearman columns: {spearman_cols}")
            
            # Identify mean, std, count
            s_mean = next((c for c in spearman_cols if 'mean' in c.lower()), None)
            s_std = next((c for c in spearman_cols if 'std' in c.lower()), None)
            s_count = next((c for c in spearman_cols if 'count' in c.lower()), None)
            
            # If no explicit mean column but we have columns, and one doesn't look like std/count
            if not s_mean:
                 candidates = [c for c in spearman_cols if c not in [s_std, s_count] and c is not None]
                 if candidates:
                     s_mean = candidates[0]

            rename_map_spearman = {}
            if s_mean: rename_map_spearman[s_mean] = 'spearman_mean'
            if s_std: rename_map_spearman[s_std] = 'spearman_std'
            if s_count: rename_map_spearman[s_count] = 'spearman_count'
            
            if rename_map_spearman:
                print(f"Renaming spearman columns: {rename_map_spearman}")
                df = df.rename(columns=rename_map_spearman)
        else:
             # Debug: print columns if spearman not found
             print("No spearman columns found in dataframe.")
        
        # Keep only grouping keys and stats to avoid merging on other metrics
        known_keys = ['ensemble', 'split_type', 'loss_function', 'bv_reg_function', 'bv_reg_value', 'score_metric', 'direction', 'mean', 'std', 'count', 'min', 'spearman_mean', 'spearman_std', 'spearman_count']
        keep_cols = [c for c in df.columns if c in known_keys]
        return df[keep_cols]

    df_before = standardize_df(df_before)
    df_after = standardize_df(df_after)

    if not include_transformed:
        print("Filtering out transformed metrics...")
        if 'score_metric' in df_before.columns:
            df_before = df_before[~df_before['score_metric'].astype(str).str.contains('transformed', case=False, na=False)]
        if 'score_metric' in df_after.columns:
            df_after = df_after[~df_after['score_metric'].astype(str).str.contains('transformed', case=False, na=False)]

    df_before['condition'] = 'Before Filtering'
    df_after['condition'] = 'After Filtering'
    
    # Compute Difference (Before - After)
    # Identify merge columns (exclude stats)
    exclude_cols = ['mean', 'std', 'count', 'condition', 'direction', 'min', 'spearman_mean', 'spearman_std', 'spearman_count']
    merge_cols = [c for c in df_before.columns if c not in exclude_cols]
    
    print(f"Merging on: {merge_cols}")
    
    df_merged = pd.merge(
        df_after, 
        df_before, 
        on=merge_cols, 
        suffixes=('_after', '_before'),
        how='inner'
    )
    
    df_diff = df_merged.copy()
    df_diff['mean'] = df_diff['mean_before'] - df_diff['mean_after']
    # Error propagation for difference: sqrt(s1^2 + s2^2)
    df_diff['std'] = np.sqrt(df_diff['std_after']**2 + df_diff['std_before']**2)
    # Count is min of both
    df_diff['count'] = np.minimum(df_diff['count_after'], df_diff['count_before'])
    df_diff['condition'] = 'Difference (Before - After)'
    
    # Keep structure consistent
    cols_to_keep = merge_cols + ['mean', 'std', 'count', 'condition']
    if 'direction_after' in df_diff.columns:
        df_diff['direction'] = df_diff['direction_after']
        cols_to_keep.append('direction')
        
    df_diff = df_diff[cols_to_keep]
    
    # Calculate CV and approx error
    # SE_cv approx = CV / sqrt(2n)
    df_after['cv'] = df_after['std'] / df_after['mean']
    df_after['cv_err'] = df_after['cv'] / np.sqrt(2 * df_after['count'])

    df_before['cv'] = df_before['std'] / df_before['mean']
    df_before['cv_err'] = df_before['cv'] / np.sqrt(2 * df_before['count'])
    
    return df_before, df_after, df_diff


def plot_minimax_panel(df, output_dir, filename):
    """
    Plot minimum of mean recovery across losses (Main Bar)
    and minimum of min recovery across losses (Sub Bar).
    """
    if 'loss_function' not in df.columns:
        print("Cannot plot minimax: 'loss_function' column missing.")
        return

    split_types = sorted(df['split_type'].unique())
    metrics = get_metric_order(df['score_metric'].unique())
    ensembles = sorted(df['ensemble'].unique())
    
    n_splits = len(split_types)
    fig, axes = plt.subplots(1, n_splits, figsize=(6 * n_splits, 6), sharey=False)
    if n_splits == 1:
        axes = [axes]
        
    width = 0.8 / len(ensembles)
    x = np.arange(len(metrics))
    
    for i, split in enumerate(split_types):
        ax = axes[i]
        split_data = df[df['split_type'] == split]
        
        for j, ens in enumerate(ensembles):
            ens_data = split_data[split_data['ensemble'] == ens]
            
            min_means = []
            abs_mins = []
            
            for m in metrics:
                # Get all rows for this metric (across losses)
                rows = ens_data[ens_data['score_metric'] == m]
                if not rows.empty:
                    # Minimum of the means
                    min_mean = rows['mean'].min()
                    min_means.append(min_mean)
                    
                    # Minimum of the mins (if available)
                    if 'min' in rows.columns:
                        abs_min = rows['min'].min()
                    else:
                        abs_min = 0 
                    abs_mins.append(abs_min)
                else:
                    min_means.append(0)
                    abs_mins.append(0)
            
            offset = (j - len(ensembles)/2 + 0.5) * width
            color = ensemble_colors.get(ens, 'grey')
            
            # Main bar (Min Mean)
            ax.bar(x + offset, min_means, width, label=ens if i==0 else "", 
                   color=color, capsize=4, edgecolor='black', alpha=0.6, linewidth=1)
            
            # Sub bar (Abs Min) - Narrower and darker/solid
            if any(a > 0 for a in abs_mins):
                 ax.bar(x + offset, abs_mins, width*0.5, 
                   color=color, edgecolor='black', alpha=1.0, linewidth=1, hatch='//')

        ax.set_title(f"{split_name_mapping.get(split, split)}", fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_xlabel("Metric", fontweight='bold')
        if i == 0:
            ax.set_ylabel("Min Recovery Score (%)", fontweight='bold')
        
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_ylim(0, 110)
        
    # Legend
    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.0, 1.0), title="Ensemble")
    
    plt.suptitle("Minimax Recovery (Min Mean & Abs Min across Losses)", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()


def plot_score_panel(df, y_col, y_err_col, ylabel, title, output_dir, filename, metric_col='score_metric'):
    """
    Plot scores with split types on separate panels.
    """
    split_types = sorted(df['split_type'].unique())
    metrics = get_metric_order(df[metric_col].unique())
    ensembles = sorted(df['ensemble'].unique())
    
    n_splits = len(split_types)
    fig, axes = plt.subplots(1, n_splits, figsize=(6 * n_splits, 6), sharey=False)
    if n_splits == 1:
        axes = [axes]
        
    width = 0.8 / len(ensembles)
    x = np.arange(len(metrics))
    
    for i, split in enumerate(split_types):
        ax = axes[i]
        split_data = df[df['split_type'] == split]
        
        for j, ens in enumerate(ensembles):
            ens_data = split_data[split_data['ensemble'] == ens]
            
            means = []
            stds = []
            counts = []
            for m in metrics:
                row = ens_data[ens_data[metric_col] == m]
                if not row.empty:
                    means.append(row[y_col].values[0])
                    if y_err_col and y_err_col in row.columns:
                        stds.append(row[y_err_col].values[0])
                        counts.append(row['count'].values[0])
                    else:
                        stds.append(0)
                        counts.append(0)
                else:
                    means.append(0)
                    stds.append(0)
                    counts.append(0)
            
            offset = (j - len(ensembles)/2 + 0.5) * width
            color = ensemble_colors.get(ens, 'grey')
            
            ax.bar(x + offset, means, width, yerr=stds if y_err_col else None, label=ens, 
                   color=color, capsize=4, edgecolor='black', alpha=0.9, linewidth=1)

        # Significance testing between ensembles (if exactly 2 and we have error info)
        if len(ensembles) == 2 and y_err_col:
            ens1, ens2 = ensembles
            for k, m in enumerate(metrics):
                d1 = split_data[(split_data['ensemble'] == ens1) & (split_data[metric_col] == m)]
                d2 = split_data[(split_data['ensemble'] == ens2) & (split_data[metric_col] == m)]
                
                if not d1.empty and not d2.empty:
                    t, p = ttest_from_stats(
                        d1[y_col].values[0], d1[y_err_col].values[0], d1['count'].values[0],
                        d2[y_col].values[0], d2[y_err_col].values[0], d2['count'].values[0]
                    )
                    
                    star = p_to_stars(p)
                    h1 = d1[y_col].values[0] + d1[y_err_col].values[0]
                    h2 = d2[y_col].values[0] + d2[y_err_col].values[0]
                    h = max(h1, h2)
                    if h < 0: h = 0
                    
                    star_fontsize = 16 if star != 'n.s.' else 10
                    ax.text(k, h + (1 if 'Regret' not in title else 1), 
                            star, ha='center', va='bottom', fontsize=star_fontsize, fontweight='bold')

        ax.set_title(f"{split_name_mapping.get(split, split)}", fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_xlabel("Metric", fontweight='bold')
        if i == 0:
            ax.set_ylabel(ylabel, fontweight='bold')
        
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Set Y limit for Recovery
        if "Mean Recovery" in title:
            ax.set_ylim(0, 110)
        
    # Legend
    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.0, 1.0), title="Ensemble")
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()


def plot_p_values(df, name, output_dir):
    """
    Plot -log10(1-p) (Insignificance) between ensembles for a single condition.
    """
    data_rows = []
    
    split_types = df['split_type'].unique()
    metrics = get_metric_order(df['score_metric'].unique())
    ensembles = sorted(df['ensemble'].unique())
    
    if len(ensembles) != 2:
        print(f"Skipping p-value plot for {name}: need exactly 2 ensembles, got {len(ensembles)}")
        return
        
    ens1, ens2 = ensembles
    
    for split in split_types:
        for m in metrics:
            d1 = df[(df['ensemble'] == ens1) & (df['split_type'] == split) & (df['score_metric'] == m)]
            d2 = df[(df['ensemble'] == ens2) & (df['split_type'] == split) & (df['score_metric'] == m)]
            
            if not d1.empty and not d2.empty:
                t, p = ttest_from_stats(
                    d1['mean'].values[0], d1['std'].values[0], d1['count'].values[0],
                    d2['mean'].values[0], d2['std'].values[0], d2['count'].values[0]
                )
                
                # Use -log10(1-p) as proxy for Insignificance
                # Clip p to avoid log(0) if p=1
                p_clipped = min(p, 1.0 - 1e-15)
                val = -np.log10(1 - p_clipped)
                
                data_rows.append({
                    'Condition': name,
                    'Split Type': split_name_mapping.get(split, split),
                    'Metric': m,
                    'Insignificance': val,
                    'p-value': p
                })

    if not data_rows:
        return

    p_df = pd.DataFrame(data_rows)
    
    # Plot
    plt.figure(figsize=(10, 6)) # dims: width, height
    ax = sns.barplot(
        data=p_df,
        x='Metric',
        y='Insignificance',
        hue='Split Type',
        palette=split_colors,
        edgecolor='black',
        linewidth=1,
        order=metrics
    )
    
    # Add significance threshold lines
    # Levels from p_to_stars: 0.25, 0.1, 0.05, 0.01, 0.001, 0.0001
    # "0.25 thick to 0.0001 thin"
    thresholds = [0.25, 0.1, 0.05, 0.01, 0.001, 0.0001]
    symbols = ['†', '‡', '*', '**', '***', '****']
    linewidths = np.linspace(2.5, 0.5, len(thresholds))
    
    custom_lines = []
    for p_thresh, lw, sym in zip(thresholds, linewidths, symbols):
        y_val = -np.log10(1 - p_thresh)
        ax.axhline(y_val, color='black', linestyle='--', linewidth=lw, alpha=0.6)
        custom_lines.append(
            Line2D([0], [0], color='black', lw=lw, linestyle='--', alpha=0.6, label=f'p<{p_thresh} ({sym})')
        )
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    # ax.grid(axis='y', alpha=0.3)
    
    ax.set_xlabel("Metric", fontweight='bold')
    ax.set_ylabel("Insignificance (-log10(1-p))", fontweight='bold')
    # if name != "Difference":
    ax.set_yscale('log')
    ax.set_title(f"Statistical Insignificance of Ensemble Differences ({name})", fontsize=18, fontweight='bold')
    
    # Combine legend handles
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + custom_lines, bbox_to_anchor=(1.05, 1), loc='upper left', title="Split Type / Significance")
    
    out_path = os.path.join(output_dir, f"insignificance_comparison_{name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()


def aggregate_df(df):
    """
    Aggregate dataframe over loss_function if present.
    Groups by all columns except loss_function, mean, std, count, cv.
    Returns dataframe with mean of mean, std, count, cv.
    """
    if df is None or df.empty:
        return df
        
    if 'loss_function' not in df.columns:
        return df.copy()
        
    # Identify columns to group by (all except stats and loss_function)
    exclude_cols = ['mean', 'std', 'count', 'loss_function', 'cv', 'cv_err', 'cv_std', 'spearman_mean', 'spearman_std', 'spearman_count']
    group_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Aggregate
    grouped = df.groupby(group_cols, as_index=False)
    
    # Mean of means
    agg_cols = ['mean', 'std', 'count']
    if 'spearman_mean' in df.columns:
        agg_cols.append('spearman_mean')
    if 'spearman_std' in df.columns:
        agg_cols.append('spearman_std')
    if 'spearman_count' in df.columns:
        agg_cols.append('spearman_count')
        
    df_agg = grouped[agg_cols].mean()
    
    if 'cv' in df.columns:
        # Mean CV
        df_agg['cv'] = grouped['cv'].mean()['cv']
        # Std of CV (variation across loss functions)
        df_agg['cv_std'] = grouped['cv'].std()['cv'].fillna(0)
        
    return df_agg


def compute_minimax_df(df):
    """
    Compute minimax dataframe: Min of Mean Recovery across loss functions.
    """
    if 'loss_function' not in df.columns:
        return df.copy()
    # Group by Ensemble, Split, Metric
    # Take min of mean
    grp = df.groupby(['ensemble', 'split_type', 'score_metric'], as_index=False)['mean'].min()
    return grp


def plot_rank_panel(df, value_col, title, output_dir, filename, ascending=False, transform_func=None):
    """
    Plot ensemble-normalised ranks for a given score.
    Ranks metrics within each (Ensemble, Split, Loss) group.
    Aggregates ranks across Ensembles and Losses for plotting.
    """
    local_df = df.copy()
    
    # Map split types to display names for consistent coloring
    local_df['split_type'] = local_df['split_type'].map(lambda x: split_name_mapping.get(x, x))
    
    if transform_func:
        local_df[value_col] = local_df[value_col].apply(transform_func)
        
    # Grouping for ranking: Ensemble + Split + (Loss if exists)
    rank_groups = ['ensemble', 'split_type']
    if 'loss_function' in local_df.columns:
        rank_groups.append('loss_function')
        
    # Calculate Rank
    # ascending=False -> Higher Value is Rank 1 (Best)
    local_df['rank'] = local_df.groupby(rank_groups)[value_col].rank(ascending=ascending, method='min')
    
    metrics = get_metric_order(local_df['score_metric'].unique())
    
    plt.figure(figsize=(12, 6))
    # sns.barplot will aggregate (mean + ci) across ensembles and losses
    ax = sns.barplot(data=local_df, x='score_metric', y='rank', hue='split_type', 
                palette=split_colors, order=metrics, edgecolor='black', linewidth=1)
    
    ax.set_title(f"{title} (Aggregated)", fontweight='bold')
    ax.set_xlabel("Metric", fontweight='bold')
    ax.set_ylabel("Rank (1 is Best)", fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.invert_yaxis() # Best rank (1) at top
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Move legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Split Type")
    
    out_path = os.path.join(output_dir, f"{filename}_rank_aggregated.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()


def run_fixed_effects_for_score(df, value_col, score_name, output_dir, transform_func=None, has_loss=True):
    """
    Run fixed effects analysis for a specific score (e.g. Mean, CV, Regret).
    Steps:
    1. Transform value.
    2. Compute Ensemble-Normalised Percentile.
    3. Fit Fixed Effects Model on Percentile.
    4. Plot Performance vs Inconsistency (Scatter).
    5. Calculate Inconsistency Percentile.
    6. Plot Performance vs Inconsistency (Bar).
    """
    print(f"\n--- Running Fixed Effects Analysis: {score_name} ---")
    
    local_df = df.copy()
    
    # Basic Cleaning
    required_cols = ['score_metric', 'ensemble', 'split_type']
    if has_loss:
        required_cols.append('loss_function')
    required_cols.append(value_col)
    
    local_df = local_df.dropna(subset=required_cols)
    
    # 1. Transform
    if transform_func:
        local_df['val_transformed'] = local_df[value_col].apply(transform_func)
    else:
        local_df['val_transformed'] = local_df[value_col]
        
    # 2. Ensemble-Normalised Percentile
    local_df['percentile'] = local_df.groupby('ensemble')['val_transformed'].rank(pct=True) * 100
    
    # Ensure categorical types
    local_df['score_metric'] = local_df['score_metric'].astype(str)
    local_df['ensemble'] = local_df['ensemble'].astype(str)
    local_df['split_type'] = local_df['split_type'].astype(str)
    if has_loss:
        local_df['loss_function'] = local_df['loss_function'].astype(str)
        
    # 3. Fit Fixed Effects Model
    formula = "percentile ~ C(score_metric) + C(split_type) + C(ensemble)"
    if has_loss:
        formula += " + C(loss_function)"
        
    try:
        model = smf.ols(formula, data=local_df).fit()
    except Exception as e:
        print(f"Model fitting failed for {score_name}: {e}")
        return None

    # 4. Analyze Residuals & Main Effects
    local_df['resid'] = model.resid
    local_df['fitted'] = model.fittedvalues
    
    # Inconsistency = RMS of Residuals
    consistency_df = local_df.groupby('score_metric')['resid'].apply(lambda x: np.sqrt((x**2).mean())).reset_index()
    consistency_df.rename(columns={'resid': 'Inconsistency'}, inplace=True)
    
    # Performance = Mean of Fitted Values
    performance_df = local_df.groupby('score_metric')['fitted'].mean().reset_index()
    performance_df.rename(columns={'fitted': 'Performance_Percentile'}, inplace=True)
    
    results = pd.merge(performance_df, consistency_df, on='score_metric')
    
    # 5. Calculate Inconsistency Percentile (from -RMS)
    # Lower RMS -> Better -> Higher Percentile
    results['Inconsistency_Percentile'] = results['Inconsistency'].apply(lambda x: -x).rank(pct=True) * 100
    
    results = results.sort_values('Performance_Percentile', ascending=False)
    
    out_csv = os.path.join(output_dir, f"fixed_effects_{score_name}.csv")
    results.to_csv(out_csv, index=False)
    
    # Plot Scatter
    plt.figure(figsize=(10, 8))
    metrics = get_metric_order(results['score_metric'].unique())
    metric_palette = {m: get_metric_color(m) for m in metrics}
    
    sns.scatterplot(data=results, x='Performance_Percentile', y='Inconsistency', hue='score_metric', 
                    s=150, edgecolor='black', palette=metric_palette, alpha=0.8)
    
    for i, row in results.iterrows():
        plt.text(row['Performance_Percentile'], row['Inconsistency'] + 0.05, 
                 row['score_metric'], fontsize=9, ha='center', va='bottom')

    plt.xlabel("Average Performance (Percentile Rank)", fontweight='bold', fontsize=14)
    plt.ylabel("Inconsistency (RMS Residuals)", fontweight='bold', fontsize=14)
    plt.title(f"{score_name}: Performance vs. Consistency\n(Fixed Effects on Ensemble-Norm Percentiles)", fontweight='bold', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(0, 105)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Metric")
    plt.tight_layout()
    
    out_plot = os.path.join(output_dir, f"fixed_effects_scatter_{score_name}.png")
    plt.savefig(out_plot, dpi=300, bbox_inches='tight')
    print(f"Saved {out_plot}")
    plt.close()
    
    # 6. Plot Bar Chart (Performance vs Inconsistency Ranks)
    melted = results.melt(id_vars=['score_metric'], 
                          value_vars=['Performance_Percentile', 'Inconsistency_Percentile'],
                          var_name='Type', value_name='Percentile')
    # Clean up Type names
    melted['Type'] = melted['Type'].replace({
        'Performance_Percentile': 'Performance', 
        'Inconsistency_Percentile': 'Inconsistency'
    })
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x='score_metric', y='Percentile', hue='Type', 
                order=metrics, palette={'Performance': 'skyblue', 'Inconsistency': 'salmon'},
                edgecolor='black', linewidth=1)
    
    plt.title(f"{score_name}: Performance vs Inconsistency (Percentile Ranks)", fontweight='bold')
    plt.xlabel("Metric", fontweight='bold')
    plt.ylabel("Percentile Rank", fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    out_bar = os.path.join(output_dir, f"fixed_effects_bar_{score_name}.png")
    plt.savefig(out_bar, dpi=300, bbox_inches='tight')
    print(f"Saved {out_bar}")
    plt.close()
    
    # Return both summary and the full df with residuals for concordance calculation
    return results, local_df


def calculate_concordance_maps(results_dict):
    """
    Calculate Kendall's W for Performance, Inconsistency, and Combined
    for each metric across the 4 GT scores.
    
    results_dict: { 'GT_Score_Name': (summary_df, full_df_with_resid) }
    """
    print("\n--- Calculating Concordance Maps (Performance, Inconsistency, Combined) ---")
    
    # 1. Consolidate data
    gt_scores = list(results_dict.keys())
    if len(gt_scores) < 2:
        return {k: {} for k in ['Performance', 'Inconsistency', 'Combined']}

    # Extract full DFs
    full_dfs = {k: v[1] for k, v in results_dict.items()}
    
    # Get list of all metrics
    all_metrics = set()
    for df in full_dfs.values():
        all_metrics.update(df['score_metric'].unique())
        
    perf_map = {}
    inc_map = {}
    comb_map = {}
    
    for m in all_metrics:
        # For this metric, gather the vectors from each GT score
        
        # Storage for ranks
        perf_ranks = []
        inc_ranks = []
        comb_ranks = []
        
        valid_metric = True
        
        # We need to ensure we are comparing the exact same set of (ensemble, split) points.
        # Since some GT scores (Mean, CV) have multiple loss functions per (Ensemble, Split)
        # and others (Minimax) have only one, we must AGGREGATE to the (Ensemble, Split) level
        # to compare them.
        
        common_index = None
        subsets = {}
        
        for gt in gt_scores:
            df = full_dfs[gt]
            sub = df[df['score_metric'] == m].copy()
            
            if sub.empty:
                valid_metric = False
                break
            
            # Aggregate to (Ensemble, Split) level
            # Performance: Mean of transformed values
            # Inconsistency: RMS of residuals
            grp = sub.groupby(['ensemble', 'split_type'])
            agg_sub = grp.agg({
                'val_transformed': 'mean',
                'resid': lambda x: np.sqrt((x**2).mean())
            }).reset_index()
            
            # Create ID
            agg_sub['id'] = agg_sub['ensemble'].astype(str) + "_" + agg_sub['split_type'].astype(str)
            agg_sub = agg_sub.set_index('id')
            
            subsets[gt] = agg_sub
            
            if common_index is None:
                common_index = agg_sub.index
            else:
                common_index = common_index.intersection(agg_sub.index)
        
        if not valid_metric or common_index is None or len(common_index) < 2:
            perf_map[m] = 0.0
            inc_map[m] = 0.0
            comb_map[m] = 0.0
            continue
        
        for gt in gt_scores:
            # Align to common index
            sub = subsets[gt].loc[common_index]
            
            # 1. Performance Vector (val_transformed)
            # ALL val_transformed are "High is Best" (due to transformations in analysis step)
            val = sub['val_transformed'].values
            # Rank: Descending (Largest value = Rank 1)
            r_perf = stats.rankdata(-val, method='min')
            perf_ranks.append(r_perf)
            
            # 2. Inconsistency Vector (RMS resid)
            # Low residual = Best.
            resid = sub['resid'].values
            # Rank: Ascending (Smallest value = Rank 1)
            r_inc = stats.rankdata(resid, method='min')
            inc_ranks.append(r_inc)
            
            # 3. Combined Vector (Average of Ranks)
            # (r_perf + r_inc) / 2 -> Smallest is Best
            r_comb_score = (r_perf + r_inc) / 2.0
            r_comb = stats.rankdata(r_comb_score, method='min')
            comb_ranks.append(r_comb)
        
        # Calculate Ws
        perf_map[m] = calculate_kendalls_w(np.array(perf_ranks))
        inc_map[m] = calculate_kendalls_w(np.array(inc_ranks))
        comb_map[m] = calculate_kendalls_w(np.array(comb_ranks))
            
    return {'Performance': perf_map, 'Inconsistency': inc_map, 'Combined': comb_map}


def run_fixed_effects_analysis(df_before, df_diff, df_minimax, output_dir):
    """
    Orchestrate FE analysis for Mean, CV, Regret, Minimax.
    Returns a dictionary of result tuples (summary, full_df).
    """
    results_dict = {}
    
    # 1. Mean Recovery (Higher is better)
    res = run_fixed_effects_for_score(
        df_before, 'mean', "Mean_Recovery", output_dir, 
        transform_func=None, has_loss=True
    )
    if res is not None: results_dict['Mean_Recovery'] = res
    
    # 2. CV (Lower is better -> -log(CV) higher is better)
    res = run_fixed_effects_for_score(
        df_before, 'cv', "CV", output_dir, 
        transform_func=lambda x: -np.log(x + 1e-10), has_loss=True
    )
    if res is not None: results_dict['CV'] = res
    
    # 3. Regret (Lower is better -> -Regret higher is better)
    # Regret is in df_diff['mean']
    res = run_fixed_effects_for_score(
        df_diff, 'mean', "Regret", output_dir,
        transform_func=lambda x: -x, has_loss=True
    )
    if res is not None: results_dict['Regret'] = res
    
    # 4. Minimax (Higher is better)
    # Minimax has no loss_function column usually (aggregated out)
    res = run_fixed_effects_for_score(
        df_minimax, 'mean', "Minimax", output_dir,
        transform_func=None, has_loss=False
    )
    if res is not None: results_dict['Minimax'] = res
    
    return results_dict

def plot_aggregated_analysis(results_dict, concordance_maps, output_dir):
    """
    Plot aggregated Performance, Inconsistency, and Combined Ranks across GT Scores.
    Also plot Spearman correlations.
    """
    print("\n--- Running Aggregated Analysis ---")
    
    all_rows = []
    for gt_score, (df, _) in results_dict.items():
        df_temp = df.copy()
        df_temp['GT_Score'] = gt_score
        all_rows.append(df_temp)
        
    if not all_rows:
        print("No results to aggregate.")
        return
        
    full_df = pd.concat(all_rows, ignore_index=True)
    
    full_df['Combined_Percentile'] = (full_df['Performance_Percentile'] + full_df['Inconsistency_Percentile']) / 2.0
    
    metrics = get_metric_order(full_df['score_metric'].unique())
    gt_scores = sorted(full_df['GT_Score'].unique())
    
    # Define plotting helper
    def plot_grouped_ranks(y_col, title_prefix, filename_suffix, concordance_key):
        plt.figure(figsize=(14, 7))
        
        # Global concordance (Raters=GT Scores, Subjects=Metrics)
        # Note: This is the OLD global W (agreement on metric ranking)
        pivot_df = full_df.pivot(index='GT_Score', columns='score_metric', values=y_col)
        pivot_df = pivot_df[metrics]
        if pivot_df.isnull().values.any():
            pivot_df = pivot_df.fillna(0)
        global_w = calculate_kendalls_w(pivot_df.values)
        
        ax = sns.barplot(data=full_df, x='score_metric', y=y_col, hue='GT_Score', 
                         order=metrics, palette='viridis', edgecolor='black', linewidth=1)
        
        # Annotate bars with PER-METRIC concordance (agreement on model ranking)
        # We place the annotation above the group of bars for each metric
        
        # Get current y-axis limit to position text
        y_max = 110
        
        # Iterate over metrics (x-axis)
        for i, m in enumerate(metrics):
            w = concordance_maps[concordance_key].get(m, 0.0)
            
            # Find max height in this group to position text
            group_data = full_df[full_df['score_metric'] == m]
            if not group_data.empty:
                max_h = group_data[y_col].max()
            else:
                max_h = 0
                
            # Annotate
            ax.text(i, max_h + 2, f"W={w:.2f}", ha='center', va='bottom', 
                    fontsize=9, fontweight='bold', color='darkred')

        plt.title(f"{title_prefix} (Global Metric Rank W = {global_w:.3f})", fontweight='bold', fontsize=16)
        plt.xlabel("Metric", fontweight='bold')
        plt.ylabel("Percentile Rank", fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 115) # Increased for annotations
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="GT Score")
        plt.tight_layout()
        
        out_path = os.path.join(output_dir, f"aggregated_{filename_suffix}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved {out_path}")
        plt.close()
        
    # 1. Performance
    plot_grouped_ranks('Performance_Percentile', "Aggregated Performance Ranks", "performance_ranks", 'Performance')
    
    # 2. Inconsistency
    plot_grouped_ranks('Inconsistency_Percentile', "Aggregated Inconsistency Ranks", "inconsistency_ranks", 'Inconsistency')
    
    # 3. Combined
    plot_grouped_ranks('Combined_Percentile', "Aggregated Combined Ranks", "combined_ranks", 'Combined')
    
    # 4. Spearman Correlation per Metric
    # Correlation between Performance and Inconsistency across GT Scores
    corr_rows = []
    for m in metrics:
        m_data = full_df[full_df['score_metric'] == m]
        if len(m_data) > 1:
            perf = m_data['Performance_Percentile'].values
            inc = m_data['Inconsistency_Percentile'].values
            rho, p = stats.spearmanr(perf, inc)
            if np.isnan(rho): rho = 0
            corr_rows.append({'score_metric': m, 'Spearman_Rho': rho, 'p_value': p})
            
    if corr_rows:
        corr_df = pd.DataFrame(corr_rows)
        metric_palette = {m: get_metric_color(m) for m in metrics}
        plt.figure(figsize=(12, 6))
        sns.barplot(data=corr_df, x='score_metric', y='Spearman_Rho', order=metrics,
                    palette=metric_palette, edgecolor='black', linewidth=1)
        
        plt.title("Spearman Correlation between Performance and Inconsistency (across GT Scores)", fontweight='bold')
        plt.xlabel("Metric", fontweight='bold')
        plt.ylabel("Spearman's Rho", fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.axhline(0, color='black', linewidth=1)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.ylim(-1.1, 1.1)
        plt.tight_layout()
        
        out_corr = os.path.join(output_dir, "aggregated_spearman_correlation.png")
        plt.savefig(out_corr, dpi=300, bbox_inches='tight')
        print(f"Saved {out_corr}")
        plt.close()

    # 5. Combined Kendall's W Plot
    w_rows = []
    for m in metrics:
        w = concordance_maps['Combined'].get(m, 0.0)
        w_rows.append({'score_metric': m, 'Kendall_W': w})
        
    if w_rows:
        w_df = pd.DataFrame(w_rows)
        plt.figure(figsize=(12, 6))
        metric_palette = {m: get_metric_color(m) for m in metrics}
        sns.barplot(data=w_df, x='score_metric', y='Kendall_W', order=metrics,
                    palette=metric_palette, edgecolor='black', linewidth=1)
        
        plt.title("Concordance of Combined Ranks (Kendall's W across GT Scores)", fontweight='bold', fontsize=16)
        plt.xlabel("Metric", fontweight='bold')
        plt.ylabel("Kendall's W", fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        out_w = os.path.join(output_dir, "aggregated_combined_concordance_w.png")
        plt.savefig(out_w, dpi=300, bbox_inches='tight')
        print(f"Saved {out_w}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot recovery scores and selection analysis.")
    parser.add_argument("--before-csv", required=True, help="Path to summary CSV before filtering")
    parser.add_argument("--after-csv", required=True, help="Path to summary CSV after filtering")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--include-transformed", action="store_true", help="Include metrics with 'transformed' in their name")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading data...")
    try:
        df_before, df_after, df_diff = load_and_process_data(args.before_csv, args.after_csv, args.include_transformed)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # 1. Plot Overall (Aggregated over loss functions if present)
    print("\n--- Plotting Overall Results (Aggregated) ---")
    df_after_agg = aggregate_df(df_after)
    df_before_agg = aggregate_df(df_before)
    df_diff_agg = aggregate_df(df_diff)
    
    print("Plotting GT Scores (Overall)...")
    # Recivert Scores -> Before
    plot_score_panel(df_before_agg, 'mean', 'std', "Mean Recovery Score (%)", "Mean Recovery Score (Overall)", args.output_dir, "mean_recovery_score_overall")
    # CV -> Before
    plot_score_panel(df_before_agg, 'cv', 'cv_std', "Coefficient of Variation", "Coefficient of Variation (Overall)", args.output_dir, "cv_overall")
    # Regret -> Diff
    plot_score_panel(df_diff_agg, 'mean', 'std', "Filtering Regret (%)", "Filtering Regret (Overall)", args.output_dir, "filtering_regret_overall")

    if 'spearman_mean' in df_before_agg.columns:
        plot_score_panel(df_before_agg, 'spearman_mean', 'spearman_std', "Spearman Correlation", "Spearman Correlation (Overall)", args.output_dir, "spearman_correlation_overall")

    plot_minimax_panel(df_before, args.output_dir, "minimax_recovery_score")
    
    # Minimax DF computation for analysis
    df_minimax = compute_minimax_df(df_before)

    # --- Fixed Effects Analysis ---
    fe_results = run_fixed_effects_analysis(df_before, df_diff, df_minimax, args.output_dir)
    
    # --- Calculate Concordance Maps (Perf, Inc, Combined) ---
    concordance_maps = calculate_concordance_maps(fe_results)
    
    # --- Aggregated Analysis (New) ---
    plot_aggregated_analysis(fe_results, concordance_maps, args.output_dir)

    print("\n--- Plotting Ranks ---")
    # 1. Mean Recovery (Higher is better -> ascending=False)
    plot_rank_panel(df_before, 'mean', "Rank of Mean Recovery", args.output_dir, "rank_mean_recovery", ascending=False)
    
    # 2. CV (Lower is better -> -log(CV) higher is better -> ascending=False)
    plot_rank_panel(df_before, 'cv', "Rank of CV (-log)", args.output_dir, "rank_cv", ascending=False, 
                    transform_func=lambda x: -np.log(x + 1e-10))
    
    # 3. Regret (Lower is better -> -Regret higher is better -> ascending=False)
    # df_diff['mean'] is Regret (positive value = loss).
    plot_rank_panel(df_diff, 'mean', "Rank of Regret (-val)", args.output_dir, "rank_regret", ascending=False,
                    transform_func=lambda x: -x)
                    
    # 4. Minimax (Higher is better -> ascending=False)
    df_minimax = compute_minimax_df(df_before)
    plot_rank_panel(df_minimax, 'mean', "Rank of Minimax Recovery", args.output_dir, "rank_minimax", ascending=False)

    save_gt_scores(df_before, df_diff, df_minimax, args.output_dir)


def save_gt_scores(df_before, df_diff, df_minimax, output_dir):
    """
    Save GT scores to a long-format CSV for cross-experiment analysis.
    """
    print("\n--- Saving GT Scores for Mixed Effects Model ---")
    data_rows = []
    
    # 1. Mean Recovery (from df_before 'mean')
    if 'mean' in df_before.columns:
        for _, row in df_before.iterrows():
            data_rows.append({
                'ensemble': row['ensemble'],
                'split_type': row['split_type'],
                'loss_function': row.get('loss_function', 'Aggr'),
                'score_metric': row['score_metric'],
                'gt_score_type': 'Mean_Recovery',
                'value': row['mean']
            })

    # 2. CV (from df_before 'cv')
    if 'cv' in df_before.columns:
        for _, row in df_before.iterrows():
             data_rows.append({
                'ensemble': row['ensemble'],
                'split_type': row['split_type'],
                'loss_function': row.get('loss_function', 'Aggr'),
                'score_metric': row['score_metric'],
                'gt_score_type': 'CV',
                'value': row['cv']
            })

    # 3. Regret (from df_diff 'mean')
    if df_diff is not None and 'mean' in df_diff.columns:
        for _, row in df_diff.iterrows():
             data_rows.append({
                'ensemble': row['ensemble'],
                'split_type': row['split_type'],
                'loss_function': row.get('loss_function', 'Aggr'),
                'score_metric': row['score_metric'],
                'gt_score_type': 'Regret',
                'value': row['mean']
            })
            
    # 4. Minimax (from df_minimax 'mean')
    if df_minimax is not None and 'mean' in df_minimax.columns:
        for _, row in df_minimax.iterrows():
             data_rows.append({
                'ensemble': row['ensemble'],
                'split_type': row['split_type'],
                'loss_function': 'Aggr', # Minimax aggregates over loss
                'score_metric': row['score_metric'],
                'gt_score_type': 'Minimax',
                'value': row['mean']
            })
            
    out_df = pd.DataFrame(data_rows)
    out_path = os.path.join(output_dir, "gt_scores_long.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
