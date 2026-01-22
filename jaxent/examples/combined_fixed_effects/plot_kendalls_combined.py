import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from itertools import cycle

# Set style
sns.set_style("ticks")
sns.set_context("paper", rc={"axes.labelsize": 16, "axes.titlesize": 18, "xtick.labelsize": 12, "ytick.labelsize": 12})

score_metric_colors = {
    "loss": "#c0392b", "mse": "#2980b9", "d_mse": "#1abc9c",
    "kl": "#8e44ad", "work": "#d35400", "recovery": "#27ae60",
    "regret": "#f1c40f", "spearman": "#16a085",
}
_metric_color_cache = {}
_metric_palette_cycle = cycle(sns.color_palette("tab20"))

def get_metric_color(metric):
    key = str(metric)
    color = score_metric_colors.get(key) or score_metric_colors.get(key.lower())
    if color: return color
    if key not in _metric_color_cache:
        _metric_color_cache[key] = next(_metric_palette_cycle)
    return _metric_color_cache[key]

def get_metric_order(metrics):
    g1, g2, g3, g4 = [], [], [], []
    for m in metrics:
        m_lower = str(m).lower()
        if 'loss' in m_lower: g1.append(m)
        elif 'mse' in m_lower: g2.append(m)
        elif 'kl' in m_lower or 'work' in m_lower: g3.append(m)
        else: g4.append(m)
    return sorted(g1) + sorted(g2) + sorted(g3) + sorted(g4)

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
    mean_sum_ranks = m * (n + 1) / 2
    S = np.sum((sum_ranks - mean_sum_ranks)**2)
    
    denom = m**2 * (n**3 - n) / 12.0
    if denom == 0: return 0.0
    return S / denom

def analyze_experiment(df_exp, experiment_name):
    """
    Run FE analysis for each GT score in the experiment.
    Returns:
        w_scores: dict {metric: W_value}
        rho_scores: dict {metric: Rho_value}
    """
    print(f"Analyzing {experiment_name}...")
    
    gt_scores = df_exp['gt_score_type'].unique()
    
    # Storage for per-GT-score FE results
    # {gt_score: df_with_resid}
    fe_results = {}
    
    # Summary of Performance/Inconsistency for Spearman correlation
    # List of dicts: {'score_metric', 'gt_score', 'Performance', 'Inconsistency'}
    summary_rows = []
    
    for gt in gt_scores:
        sub_df = df_exp[df_exp['gt_score_type'] == gt].copy()
        if sub_df.empty: continue
        
        # Ensure types
        sub_df['score_metric'] = sub_df['score_metric'].astype(str)
        sub_df['split_type'] = sub_df['split_type'].astype(str)
        sub_df['ensemble'] = sub_df['ensemble'].astype(str)
        
        # Check if loss_function is relevant (has multiple values)
        has_loss = 'loss_function' in sub_df.columns and sub_df['loss_function'].nunique() > 1
        
        formula = "percentile ~ C(score_metric) + C(split_type) + C(ensemble)"
        if has_loss:
            formula += " + C(loss_function)"
            
        try:
            model = smf.ols(formula, data=sub_df).fit()
            sub_df['resid'] = model.resid
            sub_df['fitted'] = model.fittedvalues
            
            # Save for W calculation
            fe_results[gt] = sub_df
            
            # Aggregate for Spearman (Perf vs Inc per metric)
            # Performance = Mean of fitted
            # Inconsistency = RMS of resid
            grp = sub_df.groupby('score_metric')
            perf = grp['fitted'].mean()
            inc = grp['resid'].apply(lambda x: np.sqrt((x**2).mean()))
            
            for m in perf.index:
                summary_rows.append({
                    'score_metric': m,
                    'gt_score': gt,
                    'Performance': perf[m],
                    'Inconsistency': inc[m]
                })
                
        except Exception as e:
            print(f"  FE failed for {gt}: {e}")
            
    # --- Calculate Kendall's W (Combined Ranks) ---
    w_scores = {}
    
    # Get all metrics
    all_metrics = set()
    for df in fe_results.values():
        all_metrics.update(df['score_metric'].unique())
        
    for m in all_metrics:
        # Collect (Ensemble, Split) ranks from each GT score
        ranks_list = []
        
        # We need a common index (Ensemble, Split) across GT scores
        # Aggregate to (Ensemble, Split) level first
        
        common_index = None
        subsets = {}
        valid_metric = True
        
        for gt, df in fe_results.items():
            sub = df[df['score_metric'] == m].copy()
            if sub.empty:
                valid_metric = False
                break
            
            grp = sub.groupby(['ensemble', 'split_type'])
            agg = grp.agg({
                'percentile': 'mean', # Or fitted? Reference used val_transformed which correlates with percentile
                'resid': lambda x: np.sqrt((x**2).mean())
            }).reset_index()
            
            agg['id'] = agg['ensemble'] + "_" + agg['split_type']
            agg = agg.set_index('id')
            subsets[gt] = agg
            
            if common_index is None:
                common_index = agg.index
            else:
                common_index = common_index.intersection(agg.index)
        
        if not valid_metric or common_index is None or len(common_index) < 2:
            w_scores[m] = np.nan
            continue
            
        for gt in fe_results.keys():
            if gt not in subsets: continue
            dat = subsets[gt].loc[common_index]
            
            # Rank Performance (High is best -> Descending)
            r_perf = stats.rankdata(-dat['percentile'], method='min')
            
            # Rank Inconsistency (Low is best -> Ascending)
            r_inc = stats.rankdata(dat['resid'], method='min')
            
            # Combined Rank
            r_comb = stats.rankdata((r_perf + r_inc)/2.0, method='min')
            ranks_list.append(r_comb)
            
        if len(ranks_list) > 1:
            w = calculate_kendalls_w(np.array(ranks_list))
            w_scores[m] = w
        else:
            w_scores[m] = np.nan

    # --- Calculate Spearman Rho (Perf vs Inc across GT scores) ---
    rho_scores = {}
    summary_df = pd.DataFrame(summary_rows)
    
    for m in all_metrics:
        m_data = summary_df[summary_df['score_metric'] == m]
        if len(m_data) > 2: # Need at least 3 points for correlation to mean anything, but 4 is available
            # Note: Performance is "Higher is Better" (percentile)
            # Inconsistency is "Lower is Better" (RMS resid)
            # Ideally, we want High Perf to have Low Inc -> Negative Correlation is "Good"?
            # Or are we checking if they are correlated in general?
            # Reference script: `rho, p = stats.spearmanr(perf, inc)`
            # It just plots Rho.
            
            # Rank Performance (Percentile)
            # Rank Inconsistency (RMS)
            # We correlate the raw values (or ranks, Spearman does ranks internally)
            rho, _ = stats.spearmanr(m_data['Performance'], -m_data['Inconsistency'])
            if np.isnan(rho): rho = 0
            rho_scores[m] = rho
        else:
            rho_scores[m] = np.nan
            
    return w_scores, rho_scores

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

def plot_grouped_boxplot(df, y_col, title, filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Categorize metrics
    df['Category'] = df['score_metric'].apply(get_metric_category)
    
    # Filter out 'Other' if any
    df = df[df['Category'] != 'Other']
    
    # Define order of categories
    categories = ['Error', 'Work Done', 'Controls']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    for ax, cat in zip(axes, categories):
        cat_data = df[df['Category'] == cat]
        metrics = sorted(cat_data['score_metric'].unique())
        metric_palette = {m: get_metric_color(m) for m in metrics}
        
        if not cat_data.empty:
            sns.boxplot(data=cat_data, x='score_metric', y=y_col, order=metrics,
                        palette=metric_palette, showfliers=False, ax=ax)
            sns.stripplot(data=cat_data, x='score_metric', y=y_col, order=metrics,
                          color='black', alpha=0.5, jitter=True, ax=ax)
        
        ax.set_title(cat, fontweight='bold', fontsize=14)
        ax.set_xlabel("")
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        if cat == 'Error':
            ax.set_ylabel(title.split(":")[0], fontweight='bold') # Short label
        else:
            ax.set_ylabel("")

    plt.suptitle(title, fontweight='bold', fontsize=16)
    
    # Set ylim on the first axis (shared)
    title_l = (title or "").lower()
    y_col_l = (y_col or "").lower()
    is_rho = ('rho' in y_col_l) or ('spearman' in title_l) or ('rho' in title_l)
    if is_rho:
        axes[0].set_ylim(-1.1, 1.1)
        # Add a light dashed horizontal line at y=0 across all subplots for Rho plots
        for ax in axes:
            ax.axhline(0.0, linestyle='--', color='gray', linewidth=1.0, alpha=0.5)
    else:
        axes[0].set_ylim(0, 1.1)
        
    plt.tight_layout()
    
    out_path = os.path.join(script_dir, filename)
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")

def main():
    data_path = "/home/alexi/Documents/JAX-ENT/cross_experiment_analysis/combined_processed_data.csv"
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Filter out transformed
    df = df[~df['score_metric'].astype(str).str.contains('transformed', case=False, na=False)]

    # Filter out unwanted clusters
    unwanted_clusters = ['cluster_2','cluster_3', 'cluster_4', 'cluster_5']
    df = df[~df['score_metric'].isin(unwanted_clusters)]
    
    experiments = df['experiment'].unique()
    
    all_w = []
    all_rho = []
    
    for exp in experiments:
        df_exp = df[df['experiment'] == exp]
        w_scores, rho_scores = analyze_experiment(df_exp, exp)
        
        for m, w in w_scores.items():
            if not np.isnan(w):
                all_w.append({'Experiment': exp, 'score_metric': m, 'Kendall_W': w})
                
        for m, rho in rho_scores.items():
            if not np.isnan(rho):
                all_rho.append({'Experiment': exp, 'score_metric': m, 'Spearman_Rho': rho})
                
    df_w = pd.DataFrame(all_w)
    df_rho = pd.DataFrame(all_rho)
    
    # Plot Kendall's W
    if not df_w.empty:
        plot_grouped_boxplot(df_w, 'Kendall_W', 
                             "Concordance across GT Scores", 
                             "combined_kendalls_w_boxplot.png")
        
    # Plot Spearman Rho
    if not df_rho.empty:
        plot_grouped_boxplot(df_rho, 'Spearman_Rho', 
                             "Spearman Correlation: Performance vs Consistency", 
                             "combined_spearman_rho_consistency_boxplot.png")

if __name__ == "__main__":
    main()
