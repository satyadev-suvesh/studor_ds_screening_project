"""
Task 1 Visualizations
=====================
Generates all plots for the PDF report and analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'Distinction': '#2ecc71',
    'Pass': '#3498db',
    'Fail': '#e74c3c',
    'Withdrawn': '#95a5a6',
    'Success': '#2ecc71',
    'At-Risk': '#e74c3c'
}
ARCHETYPE_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']


def plot_feature_distributions(features_df, output_dir):
    """Box plots showing feature distributions by outcome group."""
    
    from task1_scoring import FEATURE_COLS
    
    df = features_df.copy()
    df['outcome_group'] = df['final_result'].map({
        'Distinction': 'Success', 'Pass': 'Success',
        'Fail': 'At-Risk', 'Withdrawn': 'At-Risk'
    })
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, feat in enumerate(FEATURE_COLS):
        ax = axes[i]
        data = df[[feat, 'outcome_group']].replace([np.inf, -np.inf], np.nan).dropna()
        
        # Clip outliers for visualization
        q1, q99 = data[feat].quantile(0.01), data[feat].quantile(0.99)
        data = data[(data[feat] >= q1) & (data[feat] <= q99)]
        
        sns.boxplot(data=data, x='outcome_group', y=feat, ax=ax,
                   palette=COLORS, width=0.5)
        ax.set_title(feat.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    # Hide last empty subplot
    if len(FEATURE_COLS) < len(axes):
        axes[-1].set_visible(False)
    
    fig.suptitle('Feature Distributions: Success vs At-Risk Students', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved feature_distributions.png")


def plot_archetype_trajectories(archetype_result, output_dir, method_name='kmeans'):
    """Plot mean engagement trajectory for each archetype."""
    
    archetypes = archetype_result['archetypes']
    n_clusters = len(archetypes)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: all archetypes overlaid
    ax = axes[0]
    for c, info in archetypes.items():
        traj = info['mean_trajectory']
        weeks = np.arange(1, len(traj) + 1)
        color = ARCHETYPE_COLORS[c % len(ARCHETYPE_COLORS)]
        ax.plot(weeks, traj, linewidth=2.5, color=color,
               label=f"{info['label']} (n={info['n_students']})")
        ax.fill_between(weeks, traj * 0.85, traj * 1.15, alpha=0.1, color=color)
    
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Engagement Score', fontsize=12)
    ax.set_title(f'Student Archetypes ({method_name.upper()})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3, label='Midpoint')
    
    # Right: outcome breakdown per archetype
    ax = axes[1]
    cluster_labels = []
    outcome_data = {'Distinction': [], 'Pass': [], 'Fail': [], 'Withdrawn': []}
    
    for c in sorted(archetypes.keys()):
        info = archetypes[c]
        cluster_labels.append(f"{info['label']}\n(n={info['n_students']})")
        for outcome in outcome_data:
            outcome_data[outcome].append(info['outcome_distribution'].get(outcome, 0))
    
    x = np.arange(len(cluster_labels))
    bottom = np.zeros(len(cluster_labels))
    
    for outcome in ['Distinction', 'Pass', 'Fail', 'Withdrawn']:
        values = outcome_data[outcome]
        ax.bar(x, values, bottom=bottom, label=outcome, color=COLORS[outcome], width=0.6)
        bottom += np.array(values)
    
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_labels, fontsize=9)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Outcome Distribution by Archetype', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    fig.savefig(output_dir / f'archetypes_{method_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved archetypes_{method_name}.png")


def plot_shap_heatmap(shap_importance, output_dir):
    """Heatmap showing how feature importance shifts across weeks."""
    
    from task1_scoring import FEATURE_COLS
    
    weeks = sorted(shap_importance.keys())
    
    matrix = []
    for w in weeks:
        row = [shap_importance[w].get(f, 0) for f in FEATURE_COLS]
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sns.heatmap(matrix.T, 
                xticklabels=[f'W{w}' for w in weeks],
                yticklabels=[f.replace('_', ' ').title() for f in FEATURE_COLS],
                cmap='YlOrRd', annot=True, fmt='.2f', 
                linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Feature Importance (SHAP)'})
    
    ax.set_title('How Engagement Drivers Shift Over the Semester', 
                fontsize=13, fontweight='bold')
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'shap_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved shap_heatmap.png")


def plot_score_by_outcome(scored_df, output_dir):
    """Mean engagement score trajectory by final outcome."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: absolute score
    ax = axes[0]
    for outcome in ['Distinction', 'Pass', 'Fail', 'Withdrawn']:
        subset = scored_df[scored_df.final_result == outcome]
        weekly_mean = subset.groupby('week')['engagement_score'].mean()
        weekly_std = subset.groupby('week')['engagement_score'].std()
        
        weeks = weekly_mean.index
        ax.plot(weeks, weekly_mean.values, linewidth=2, color=COLORS[outcome], label=outcome)
        ax.fill_between(weeks, 
                        (weekly_mean - weekly_std).values, 
                        (weekly_mean + weekly_std).values,
                        alpha=0.15, color=COLORS[outcome])
    
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Engagement Score', fontsize=12)
    ax.set_title('Engagement Score by Final Outcome', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    
    # Right: velocity
    ax = axes[1]
    for outcome in ['Distinction', 'Pass', 'Fail', 'Withdrawn']:
        subset = scored_df[scored_df.final_result == outcome]
        weekly_vel = subset.groupby('week')['score_velocity'].mean()
        ax.plot(weekly_vel.index, weekly_vel.values, linewidth=2, 
               color=COLORS[outcome], label=outcome)
    
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Score Velocity (weekly change)', fontsize=12)
    ax.set_title('Engagement Velocity by Final Outcome', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'score_by_outcome.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved score_by_outcome.png")


def plot_feature_weights(model_info, output_dir):
    """Bar chart of learned feature weights."""
    
    weights = model_info['feature_weights']
    sorted_feats = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    
    feats = [f[0].replace('_', ' ').title() for f in sorted_feats]
    vals = [f[1] for f in sorted_feats]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(feats[::-1], vals[::-1], color='#3498db', edgecolor='white')
    
    # Highlight top 3
    for bar in bars[-3:]:
        bar.set_color('#e74c3c')
    
    ax.set_xlabel('Learned Weight (Normalized)', fontsize=12)
    ax.set_title('Feature Weights Learned from Outcome Calibration', 
                fontsize=13, fontweight='bold')
    
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'feature_weights.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved feature_weights.png")


def plot_sample_student_trajectories(scored_df, output_dir, n_samples=3):
    """Show individual student trajectories as examples."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, outcome in enumerate(['Distinction', 'Pass', 'Fail', 'Withdrawn']):
        ax = axes[i]
        subset = scored_df[scored_df.final_result == outcome]
        
        # Pick students with enough data
        stu_weeks = subset.groupby(['code_module', 'code_presentation', 'id_student'])['week'].count()
        valid_students = stu_weeks[stu_weeks >= 8].index.tolist()
        
        if len(valid_students) > n_samples:
            sampled = np.random.choice(len(valid_students), n_samples, replace=False)
            valid_students = [valid_students[s] for s in sampled]
        
        for mod, pres, sid in valid_students:
            stu_data = subset[
                (subset.code_module == mod) & 
                (subset.code_presentation == pres) & 
                (subset.id_student == sid)
            ].sort_values('week')
            ax.plot(stu_data.week, stu_data.engagement_score, 
                   alpha=0.7, linewidth=1.5)
        
        # Mean trajectory
        weekly_mean = subset.groupby('week')['engagement_score'].mean()
        ax.plot(weekly_mean.index, weekly_mean.values, 
               color=COLORS[outcome], linewidth=3, linestyle='--', label='Mean')
        
        ax.set_title(f'{outcome} Students', fontsize=12, fontweight='bold',
                    color=COLORS[outcome])
        ax.set_xlabel('Week')
        ax.set_ylabel('Engagement Score')
        ax.set_ylim(0, 100)
        ax.legend()
    
    fig.suptitle('Sample Student Trajectories by Outcome', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / 'sample_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved sample_trajectories.png")


def plot_normalized_trajectories(scored_df, output_dir):
    """Secondary view: trajectories by % of course completed."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Bin pct_complete into 5% buckets
    scored_df = scored_df.copy()
    scored_df['pct_bin'] = (scored_df['pct_complete'] // 5) * 5
    
    for outcome in ['Distinction', 'Pass', 'Fail', 'Withdrawn']:
        subset = scored_df[scored_df.final_result == outcome]
        binned = subset.groupby('pct_bin')['engagement_score'].mean()
        ax.plot(binned.index, binned.values, linewidth=2.5, 
               color=COLORS[outcome], label=outcome)
    
    ax.set_xlabel('Course Completion (%)', fontsize=12)
    ax.set_ylabel('Mean Engagement Score', fontsize=12)
    ax.set_title('Engagement Trajectories (Normalized by Course Length)', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 100)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'normalized_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved normalized_trajectories.png")


def generate_all_plots(results: dict):
    """Generate all Task 1 visualizations."""
    
    output_dir = Path(results['output_dir'])
    
    print("\nGenerating visualizations...")
    
    plot_feature_distributions(results['features_df'], output_dir)
    plot_feature_weights(results['model_info'], output_dir)
    plot_score_by_outcome(results['scored_df'], output_dir)
    plot_sample_student_trajectories(results['scored_df'], output_dir)
    plot_normalized_trajectories(results['scored_df'], output_dir)
    
    plot_archetype_trajectories(results['kmeans_archetypes'], output_dir, 'kmeans')
    if results.get('dtw_archetypes'):
        plot_archetype_trajectories(results['dtw_archetypes'], output_dir, 'dtw')
    
    if results.get('shap_importance'):
        plot_shap_heatmap(results['shap_importance'], output_dir)
    
    print("\nAll visualizations saved!")


if __name__ == "__main__":
    # This would be called after running the main pipeline
    print("Run task1_engagement_scoring.py first, then call generate_all_plots(results)")
