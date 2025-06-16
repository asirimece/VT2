import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

def load_and_agg(file, value_col='accuracy', run_col='run', group_cols=['subject'], aggfunc='mean'):
    df = pd.read_csv(file)
    # If only 1 run, just take values as is; else, aggregate
    if run_col in df.columns:
        df_agg = df.groupby(group_cols).agg({value_col: [aggfunc, 'std']}).reset_index()
        # Flatten multiindex columns
        df_agg.columns = [col if isinstance(col, str) else col[0]+'_'+col[1] for col in df_agg.columns]
    else:
        # If no run column, assume single run
        df_agg = df[group_cols + [value_col]].copy()
        df_agg[value_col+'_std'] = np.nan
    return df_agg

def merge_and_plot(condition_name, df_single, df_clustered, value_col='accuracy', cluster_col='cluster'):
    # Merge on subject
    df_cmp = pd.merge(
        df_single[['subject', value_col+'_mean', value_col+'_std']],
        df_clustered[['subject', value_col+'_mean', value_col+'_std', cluster_col]],
        on='subject',
        suffixes=('_single', '_clustered')
    )
    df_cmp['accuracy_delta'] = df_cmp[f'{value_col}_mean_clustered'] - df_cmp[f'{value_col}_mean_single']

    # ---- Scatterplot: Single vs. Clustered ----
    plt.figure(figsize=(8,8))
    scatter = plt.scatter(
        df_cmp[f'{value_col}_mean_single'],
        df_cmp[f'{value_col}_mean_clustered'],
        c=df_cmp[cluster_col], cmap="tab10", s=75, edgecolor='k'
    )
    plt.plot([0, 1], [0, 1], 'k--', label="No change (y=x)")
    plt.xlabel(f"{condition_name} Accuracy (Single/Global Head)")
    plt.ylabel(f"{condition_name} Accuracy (Clustered Head, n=4)")
    plt.title(f"Subject-wise {condition_name} Accuracy: Single vs. Clustered\n(Color=Cluster Assignment)")

    # Legend for clusters
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.8)
    cluster_labels = [f"Cluster {i}" for i in sorted(df_cmp[cluster_col].unique())]
    plt.legend(handles, cluster_labels, title="Cluster", loc="lower right")
    plt.tight_layout()
    plt.savefig(f"scatter_single_vs_clustered_{condition_name.lower()}.png")
    plt.close()

    # ---- ΔAccuracy Histogram ----
    plt.figure(figsize=(6,4))
    plt.hist(df_cmp['accuracy_delta'], bins=20, color='skyblue')
    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel("ΔAccuracy (Clustered - Single)")
    plt.title(f"Distribution of Subject ΔAccuracy ({condition_name})")
    plt.tight_layout()
    plt.savefig(f"hist_delta_accuracy_{condition_name.lower()}.png")
    plt.close()

    # ---- ΔAccuracy by Cluster (Boxplot) ----
    plt.figure(figsize=(8,5))
    sns.boxplot(x=cluster_col, y='accuracy_delta', data=df_cmp, palette='pastel')
    plt.axhline(0, color='k', linestyle='--')
    plt.title(f"ΔAccuracy by Cluster ({condition_name} Clustered - Single)")
    plt.xlabel("Cluster")
    plt.ylabel("ΔAccuracy")
    plt.tight_layout()
    plt.savefig(f"boxplot_delta_accuracy_by_cluster_{condition_name.lower()}.png")
    plt.close()

    # ---- ΔAccuracy Sorted by Subject (Barplot) ----
    df_sorted = df_cmp.sort_values('accuracy_delta', ascending=False)
    plt.figure(figsize=(16,5))
    plt.bar(df_sorted['subject'].astype(str), df_sorted['accuracy_delta'], 
            color=sns.color_palette("tab10", n_colors=len(df_cmp[cluster_col].unique())))
    plt.xticks(rotation=90)
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel("Subject")
    plt.ylabel("ΔAccuracy (Clustered - Single)")
    plt.title(f"ΔAccuracy Sorted by Subject ({condition_name})")
    plt.tight_layout()
    plt.savefig(f"barplot_delta_accuracy_sorted_{condition_name.lower()}.png")
    plt.close()

    # ---- ΔAccuracy vs. Single Head Accuracy (Correlation/Scatter) ----
    plt.figure(figsize=(7,5))
    scatter2 = plt.scatter(
        df_cmp[f'{value_col}_mean_single'],
        df_cmp['accuracy_delta'],
        c=df_cmp[cluster_col], cmap='tab10', s=75, edgecolor='k'
    )
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel(f"{condition_name} Single/Global Head Accuracy")
    plt.ylabel("ΔAccuracy (Clustered - Single)")
    plt.title(f"Benefit of Clustering vs. Single {condition_name} Accuracy")
    handles2, labels2 = scatter2.legend_elements(prop="colors", alpha=0.8)
    cluster_labels2 = [f"Cluster {i}" for i in sorted(df_cmp[cluster_col].unique())]
    plt.legend(handles2, cluster_labels2, title="Cluster", loc="best")
    plt.tight_layout()
    plt.savefig(f"scatter_delta_vs_single_{condition_name.lower()}.png")
    plt.close()

    # Correlation
    corr, pval = spearmanr(df_cmp[f'{value_col}_mean_single'], df_cmp['accuracy_delta'])
    print(f"{condition_name}: Spearman correlation (single vs Δaccuracy): {corr:.3f} (p={pval:.3g})")

    # ---- Optional: Violin Plot of ΔAccuracy ----
    plt.figure(figsize=(6,4))
    sns.violinplot(y=df_cmp['accuracy_delta'], color='lightcoral', inner='quart')
    plt.axhline(0, color='k', linestyle='--')
    plt.title(f"Distribution of ΔAccuracy (Clustered - Single, {condition_name})")
    plt.ylabel("ΔAccuracy")
    plt.tight_layout()
    plt.savefig(f"violin_delta_accuracy_{condition_name.lower()}.png")
    plt.close()

    print(f"Plots for {condition_name} saved.\n")

# --------- Main Script ---------
# ---- MTL Analysis ----
df_single_mtl = load_and_agg('mtl_single.csv', value_col='accuracy', run_col='run', group_cols=['subject'])
df_clustered_mtl = load_and_agg('mtl_n4.csv', value_col='accuracy', run_col='run', group_cols=['subject','cluster'])
merge_and_plot('MTL', df_single_mtl, df_clustered_mtl, value_col='accuracy', cluster_col='cluster')

# ---- TL Analysis ----
df_single_tl = load_and_agg('tl_single.csv', value_col='accuracy', run_col='run', group_cols=['subject'])
df_clustered_tl = load_and_agg('tl_n4.csv', value_col='accuracy', run_col='run', group_cols=['subject','cluster'])
merge_and_plot('TL', df_single_tl, df_clustered_tl, value_col='accuracy', cluster_col='cluster')
