import pandas as pd
from scipy.stats import mannwhitneyu

# --- Load and label data ---
df1 = pd.read_csv("results/fbcsp_ncluster4_full_all_model_comparison.csv")
df1['experiment'] = 'ncluster=1'

df4 = pd.read_csv("results/fbcsp_singlecluster_full_all_model_comparison.csv")
df4['experiment'] = 'ncluster=4'

df = pd.concat([df1, df4], ignore_index=True)

# --- Comparison Function ---
def compare_with_baseline(data, cluster_label):
    print(f"\n=== {cluster_label.upper()} ===")
    baseline = data[(data['experiment'] == cluster_label) & (data['model'] == 'baseline')]['accuracy']
    tl = data[(data['experiment'] == cluster_label) & (data['model'] == 'tl')]['accuracy']
    mtl = data[(data['experiment'] == cluster_label) & (data['model'] == 'mtl')]['accuracy']
    
    print("Baseline vs TL:", mannwhitneyu(baseline, tl, alternative='two-sided'))
    print("Baseline vs MTL:", mannwhitneyu(baseline, mtl, alternative='two-sided'))

def compare_model_across_experiments(data, model_name):
    acc1 = data[(data['experiment'] == 'ncluster=1') & (data['model'] == model_name)]['accuracy']
    acc4 = data[(data['experiment'] == 'ncluster=4') & (data['model'] == model_name)]['accuracy']
    
    print(f"\n=== {model_name.upper()} in ncluster=1 vs ncluster=4 ===")
    print(mannwhitneyu(acc1, acc4, alternative='two-sided'))

# --- Run Comparisons ---
compare_with_baseline(df, 'ncluster=1')
compare_with_baseline(df, 'ncluster=4')
compare_model_across_experiments(df, 'tl')
compare_model_across_experiments(df, 'mtl')
