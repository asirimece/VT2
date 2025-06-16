import pandas as pd

# Load all per-run CSVs
df_deep = pd.read_csv('deep4net.csv')        # must have: run, subject, accuracy
df_mtl_global = pd.read_csv('mtl_single.csv')    # run, subject, accuracy
df_mtl_cluster = pd.read_csv('mtl_n4.csv')       # run, subject, accuracy, cluster
df_tl_global = pd.read_csv('tl_single.csv')      # run, subject, accuracy
df_tl_cluster = pd.read_csv('tl_n4.csv')         # run, subject, accuracy, cluster

# Merge on run + subject
df = pd.merge(df_deep[['run','subject','accuracy']], df_mtl_global[['run','subject','accuracy']], on=['run','subject'], suffixes=('_deep','_mtl_global'))
df = pd.merge(df, df_mtl_cluster[['run','subject','accuracy','cluster']], on=['run','subject'], how='outer')
df = df.rename(columns={'accuracy': 'MTL_cluster', 'cluster': 'cluster'})
df = pd.merge(df, df_tl_global[['run','subject','accuracy']], on=['run','subject'], how='outer')
df = df.rename(columns={'accuracy': 'TL_global'})
df = pd.merge(df, df_tl_cluster[['run','subject','accuracy']], on=['run','subject'], how='outer')
df = df.rename(columns={'accuracy': 'TL_cluster'})

# Rename for clarity
df = df.rename(columns={'accuracy_deep':'Deep4Net', 'accuracy_mtl_global':'MTL_global'})

# Rearrange columns for neatness
cols = ['run', 'subject', 'cluster', 'Deep4Net', 'MTL_global', 'MTL_cluster', 'TL_global', 'TL_cluster']
df = df[cols]

df.to_csv("all_models_runs.csv", index=False)
print(df.head())
