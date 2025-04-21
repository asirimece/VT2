#!/usr/bin/env python
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

# Configuration
PIPELINE_ROOTS = {
    'scratch': 'tl_results/scratch',
    'global':  'tl_results/global',
    'cluster': 'tl_results/cluster'
}
SUBJECTS = [1,2,3,4,5,6,7,8,9]  # adjust to your subjects
N_RUNS = 3
OUTDIR = 'evaluation'

# 1) Load wrappers and compute per-run metrics
records = []
for pipeline, root in PIPELINE_ROOTS.items():
    for subj in SUBJECTS:
        for run in range(N_RUNS):
            # Path: root/<subject>/run_<run>/tl_<subject>_results.pkl
            wrapper_path = os.path.join(root, str(subj), f'run_{run}', f'tl_{subj}_results.pkl')
            if not os.path.isfile(wrapper_path):
                print(f"Missing {wrapper_path}, skipping")
                continue
            wrapper = pickle.load(open(wrapper_path, 'rb'))
            gt = wrapper.ground_truth
            pr = wrapper.predictions
            acc = accuracy_score(gt, pr)
            kappa = cohen_kappa_score(gt, pr)
            cm = confusion_matrix(gt, pr)
            records.append({
                'pipeline': pipeline,
                'subject': subj,
                'run': run,
                'accuracy': acc,
                'kappa': kappa,
                'confusion_matrix': cm
            })

# 2) Build DataFrame
df = pd.DataFrame(records)
if df.empty:
    raise RuntimeError("No data loaded; check your pipeline roots and file structure.")

os.makedirs(OUTDIR, exist_ok=True)
df.to_csv(os.path.join(OUTDIR, 'summary_raw.csv'), index=False)

# 3) Boxplots for accuracy and kappa
plt.figure()
df.boxplot(column='accuracy', by='pipeline')
plt.title('TL Accuracy by Pipeline')
plt.suptitle('')
plt.savefig(os.path.join(OUTDIR, 'accuracy_boxplot.png'))

plt.figure()
df.boxplot(column='kappa', by='pipeline')
plt.title('TL Cohen Kappa by Pipeline')
plt.suptitle('')
plt.savefig(os.path.join(OUTDIR, 'kappa_boxplot.png'))

# 4) Scatter Global vs Cluster (subject-level mean)
pivot = df.groupby(['pipeline','subject'])['accuracy'].mean().unstack(level=0)
x = pivot['global']; y = pivot['cluster']
plt.figure()
plt.scatter(x, y)
maxv = max(x.max(), y.max())
plt.plot([0, maxv], [0, maxv], '--', color='gray')
plt.xlabel('Global TL Accuracy')
plt.ylabel('Cluster TL Accuracy')
plt.title('Global vs Cluster TL (mean over runs)')
plt.savefig(os.path.join(OUTDIR, 'global_vs_cluster_scatter.png'))

# 5) Wilcoxon signed-rank tests
pairs = [('scratch','global'), ('scratch','cluster'), ('global','cluster')]
with open(os.path.join(OUTDIR, 'wilcoxon.txt'), 'w') as f:
    for a,b in pairs:
        arr_a = df[df.pipeline==a].groupby('subject')['accuracy'].mean()
        arr_b = df[df.pipeline==b].groupby('subject')['accuracy'].mean()
        stat, p = wilcoxon(arr_a, arr_b)
        line = f"Wilcoxon {a} vs {b}: stat={stat:.3f}, p={p:.3e}\n"
        print(line.strip())
        f.write(line)

# 6) Average normalized confusion matrix heatmaps
for pipeline, root in PIPELINE_ROOTS.items():
    cms = []
    for entry in records:
        if entry['pipeline'] != pipeline:
            continue
        cms.append(entry['confusion_matrix'])
    if not cms:
        continue
    cms = np.stack(cms)  # shape (n_samples, n_classes, n_classes)
    # normalize each cm row-wise
    cms_norm = np.array([cm.astype(float)/cm.sum(axis=1, keepdims=True) for cm in cms])
    avg_cm = cms_norm.mean(axis=0)
    plt.figure()
    im = plt.imshow(avg_cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Normalized Avg Confusion Matrix: {pipeline}')
    plt.colorbar(im)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(os.path.join(OUTDIR, f'cm_{pipeline}.png'))

# 7) Performance by original cluster membership
import yaml
from lib.pipeline.cluster.cluster import SubjectClusterer
# load clustering config
clust_cfg = yaml.safe_load(open('config/experiment/mtl.yaml'))['experiment']['clustering']
# perform clustering to retrieve assignments
cw = SubjectClusterer('dump/features.pkl', clust_cfg).cluster_subjects(method=clust_cfg['method'])
cluster_map = {sid: lab for sid, lab in cw.labels.items()}

# merge cluster assignments into DataFrame
df['cluster'] = df['subject'].map(cluster_map)

# boxplot: accuracy by cluster
plt.figure()
df.boxplot(column='accuracy', by='cluster')
plt.title('TL Accuracy by Original Cluster')
plt.suptitle('')
plt.savefig(os.path.join(OUTDIR, 'accuracy_by_cluster.png'))

# statistical test: Kruskal-Wallis across clusters
from scipy.stats import kruskal
groups = [group['accuracy'].values for _, group in df.groupby('cluster')]
stat, p = kruskal(*groups)
with open(os.path.join(OUTDIR, 'cluster_stat_test.txt'), 'w') as f:
    f.write(f"Kruskal-Wallis across clusters: stat={stat:.3f}, p={p:.3e}")

# 8) Matched vs. mismatched cluster‑backbones
# assume per‑cluster TL runs have been saved under directories cluster0, cluster1, etc.
cluster_ids = sorted(df['cluster'].unique())
cluster_pipelines = [f"cluster{cid}" for cid in cluster_ids]
# Load mean accuracy of each subject under each cluster backbone
# Build a small table: index=subject, columns=cluster pipelines
subj_cluster_acc = {}
for pid in cluster_pipelines:
    # path root for this pipeline
    root = os.path.join('tl_results', pid)
    # gather per-subject mean over runs
    accs = {}
    for subj in SUBJECTS:
        vals = []
        for run in range(N_RUNS):
            pkl = os.path.join(root, str(subj), f'run_{run}', f'tl_{subj}_results.pkl')
            if os.path.isfile(pkl):
                w = pickle.load(open(pkl,'rb'))
                vals.append(accuracy_score(w.ground_truth, w.predictions))
        if vals:
            accs[subj] = np.mean(vals)
    subj_cluster_acc[pid] = accs
# compute own vs other
own_acc = []
other_acc = []
subj_list = []
for subj in SUBJECTS:
    cid = cluster_map[subj]
    own = subj_cluster_acc.get(f'cluster{cid}', {}).get(subj, np.nan)
    # collect other clusters
    others = [subj_cluster_acc.get(pid, {}).get(subj, np.nan) for pid in cluster_pipelines if pid!=f'cluster{cid}']
    other = np.nanmean(others)
    subj_list.append(subj)
    own_acc.append(own)
    other_acc.append(other)

# Scatter matched vs mismatched
plt.figure()
plt.scatter(own_acc, other_acc)
mx = max(np.nanmax(own_acc), np.nanmax(other_acc))
plt.plot([0,mx],[0,mx],'--',color='gray')
plt.xlabel('Own‑Cluster Backbone Acc')
plt.ylabel('Other‑Clusters Mean Acc')
plt.title('Matched vs. Mismatched Cluster Backbones')
plt.savefig(os.path.join(OUTDIR,'matched_vs_mismatch_scatter.png'))

# Wilcoxon test for matched vs mismatched
stat, p = wilcoxon(own_acc, other_acc)
with open(os.path.join(OUTDIR,'matched_vs_mismatch.txt'),'w') as f:
    f.write(f"Wilcoxon matched vs mismatched: stat={stat:.3f}, p={p:.3e}")

print(f"Finished evaluation. Outputs in '{OUTDIR}'")