
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, kruskal
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import yaml
from lib.pipeline.cluster.cluster import SubjectClusterer


PIPELINE_ROOTS = {
    'scratch': 'models/tl/scratch',
    'global':  'models/tl/global',
    'cluster0': 'models/tl/cluster0',
    'cluster1': 'models/tl/cluster1',
    'cluster2': 'models/tl/cluster2',
    'cluster3': 'models/tl/cluster3',
}

SUBJECTS = [1,2,3,4,5,6,7,8,9]
N_RUNS    = 3
OUTDIR    = 'evaluation'
os.makedirs(OUTDIR, exist_ok=True)

records = []
for pipeline, root in PIPELINE_ROOTS.items():
    for run in range(N_RUNS):
        run_dir = os.path.join(root, f'run_{run}')
        for subj in SUBJECTS:
            pkl_path = os.path.join(run_dir, f'tl_{subj}_results.pkl')
            if not os.path.isfile(pkl_path):
                continue
            wrapper = pickle.load(open(pkl_path, 'rb'))
            gt, pr = wrapper.ground_truth, wrapper.predictions
            records.append({
                'pipeline': pipeline,
                'subject': subj,
                'run': run,
                'accuracy': accuracy_score(gt, pr),
                'kappa':    cohen_kappa_score(gt, pr),
                'confusion_matrix': confusion_matrix(gt, pr)
            })

if not records:
    raise RuntimeError("No data found.")

df = pd.DataFrame(records)
df.to_csv(os.path.join(OUTDIR, 'summary_raw.csv'), index=False)


plt.figure()
df.boxplot(column='accuracy', by='pipeline')
plt.title('TL Accuracy by Pipeline'); plt.suptitle('')
plt.savefig(os.path.join(OUTDIR, 'accuracy_boxplot.png'))

plt.figure()
df.boxplot(column='kappa', by='pipeline')
plt.title('TL Cohen Kappa by Pipeline'); plt.suptitle('')
plt.savefig(os.path.join(OUTDIR, 'kappa_boxplot.png'))

# scratch vs global 
arr_s = df[df.pipeline=='scratch'].groupby('subject')['accuracy'].mean()
arr_g = df[df.pipeline=='global' ].groupby('subject')['accuracy'].mean()
stat, p = wilcoxon(arr_s, arr_g)
with open(os.path.join(OUTDIR,'wilcoxon_scratch_vs_global.txt'),'w') as f:
    f.write(f"scratch vs global: stat={stat:.3f}, p={p:.3e}\n")


cluster_pipelines = sorted([pid for pid in PIPELINE_ROOTS if pid.startswith('cluster')])
for pid in cluster_pipelines:
    # scatter global vs clusterX
    pivot = df.groupby(['pipeline','subject'])['accuracy'].mean().unstack(level=0)
    x = pivot['global']
    y = pivot[pid]
    plt.figure()
    plt.scatter(x, y)
    m = max(x.max(), y.max())
    plt.plot([0,m],[0,m],'--',color='gray')
    plt.xlabel('Global TL Acc'); plt.ylabel(f'{pid} TL Acc')
    plt.title(f'Global vs {pid} (mean over runs)')
    plt.savefig(os.path.join(OUTDIR, f'global_vs_{pid}_scatter.png'))

    # wilcoxon scratch vs clusterX
    arr_c = df[df.pipeline==pid].groupby('subject')['accuracy'].mean()
    stat_s_c, p_s_c = wilcoxon(arr_s, arr_c)
    # wilcoxon global vs clusterX
    stat_g_c, p_g_c = wilcoxon(arr_g, arr_c)

    with open(os.path.join(OUTDIR, f'wilcoxon_{pid}.txt'),'w') as f:
        f.write(f"scratch vs {pid}: stat={stat_s_c:.3f}, p={p_s_c:.3e}\n")
        f.write(f"global  vs {pid}: stat={stat_g_c:.3f}, p={p_g_c:.3e}\n")


for pipeline in PIPELINE_ROOTS:
    cms = [r['confusion_matrix'] for r in records if r['pipeline']==pipeline]
    if not cms: continue
    cms_arr  = np.stack(cms)
    cms_norm = np.array([cm.astype(float)/cm.sum(axis=1,keepdims=True) for cm in cms_arr])
    avg_cm   = cms_norm.mean(axis=0)

    plt.figure()
    im = plt.imshow(avg_cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Normalized Avg CM: {pipeline}')
    plt.colorbar(im)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.savefig(os.path.join(OUTDIR, f'cm_{pipeline}.png'))

# Performance by own cluster
clus_cfg      = yaml.safe_load(open('config/experiment/transfer.yaml'))['experiment']['clustering']
cluster_map   = SubjectClusterer('dump/features.pkl', clus_cfg).cluster_subjects(method=clus_cfg['method']).labels
df['cluster'] = df['subject'].map(cluster_map)

plt.figure()
df.boxplot(column='accuracy', by='cluster')
plt.title('TL Accuracy by Original Cluster'); plt.suptitle('')
plt.savefig(os.path.join(OUTDIR,'accuracy_by_cluster.png'))

groups = [g['accuracy'].values for _,g in df.groupby('cluster')]
stat_kw, p_kw = kruskal(*groups)
with open(os.path.join(OUTDIR,'kruskal_clusters.txt'),'w') as f:
    f.write(f"Kruskal-Wallis: stat={stat_kw:.3f}, p={p_kw:.3e}\n")

# Matched vs mismached cluster backbone
own_accs, other_accs = [], []
for subj in SUBJECTS:
    own_id = cluster_map[subj]
    own_pid = f'cluster{own_id}'
    own_mean = df[(df.pipeline==own_pid)&(df.subject==subj)]['accuracy'].mean()
    # gather all other cluster pipelines
    others = []
    for pid in cluster_pipelines:
        if pid==own_pid: continue
        others += df[(df.pipeline==pid)&(df.subject==subj)]['accuracy'].tolist()
    if np.isfinite(own_mean) and others:
        own_accs.append(own_mean)
        other_accs.append(np.mean(others))

plt.figure()
plt.scatter(own_accs, other_accs)
m = max(max(own_accs), max(other_accs))
plt.plot([0,m],[0,m],'--',color='gray')
plt.xlabel('Own-cluster Acc'); plt.ylabel('Other-clusters Mean Acc')
plt.title('Matched vs Mismatched Cluster Backbones')
plt.savefig(os.path.join(OUTDIR,'matched_vs_mismatch_scatter.png'))

stat_mm, p_mm = wilcoxon(own_accs, other_accs)
with open(os.path.join(OUTDIR,'wilcoxon_matched_vs_mismatched.txt'),'w') as f:
    f.write(f"matched vs mismatched: stat={stat_mm:.3f}, p={p_mm:.3e}\n")
    