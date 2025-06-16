import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from scipy.stats import pearsonr, entropy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ====== PARAMETERS ======
BASELINE_CUTOFF = 0.70
TL_CLUSTER_CSV = "results/fbcsp_deeparch_ncluster4_full_tl_subject_run_metrics.csv"
TL_POOLED_CSV  = "results/fbcsp_deeparch_ncluster1_full_tl_subject_run_metrics.csv"
CLUSTER_LABELS = "results/subject_cluster.csv"
PREPROC_PKL     = "dump/preprocessed_data_custom.pkl"

# ====== 1. LOAD TL RESULTS & MERGE ======
df_cluster = pd.read_csv(TL_CLUSTER_CSV)
df_pooled  = pd.read_csv(TL_POOLED_CSV)
df_clusters= pd.read_csv(CLUSTER_LABELS)[["subject", "cluster"]].drop_duplicates()

metrics = ["accuracy", "kappa", "precision", "recall", "f1_score"]

df = df_cluster.merge(df_pooled, on=["run","subject"], suffixes=("_cluster","_pooled"))
for m in metrics:
    df[f"{m}_delta"] = df[f"{m}_cluster"] - df[f"{m}_pooled"]

agg = {}
for m in metrics:
    agg[f"{m}_cluster"] = "mean"
    agg[f"{m}_pooled"]  = "mean"
    agg[f"{m}_delta"]   = "mean"
df_subject = df.groupby("subject").agg(agg).reset_index()
df_subject = df_subject.merge(df_clusters, on="subject").rename(columns={"cluster":"cluster_label"})

low_baseline = df_subject[df_subject["accuracy_pooled"] < BASELINE_CUTOFF]
print(f"Subjects with baseline < {BASELINE_CUTOFF*100:.0f}%:\n", 
      low_baseline[["subject","cluster_label","accuracy_pooled","accuracy_delta"]])

# ====== 2. LOAD PREPROCESSED EPOCHS ======
with open(PREPROC_PKL, "rb") as f:
    preproc_dict = pickle.load(f)

# ====== 3. COMPUTE FEATURE METRICS ======
features = {"mu_power": [], "mu_var": [], "erd": [], "spec_entropy": []}
subjects_with_features = []

for subj in df_subject["subject"].unique():
    sid = str(subj)
    if sid not in preproc_dict:
        print(f"Missing preprocessed epochs for subject {subj}, skipping.")
        continue

    entry = preproc_dict[sid]
    if not isinstance(entry, dict) or "train" not in entry:
        print(f"Subject {subj} entry format unexpected, skipping.")
        continue

    epochs = entry["train"]  # EpochsArray
    data = epochs.get_data()
    n_epochs, n_channels, n_times = data.shape

    # μ‐band per trial
    psd_mu = epochs.compute_psd(fmin=8.0, fmax=12.0, picks="eeg",
                                method="welch", n_per_seg=n_times, verbose=False).get_data()
    trial_mu = psd_mu.mean(axis=(1,2))
    mu_mean = trial_mu.mean()
    mu_var  = trial_mu.var()

    # ERD
    times = epochs.times
    baseline_mask = (times >= 0.0) & (times < 0.5)
    mi_mask       = (times >= 0.5) & (times < 2.0)
    baseline_power = (data[..., baseline_mask]**2).mean(axis=(1,2))
    mi_power       = (data[..., mi_mask]**2).mean(axis=(1,2))
    erd_trials     = (baseline_power - mi_power) / baseline_power
    erd_mean       = erd_trials.mean()

    # Spectral entropy 1–40 Hz
    psd_all = epochs.compute_psd(fmin=1.0, fmax=40.0, picks="eeg",
                                 method="welch", n_per_seg=n_times, verbose=False).get_data()
    psd_avg  = psd_all.mean(axis=(0,1))
    psd_norm = psd_avg / psd_avg.sum()
    spec_ent = entropy(psd_norm)

    subjects_with_features.append(subj)
    features["mu_power"].append(mu_mean)
    features["mu_var"].append(mu_var)
    features["erd"].append(erd_mean)
    features["spec_entropy"].append(spec_ent)

df_feats = pd.DataFrame({
    "subject": subjects_with_features,
    "mu_power": features["mu_power"],
    "mu_var": features["mu_var"],
    "erd": features["erd"],
    "spec_entropy": features["spec_entropy"],
})
df_subject = df_subject.merge(df_feats, on="subject", how="left")

# ====== 4. CORRELATIONS & PLOTS ======
for feat in ["mu_power", "mu_var", "erd", "spec_entropy"]:
    mask = ~df_subject[feat].isna()
    if mask.sum() < 2:
        print(f"Not enough data for {feat}, skipping.")
        continue

    r_base, p_base   = pearsonr(df_subject.loc[mask, feat], df_subject.loc[mask, "accuracy_pooled"])
    r_delta, p_delta = pearsonr(df_subject.loc[mask, feat], df_subject.loc[mask, "accuracy_delta"])
    print(f"{feat} vs. baseline: r = {r_base:.2f}, p = {p_base:.4g}")
    print(f"{feat} vs. Δ:        r = {r_delta:.2f}, p = {p_delta:.4g}")

    """
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=feat, y="accuracy_pooled", data=df_subject.loc[mask],
                    s=50, edgecolor="k", alpha=0.7)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel(feat)
    plt.ylabel("Baseline Accuracy")
    plt.title(f"{feat} vs. Baseline Accuracy")
    plt.tight_layout()
    plt.savefig(f"{feat}_vs_baseline.png")
    plt.close()
    """
    # … inside the for‐loop over feats, when feat == "erd": …
    plt.figure(figsize=(6,4))
    # 1) Original scatter
    sns.scatterplot(
        x="erd",
        y="accuracy_pooled",
        data=df_subject.loc[mask],
        s=50,
        edgecolor="k",
        alpha=0.7
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)

    # 2) Add a red regression line on top
    sns.regplot(
        x="erd",
        y="accuracy_pooled",
        data=df_subject.loc[mask],
        scatter=False,
        line_kws={"color": "red", "lw": 2}
    )

    plt.xlabel("ERD (μ‐band % drop)")
    plt.ylabel("Baseline Accuracy (Pooled TL)")
    plt.title("ERD vs. Baseline Accuracy (with Regression Line)")
    plt.tight_layout()
    plt.savefig("erd_vs_baseline_with_reg.png")
    plt.close()

    # Now add ERD vs. Δ **with** regression line:
    plt.figure(figsize=(6,4))
    sns.scatterplot(
        x="erd",
        y="accuracy_delta",
        data=df_subject.loc[mask],
        s=50,
        edgecolor="k",
        alpha=0.7
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    sns.regplot(
        x="erd",
        y="accuracy_delta",
        data=df_subject.loc[mask],
        scatter=False,
        line_kws={"color": "red", "lw": 2}
    )
    plt.xlabel("ERD (μ‐band % drop)")
    plt.ylabel("Accuracy Δ (Clustered–Pooled)")
    plt.title("ERD vs. Accuracy Δ (with Regression Line)")
    plt.tight_layout()
    plt.savefig("erd_vs_delta_with_reg.png")
    plt.close()

else:
    # your existing code for other feats (mu_power, mu_var, spec_entropy) …
    plt.figure(figsize=(6,4))
    sns.scatterplot(
        x=feat,
        y="accuracy_pooled",
        data=df_subject.loc[mask],
        s=50,
        edgecolor="k",
        alpha=0.7
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel(feat)
    plt.ylabel("Baseline Accuracy")
    plt.title(f"{feat} vs. Baseline Accuracy")
    plt.tight_layout()
    plt.savefig(f"{feat}_vs_baseline.png")
    plt.close()

    plt.figure(figsize=(6,4))
    sns.scatterplot(
        x=feat,
        y="accuracy_delta",
        data=df_subject.loc[mask],
        s=50,
        edgecolor="k",
        alpha=0.7
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel(feat)
    plt.ylabel("Accuracy Δ (Clustered–Pooled)")
    plt.title(f"{feat} vs. Accuracy Gain (Clustered–Pooled)")
    plt.tight_layout()
    plt.savefig(f"{feat}_vs_delta.png")
    plt.close()
    
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=feat, y="accuracy_delta", data=df_subject.loc[mask],
                    s=50, edgecolor="k", alpha=0.7)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel(feat)
    plt.ylabel("Accuracy Δ")
    plt.title(f"{feat} vs. Accuracy Gain (Clustered–Pooled)")
    plt.tight_layout()
    plt.savefig(f"{feat}_vs_delta.png")
    plt.close()

# ====== 5. PCA WITH GUARD ======
feat_cols = ["accuracy_pooled", "accuracy_delta", "mu_power", "mu_var", "erd", "spec_entropy"]
df_pca = df_subject.dropna(subset=feat_cols).copy()

if df_pca.shape[0] < 2:
    print("Not enough subjects with all features for PCA—skipping PCA step.")
else:
    X = df_pca[feat_cols].values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    plt.figure(figsize=(6,6))
    scatter = sns.scatterplot(
        x=pcs[:, 0],
        y=pcs[:, 1],
        hue=df_pca["accuracy_delta"],
        palette="coolwarm",
        s=60,
        edgecolor="k"
    )
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("PCA of Subject Features (by Δ)")

    # Grab the mappable from the scatter plot and use it for the colorbar
    mappable = scatter.collections[0]
    plt.colorbar(mappable, label="Accuracy Δ")

    plt.tight_layout()
    plt.savefig("subject_features_pca_delta.png")
    plt.close()


# ====== 6. SAVE ======
df_subject.to_csv("tl_subject_with_all_features.csv", index=False)
print("\nSaved: tl_subject_with_all_features.csv")
