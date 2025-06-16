import pickle
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import mne
from scipy.stats import pearsonr


# ====== 1. LOAD DATA ======
df_cluster = pd.read_csv("results/fbcsp_deeparch_ncluster4_full_tl_subject_run_metrics.csv")  # CLUSTERED TL - run, subject, accuracy, kappa, ...
df_pooled  = pd.read_csv("results/fbcsp_deeparch_ncluster1_full_tl_subject_run_metrics.csv")   # NO CLUSTER TL - run, subject, accuracy, kappa, ...
df_mtl_stats = pd.read_csv("results/subject_cluster.csv")  # subject, cluster, ...

metrics = ["accuracy", "kappa", "precision", "recall", "f1_score"]

# ====== 2. MERGE TL RESULTS ======
df = df_cluster.merge(
    df_pooled,
    on=["run", "subject"],
    suffixes=("_cluster", "_pooled")
)

# ====== 3. COMPUTE DELTAS PER RUN ======
for m in metrics:
    df[f"{m}_delta"] = df[f"{m}_cluster"] - df[f"{m}_pooled"]

# ====== 4. AGGREGATE OVER RUNS PER SUBJECT ======
agg_dict = {}
for m in metrics:
    agg_dict[f"{m}_cluster"] = "mean"
    agg_dict[f"{m}_pooled"] = "mean"
    agg_dict[f"{m}_delta"] = "mean"
df_subject = df.groupby("subject").agg(agg_dict).reset_index()

# ====== 5. JOIN ORIGINAL CLUSTER LABEL FROM MTL STATS ======
df_clusters = df_mtl_stats[["subject", "cluster"]].drop_duplicates()
df_subject = df_subject.merge(df_clusters, on="subject")
df_subject = df_subject.rename(columns={"cluster": "cluster_label"})

# ====== 7. LOAD THE SINGLE PKL WITH ALL PREPROCESSED EPOCHS ======
pkl_path = "dump/preprocessed_data_custom.pkl"
with open(pkl_path, "rb") as f:
    preproc_dict = pickle.load(f)

# Just to confirm structure:
# print(type(preproc_dict))       # should be <class 'dict'>
# print(preproc_dict.keys())      # strings like '300', '301', etc.

# ====== 8. COMPUTE μ-BANDPOWER (8–12 Hz) FROM PREPROCESSED EPOCHS ======
mu_powers = {}  # subject (int) → mean μ power

for subj in df_subject["subject"].unique():
    subj_str = str(subj)
    if subj_str not in preproc_dict:
        print(f"WARNING: Subject {subj} not found; skipping.")
        continue

    entry = preproc_dict[subj_str]
    if isinstance(entry, dict) and "train" in entry:
        epochs = entry["train"]   # preprocessed EpochsArray
    else:
        print(f"WARNING: Unexpected format for subject {subj}; skipping.")
        continue

    # Get the number of time points in each epoch
    n_times = epochs.get_data().shape[-1]

    psd = epochs.compute_psd(
        fmin=8.0,
        fmax=12.0,
        picks="eeg",
        method="welch",
        n_per_seg=n_times,   # use the full epoch length for the segment
        verbose=False
    ).get_data()

    mu_power = psd.mean()  # average across epochs, channels, and frequencies
    mu_powers[subj] = mu_power

# Merge μ-power back into df_subject
df_mu = pd.DataFrame({
    "subject": list(mu_powers.keys()),
    "mu_power": list(mu_powers.values())
})
df_subject = df_subject.merge(df_mu, on="subject", how="left")

# ====== 9. CORRELATE μ-POWER with Baseline/Δ ======
mask = ~df_subject["mu_power"].isna()
r_mu_base, p_mu_base = pearsonr(df_subject.loc[mask, "mu_power"], df_subject.loc[mask, "accuracy_pooled"])
r_mu_delta, p_mu_delta = pearsonr(df_subject.loc[mask, "mu_power"], df_subject.loc[mask, "accuracy_delta"])

print(f"\nCorrelation (μ-power vs baseline): r = {r_mu_base:.2f}, p = {p_mu_base:.4g}")
print(f"Correlation (μ-power vs Δ):        r = {r_mu_delta:.2f}, p = {p_mu_delta:.4g}")

# ====== 10. PLOT UPDATED μ-POWER DISTRIBUTION ======
plt.figure(figsize=(6,4))
sns.histplot(df_subject["mu_power"].dropna(), bins=15, kde=True)
plt.xlabel("Mean μ-Band Power (8–12 Hz)")
plt.ylabel("Number of Subjects")
plt.title("Distribution of μ-Power (Preprocessed Epochs)")
plt.tight_layout()
plt.savefig("mu_power_hist_preproc.png")
plt.close()

plt.figure(figsize=(6,4))
sns.scatterplot(
    x="mu_power", 
    y="accuracy_pooled", 
    data=df_subject.loc[mask], 
    s=50, 
    edgecolor="k", 
    alpha=0.7
)
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.xlabel("Mean μ-Band Power (8–12 Hz)")
plt.ylabel("Baseline Accuracy (Pooled TL)")
plt.title("μ-Power vs. Baseline Accuracy (Preprocessed)")
plt.tight_layout()
plt.savefig("mu_power_vs_baseline_preproc.png")
plt.close()

plt.figure(figsize=(6,4))
sns.scatterplot(
    x="mu_power", 
    y="accuracy_delta", 
    data=df_subject.loc[mask],
    s=50, 
    edgecolor="k", 
    alpha=0.7
)
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.xlabel("Mean μ-Band Power (8–12 Hz)")
plt.ylabel("Accuracy Gain (Clustered – Pooled)")
plt.title("μ-Power vs. Accuracy Δ (Preprocessed)")
plt.tight_layout()
plt.savefig("mu_power_vs_delta_preproc.png")
plt.close()

# ====== 11. SAVE UPDATED TABLE ======
df_subject.to_csv("tl_subject_level_with_mu_preproc.csv", index=False)
print("\nSaved: tl_subject_level_with_mu_preproc.csv")
