import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from scipy.stats import ttest_rel, kruskal, pearsonr

# ====== PARAMETERS ======
BASELINE_CUTOFF = 0.70
RAW_FIF_DIR = "/home/ubuntu/VT2/data/04_sampleFreq200_80_events/"  # directory containing one <subject>.fif per subject

# ====== 1. LOAD DATA ======
df_cluster = pd.read_csv("results/fbcsp_deeparch_ncluster4_full_tl_subject_run_metrics.csv")
df_pooled  = pd.read_csv("results/fbcsp_deeparch_ncluster1_full_tl_subject_run_metrics.csv")
df_clusters= pd.read_csv("results/subject_cluster.csv")  # columns: subject, cluster

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

# ====== 5. JOIN ORIGINAL CLUSTER LABEL ======
df_clusters = df_clusters[["subject", "cluster"]].drop_duplicates()
df_subject = df_subject.merge(df_clusters, on="subject")
df_subject = df_subject.rename(columns={"cluster": "cluster_label"})

# ====== 6. HISTOGRAM OF ACCURACY DELTAS ======
plt.figure(figsize=(8,4))
sns.histplot(df_subject["accuracy_delta"], bins=15, kde=True)
plt.xlabel("Accuracy Delta (TL-Cluster - TL-Pooled)")
plt.ylabel("Number of Subjects")
plt.title("Per-Subject Accuracy Gain from Clustered Backbone")
plt.tight_layout()
plt.savefig("accuracy_delta_hist.png")
plt.close()

# ====== 7. DOT PLOT: POOLED vs. CLUSTERED ACCURACY ======
plt.figure(figsize=(6,6))
if df_subject["cluster_label"].nunique() > 1:
    scatter = plt.scatter(
        df_subject["accuracy_pooled"],
        df_subject["accuracy_cluster"],
        c=df_subject["cluster_label"],
        cmap="tab10",
        edgecolor="k",
        s=60
    )
else:
    plt.scatter(
        df_subject["accuracy_pooled"],
        df_subject["accuracy_cluster"],
        color="b",
        edgecolor="k",
        s=60
    )
lims = [
    min(df_subject[["accuracy_pooled", "accuracy_cluster"]].min()),
    max(df_subject[["accuracy_pooled", "accuracy_cluster"]].max()),
]
plt.plot(lims, lims, 'k--', lw=2)
plt.xlabel("TL Pooled Backbone Accuracy")
plt.ylabel("TL Clustered Backbone Accuracy")
plt.title("Subject-wise: Pooled vs. Clustered Backbone")
plt.tight_layout()
if df_subject["cluster_label"].nunique() > 1:
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster Label")
plt.savefig("pooled_vs_clustered_dotplot.png")
plt.close()

# ====== 8. PAIRWISE T-TEST (accuracy) ======
t_stat, p_val = ttest_rel(df_subject["accuracy_cluster"], df_subject["accuracy_pooled"])
print(f"Paired t-test (accuracy): t = {t_stat:.2f}, p = {p_val:.4g}")

# ====== 9. PER-CLUSTER ANALYSIS (ONLY IF MULTIPLE CLUSTERS) ======
if df_subject["cluster_label"].nunique() > 1:
    plt.figure(figsize=(6,4))
    sns.barplot(
        x="cluster_label",
        y="accuracy_delta",
        data=df_subject,
        estimator="mean",
        errorbar="sd",
        palette="viridis"
    )
    plt.xlabel("Original K-Means Cluster")
    plt.ylabel("Mean Accuracy Delta")
    plt.title("Mean Accuracy Delta by Cluster")
    plt.tight_layout()
    plt.savefig("accuracy_delta_barplot_by_cluster.png")
    plt.close()

    print("\nSubjects per cluster_label:")
    cluster_counts = df_subject.groupby("cluster_label")["subject"].count()
    print(cluster_counts)

    grouped = df_subject.groupby("cluster_label")
    groups = [group["accuracy_delta"].values for _, group in grouped if len(group) > 1]
    print(f"Number of clusters/groups with >1 subject: {len(groups)}")
    if len(groups) >= 2:
        h_stat, p_cluster = kruskal(*groups)
        print(f"Kruskal-Wallis (accuracy_delta by cluster): H = {h_stat:.2f}, p = {p_cluster:.4g}")
    else:
        print("Not enough clusters with >1 subject for Kruskal-Wallis test.")
else:
    print("Only one cluster in data; skipping cluster-based analysis and tests.")

# ====== 10. SORTED TABLE: PER-SUBJECT RESULTS ======
table_cols = [
    "subject", "cluster_label",
    "accuracy_pooled", "accuracy_cluster", "accuracy_delta",
    "kappa_pooled", "kappa_cluster", "kappa_delta",
    "precision_pooled", "precision_cluster", "precision_delta",
    "recall_pooled", "recall_cluster", "recall_delta",
    "f1_score_pooled", "f1_score_cluster", "f1_score_delta"
]
df_sorted = df_subject.sort_values("accuracy_delta", ascending=False)
print("\n=== Top 10 Most Improved Subjects ===")
print(df_sorted[table_cols].head(10).to_string(index=False))
print("\n=== Top 10 Least Improved Subjects ===")
print(df_sorted[table_cols].tail(10).to_string(index=False))

# ====== 11. IDENTIFY LOW-BASELINE SUBJECTS ======
low_baseline = df_subject[df_subject["accuracy_pooled"] < BASELINE_CUTOFF]
print(f"\nSubjects with baseline < {BASELINE_CUTOFF*100:.0f}%:")
print(low_baseline[["subject", "cluster_label", "accuracy_pooled", "accuracy_delta"]])
print("\nCluster counts among low-baseline group:")
print(low_baseline["cluster_label"].value_counts())

# ====== 12. COMPUTE μ-BANDPOWER FROM RAW .fif ======
# For each subject, load <subject>.fif, compute average 8–12 Hz power across all EEG channels
mu_powers = {}  # subject → mean μ power
for subj in df_subject["subject"].unique():
    # Replace the old fif_path line with this:
    fif_path = os.path.join(RAW_FIF_DIR, f"recording_subject_{subj}_session_1_raw.fif")
    if not os.path.isfile(fif_path):
        print(f"WARNING: Missing {fif_path}, skipping μ-power for subject {subj}")
        continue

    # Load raw data (preload for faster ops)
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
    picks = mne.pick_types(raw.info, meg=False, eeg=True)  # all EEG channels

    # Compute PSD via Welch for 8–12 Hz band
        # Compute μ-band PSD (8–12 Hz) using Raw.compute_psd
    psds = raw.compute_psd(
        fmin=8.0, fmax=12.0,
        picks=picks,
        n_fft=2048,
        verbose=False
    ).get_data()  # shape: (n_channels, n_freq_bins)
    mu_power = psds.mean()
    mu_powers[subj] = mu_power

    # psds shape: (n_channels, n_freqs); freqs are frequency bins in [8,12] 
    mu_power = psds.mean()  # average over channels and frequency bins
    mu_powers[subj] = mu_power

# Convert to DataFrame and merge
df_mu = pd.DataFrame({
    "subject": list(mu_powers.keys()),
    "mu_power": list(mu_powers.values())
})
df_subject = df_subject.merge(df_mu, on="subject", how="left")

# ====== 13. CORRELATIONS WITH μ-POWER ======
# a) μ-power vs. baseline accuracy
mask = ~df_subject["mu_power"].isna()
r_mu_base, p_mu_base = pearsonr(df_subject.loc[mask, "mu_power"], df_subject.loc[mask, "accuracy_pooled"])
print(f"\nCorrelation (μ-power vs baseline accuracy): r = {r_mu_base:.2f}, p = {p_mu_base:.4g}")

# b) μ-power vs. clustering gain (accuracy_delta)
r_mu_delta, p_mu_delta = pearsonr(df_subject.loc[mask, "mu_power"], df_subject.loc[mask, "accuracy_delta"])
print(f"Correlation (μ-power vs accuracy_delta): r = {r_mu_delta:.2f}, p = {p_mu_delta:.4g}")

plt.figure(figsize=(6,4))
sns.histplot(df_subject["mu_power"], bins=15, kde=True)
plt.xlabel("Mean μ-Band Power (8–12 Hz)")
plt.ylabel("Number of Subjects")
plt.title("Distribution of μ-Power Across Subjects")
plt.tight_layout()
plt.savefig("mu_power_hist.png")
plt.close()

# Scatterplots
plt.figure(figsize=(6,4))
sns.scatterplot(x="mu_power", y="accuracy_pooled", data=df_subject.loc[mask])
plt.xlabel("μ-band Power (8–12 Hz)")
plt.ylabel("Pooled TL Accuracy (Baseline)")
plt.title("μ-power vs. Baseline Accuracy")
plt.tight_layout()
plt.savefig("mu_power_vs_baseline.png")
plt.close()

plt.figure(figsize=(6,4))
sns.scatterplot(x="mu_power", y="accuracy_delta", data=df_subject.loc[mask])
plt.xlabel("μ-band Power (8–12 Hz)")
plt.ylabel("Accuracy Gain from Clustering")
plt.title("μ-power vs. Accuracy Delta")
plt.tight_layout()
plt.savefig("mu_power_vs_delta.png")
plt.close()

# ====== 14. OPTIONAL: METRICS DELTA HISTOGRAMS & STATS ======
for m in ["kappa", "precision", "recall", "f1_score"]:
    plt.figure(figsize=(8,4))
    sns.histplot(df_subject[f"{m}_delta"], bins=15, kde=True)
    plt.xlabel(f"{m.title()} Delta (TL-Cluster - TL-Pooled)")
    plt.ylabel("Number of Subjects")
    plt.title(f"Per-Subject {m.title()} Gain from Clustered Backbone")
    plt.tight_layout()
    plt.savefig(f"{m}_delta_hist.png")
    plt.close()

    # T-test
    t_stat, p_val = ttest_rel(df_subject[f"{m}_cluster"], df_subject[f"{m}_pooled"])
    print(f"Paired t-test ({m}): t = {t_stat:.2f}, p = {p_val:.4g}")

    if df_subject["cluster_label"].nunique() > 1:
        groups = [group[f"{m}_delta"].values for _, group in df_subject.groupby("cluster_label") if len(group) > 1]
        if len(groups) >= 2:
            h_stat, p_cluster = kruskal(*groups)
            print(f"Kruskal-Wallis ({m}_delta by cluster): H = {h_stat:.2f}, p = {p_cluster:.4g}")
        else:
            print(f"Not enough clusters with >1 subject for Kruskal-Wallis test ({m}).")
    else:
        print(f"Only one cluster in data; skipping cluster-based {m} analysis.")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the per-subject comparison CSV you already saved:
# (Either tl_subject_level_comparison_with_mu.csv or tl_subject_level_comparison.csv,
#  as long as it contains 'accuracy_pooled' and 'accuracy_delta'.)
df = pd.read_csv("tl_subject_level_comparison_with_mu.csv")


# BASELINE VS GAIN REGRESSION 
plt.figure(figsize=(8,6))

# 1) Scatterplot, colored by cluster_label
sns.scatterplot(
    x="accuracy_pooled",
    y="accuracy_delta",
    hue="cluster_label",
    palette="tab10",       # or any categorical palette
    data=df,
    s=60,
    edgecolor="k",
    alpha=0.7
)

# 2) Overlay a single regression line (ignoring cluster)
sns.regplot(
    x="accuracy_pooled",
    y="accuracy_delta",
    data=df,
    scatter=False,         # do not draw points again
    line_kws={"color": "red", "linewidth": 2}
)

# 3) Add horizontal zero line
plt.axhline(0, color="gray", linestyle="--", linewidth=1)

plt.xlabel("Baseline Accuracy (Pooled TL)")
plt.ylabel("Accuracy Gain (Clustered – Pooled)")
plt.title("Baseline vs. Gain: Colored by Cluster")

# Adjust legend title
plt.legend(title="Cluster Label", loc="best")

plt.tight_layout()
plt.savefig("baseline_vs_gain_colored_by_cluster.png")
plt.close()

# ====== 15. SAVE FINAL TABLE ======
df_sorted = df_subject.sort_values("accuracy_delta", ascending=False)
df_sorted[table_cols + ["mu_power"]].to_csv("tl_subject_level_comparison_with_mu.csv", index=False)
print("\nSaved: tl_subject_level_comparison_with_mu.csv")
