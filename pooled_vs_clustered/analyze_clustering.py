import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, kruskal

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

# ====== 6. HISTOGRAM OF ACCURACY DELTAS ======
plt.figure(figsize=(8,4))
sns.histplot(df_subject["accuracy_delta"], bins=15, kde=True)
plt.xlabel("Accuracy Delta (TL-Cluster - TL-Pooled)")
plt.ylabel("Number of Subjects")
plt.title("Per-Subject Accuracy Gain from Clustered Backbone")
plt.tight_layout()
plt.savefig("accuracy_delta_hist.png")
plt.close()

# ====== 7. DOT PLOT: POOLED vs. CLUSTERED ACCURACY (per subject) ======
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

# ====== 9. PER-CLUSTER ANALYSIS: ONLY IF MULTIPLE CLUSTERS PRESENT ======
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

    # Kruskal–Wallis
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
print("\n=== Sorted Per-Subject Results (top 10 shown) ===")
print(df_sorted[table_cols].head(10).to_string(index=False))

# ====== 11. OPTIONAL: METRICS DELTA HISTOGRAMS & STATS ======
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

# ====== 12. SAVE THE FINAL SORTED TABLE FOR REPORTING ======
df_sorted[table_cols].to_csv("tl_subject_level_comparison.csv", index=False)
print("\nSaved: tl_subject_level_comparison.csv")

# List top 10 gainers and losers
print("\nTop 10 most improved subjects:")
print(df_sorted.head(10)[["subject", "cluster_label", "accuracy_delta"]])
print("\nTop 10 least improved subjects:")
print(df_sorted.tail(10)[["subject", "cluster_label", "accuracy_delta"]])

# Pooled TL vs delta scatter
plt.figure(figsize=(6,4))
sns.scatterplot(x="accuracy_pooled", y="accuracy_delta", hue="cluster_label", data=df_subject, palette="tab10")
plt.axhline(0, color='k', ls='--')
plt.xlabel("Pooled TL Accuracy (Baseline)")
plt.ylabel("Accuracy Gain (Clustered - Pooled)")
plt.title("Does Baseline Predict Improvement from Clustering?")
plt.tight_layout()
plt.savefig("pooled_vs_delta.png")
plt.close()

# Correlation
from scipy.stats import pearsonr
r, p_corr = pearsonr(df_subject["accuracy_pooled"], df_subject["accuracy_delta"])
print(f"\nCorrelation (pooled vs delta): r = {r:.2f}, p = {p_corr:.4g}")

# Swarmplot by cluster
plt.figure(figsize=(7,4))
sns.swarmplot(x="cluster_label", y="accuracy_delta", data=df_subject)
plt.title("Accuracy Gain from Clustering by Cluster")
plt.tight_layout()
plt.savefig("swarm_accuracy_gain_by_cluster.png")
plt.close()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp, ttest_ind

# ——— assume df_subject already exists and contains at least:
#      'subject', 'accuracy_pooled', 'accuracy_cluster', 'accuracy_delta'

# 1) Define baseline cutoff and split into two groups
BASELINE_CUTOFF = 0.55
low_baseline  = df_subject[df_subject["accuracy_pooled"] < BASELINE_CUTOFF].copy()
high_baseline = df_subject[df_subject["accuracy_pooled"] >= BASELINE_CUTOFF].copy()

print(f"Low‐baseline (<{BASELINE_CUTOFF*100:.0f}%) count: {len(low_baseline)}")
print(f"High‐baseline (≥{BASELINE_CUTOFF*100:.0f}%) count: {len(high_baseline)}\n")

# 2) Test whether low‐baseline Δ is significantly > 0
t_low, p_low = ttest_1samp(low_baseline["accuracy_delta"], popmean=0, alternative='greater')
print("One‐sample t‐test (Δ>0) on low‐baseline group:")
print(f"  mean Δ_low = {low_baseline['accuracy_delta'].mean():.4f}")
print(f"  t = {t_low:.3f},  p (one‐tailed) = {p_low:.4g}\n")

# 3) Compare Δ(low‐baseline) vs. Δ(high‐baseline) with Welch’s t‐test
t_cmp, p_cmp = ttest_ind(
    low_baseline["accuracy_delta"],
    high_baseline["accuracy_delta"],
    equal_var=False,
    alternative='greater'  # test if low‐baseline Δ > high‐baseline Δ
)
print("Welch’s t‐test (Δ_low vs. Δ_high):")
print(f"  mean Δ_low  = {low_baseline['accuracy_delta'].mean():.4f}")
print(f"  mean Δ_high = {high_baseline['accuracy_delta'].mean():.4f}")
print(f"  t = {t_cmp:.3f},  p (one‐tailed) = {p_cmp:.4g}\n")

# 4) Boxplot of Δ by baseline‐group
df_plot = df_subject.copy()
df_plot["baseline_group"] = df_plot["accuracy_pooled"].apply(
    lambda x: "Low (<55%)" if x < BASELINE_CUTOFF else "High (≥55%)"
)

plt.figure(figsize=(6,4))
sns.boxplot(
    x="baseline_group",
    y="accuracy_delta",
    data=df_plot,
    palette=["#FFA07A", "#87CEFA"],  # salmon for low, skyblue for high
    showfliers=False
)
sns.swarmplot(
    x="baseline_group",
    y="accuracy_delta",
    data=df_plot,
    color="k",
    size=4,
    alpha=0.6
)

plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.xlabel("Baseline Group")
plt.ylabel("Accuracy Δ (Clustered – Pooled)")
plt.title("Clustering Gain Δ by Baseline Group")
plt.tight_layout()
plt.savefig("delta_by_baseline_group.png", dpi=150)
plt.close()
