import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from scipy.stats import pearsonr, entropy, ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ====== PARAMETERS ======
BASELINE_CUTOFF = 0.70
ERD_THRESHOLD   = 0.20   # cutoff between “low‐ERD” and “high‐ERD”
INPUT_CSV       = "tl_subject_with_all_features.csv"  # must contain: subject, cluster_label,
                                                      # accuracy_pooled, accuracy_delta, erd, etc.
PREPROC_PKL     = "dump/preprocessed_data_custom.pkl"

# ====== 1. LOAD THE MERGED SUBJECT‐LEVEL CSV ======
df_subject = pd.read_csv(INPUT_CSV)

# Drop rows missing any of the key columns
df_subject = df_subject.dropna(subset=["accuracy_pooled", "accuracy_delta", "erd", "cluster_label"])

# ====== 2. STRATIFY LOW‐ERD VS. HIGH‐ERD ======
low_erd  = df_subject[df_subject["erd"] < ERD_THRESHOLD]
high_erd = df_subject[df_subject["erd"] >= ERD_THRESHOLD]

print(f"\nSubjects with ERD < {ERD_THRESHOLD:.2f} (low‐ERD group):")
print(low_erd[["subject", "cluster_label", "accuracy_pooled", "accuracy_delta", "erd"]].to_string(index=False))

print(f"\nSubjects with ERD ≥ {ERD_THRESHOLD:.2f} (high‐ERD group):")
print(high_erd[["subject", "cluster_label", "accuracy_pooled", "accuracy_delta", "erd"]].to_string(index=False))

# ====== 2a. TWO‐SAMPLE TEST ON Δ BETWEEN LOW‐ERD AND HIGH‐ERD ======
low_vals  = low_erd["accuracy_delta"].values
high_vals = high_erd["accuracy_delta"].values

# Welch’s t‐test (unequal variances)
t_stat, p_val = ttest_ind(low_vals, high_vals, equal_var=False)
print(f"\nWelch’s t‐test comparing Δ (low‐ERD vs. high‐ERD):")
print(f"  t = {t_stat:.3f},  p = {p_val:.4g}")

# Interpretive printout:
if p_val < 0.05:
    print("  → p < 0.05: The difference in Δ between low‐ERD and high‐ERD is statistically significant.")
else:
    print("  → p ≥ 0.05: The difference in Δ between low‐ERD and high‐ERD is not statistically significant.")

# ====== 3. PLOT HISTOGRAM OF Δ FOR BOTH GROUPS ======
plt.figure(figsize=(6,4))
sns.histplot(low_vals,  bins=10, color="orange", label="ERD < 0.20", kde=False, alpha=0.7)
sns.histplot(high_vals, bins=10, color="steelblue", label="ERD ≥ 0.20", kde=False, alpha=0.7)
plt.xlabel("Accuracy Δ (Clustered – Pooled)")
plt.ylabel("Number of Subjects")
plt.title("Δ Distribution: Low‐ERD vs. High‐ERD")
plt.legend()
plt.tight_layout()
plt.savefig("delta_hist_lowvshigh_erd.png")
plt.close()

# Print summary statistics side‐by‐side
print("\nΔ summary for low‐ERD group:")
print(low_erd["accuracy_delta"].describe().to_string())
print("\nΔ summary for high‐ERD group:")
print(high_erd["accuracy_delta"].describe().to_string())

# ====== 4. VISUALIZE ERD DISTRIBUTIONS BY CLUSTER ======
plt.figure(figsize=(7,4))
sns.boxplot(x="cluster_label", y="erd", data=df_subject, palette="viridis")
plt.xlabel("Cluster Label")
plt.ylabel("ERD (μ‐band % drop)")
plt.title("ERD by Original K‐Means Cluster")
plt.tight_layout()
plt.savefig("erd_by_cluster.png")
plt.close()

# ====== 5. FIT A LINEAR MODEL: Δ ~ baseline + ERD ======
# Prepare design matrix and target
X = df_subject[["accuracy_pooled", "erd"]].values
y = df_subject["accuracy_delta"].values.reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print("\nLinear model Δ = β₀ + β₁·(baseline) + β₂·(ERD) + ε")
print(f"  β₀ (intercept)     = {model.intercept_[0]:.4f}")
print(f"  β₁ (baseline coef) = {model.coef_[0][0]:.4f}")
print(f"  β₂ (ERD coef)      = {model.coef_[0][1]:.4f}")
print(f"  R² of model        = {r2:.3f}")

# Scatter Δ vs. baseline, colored by predicted Δ, with colorbar
plt.figure(figsize=(6,4))
scatter = sns.scatterplot(
    x=df_subject["accuracy_pooled"],
    y=df_subject["accuracy_delta"],
    hue=y_pred.flatten(),
    palette="coolwarm",
    s=60,
    edgecolor="k"
)
plt.xlabel("Baseline Accuracy")
plt.ylabel("Accuracy Δ")
plt.title("Δ vs. Baseline (colored by Predicted Δ)")
mappable = scatter.collections[0]
plt.colorbar(mappable, label="Predicted Δ")
plt.tight_layout()
plt.savefig("delta_vs_baseline_with_predicted.png")
plt.close()

# ====== 6. ALTERNATIVE ERD WINDOW (Optional) ======
# Uncomment the block below to recompute ERD over 0.5–1.5 s instead of 0.5–2 s.

# with open(PREPROC_PKL, "rb") as f:
#     preproc = pickle.load(f)
#
# alt_erds = {}
# for subj in df_subject["subject"].astype(str):
#     entry = preproc.get(subj, None)
#     if entry is None or "train" not in entry:
#         continue
#     epochs = entry["train"]
#     data = epochs.get_data()
#     times = epochs.times
#     baseline_mask_alt = (times >= 0.0) & (times < 0.5)
#     mi_mask_alt       = (times >= 0.5) & (times < 1.5)
#     if not (baseline_mask_alt.any() and mi_mask_alt.any()):
#         continue
#     bpow = (data[..., baseline_mask_alt]**2).mean(axis=(1,2))
#     mpow = (data[..., mi_mask_alt]**2).mean(axis=(1,2))
#     erd_trials_alt = (bpow - mpow) / bpow
#     alt_erds[int(subj)] = erd_trials_alt.mean()
#
# df_alt = df_subject.copy()
# df_alt["erd_alt"] = df_alt["subject"].map(alt_erds)
# df_alt = df_alt.dropna(subset=["erd_alt"])
# r_alt, p_alt = pearsonr(df_alt["erd_alt"], df_alt["accuracy_pooled"])
# print(f"\nAlternative ERD (0.5–1.5 s) vs. baseline: r = {r_alt:.2f}, p = {p_alt:.4g}")

print("\nAll analyses completed. Plots and statistics are saved in the working directory.")  

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Load the CSV you already produced that contains 'erd' and 'accuracy_delta'
df = pd.read_csv("tl_subject_with_all_features.csv")

# 2) Create a new column categorizing each subject as “Low ERD” or “High ERD”
df["erd_group"] = df["erd"].apply(lambda x: "Low ERD (< 0.20)" if x < 0.20 else "High ERD (≥ 0.20)")

# 3) Draw a boxplot of clustering gain (accuracy_delta) by ERD group
plt.figure(figsize=(6,4))
sns.boxplot(
    x="erd_group",
    y="accuracy_delta",
    data=df,
    palette=["#FFA500", "#4682B4"]  # orange for low, steelblue for high
)
plt.xlabel("ERD Group")
plt.ylabel("Accuracy Δ (Clustered – Pooled)")
plt.title("Clustering Gain Δ by ERD Group")
plt.tight_layout()
plt.savefig("delta_by_erd_group.png", dpi=150)
plt.close()
