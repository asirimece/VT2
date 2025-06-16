import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from scipy.stats import pearsonr, entropy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ====== PARAMETERS ======
BASELINE_CUTOFF = 0.70
ERD_THRESHOLD   = 0.20   # “low ERD” cutoff
INPUT_CSV       = "tl_subject_with_all_features.csv"  # must include columns: subject, cluster_label,
                                                      # accuracy_pooled, accuracy_delta, erd, etc.

# ====== 1. LOAD THE MERGED SUBJECT‐LEVEL CSV ======
df = pd.read_csv(INPUT_CSV)

# Drop any rows missing the key columns
df = df.dropna(subset=["accuracy_pooled", "accuracy_delta", "erd", "cluster_label"])

# ====== 2. STRATIFY LOW‐ERD SUBJECTS AND EXAMINE THEIR Δ ======
low_erd = df[df["erd"] < ERD_THRESHOLD]
print(f"\nSubjects with ERD < {ERD_THRESHOLD:.2f} (low‐ERD group):")
print(low_erd[["subject", "cluster_label", "accuracy_pooled", "accuracy_delta", "erd"]].to_string(index=False))

# Plot a histogram of Δ for low‐ERD group vs. all others
plt.figure(figsize=(6,4))
sns.histplot(low_erd["accuracy_delta"], bins=10, color="orange", label="ERD < 0.2", kde=False, alpha=0.7)
sns.histplot(df[df["erd"] >= ERD_THRESHOLD]["accuracy_delta"], bins=10, color="steelblue", label="ERD ≥ 0.2", kde=False, alpha=0.7)
plt.xlabel("Accuracy Δ (Clustered – Pooled)")
plt.ylabel("Number of Subjects")
plt.title("Δ Distribution: Low‐ERD vs. High‐ERD")
plt.legend()
plt.tight_layout()
plt.savefig("delta_hist_lowvshigh_erd.png")
plt.close()

# Print summary statistics
print("\nΔ summary for low‐ERD group:")
print(low_erd["accuracy_delta"].describe().to_string())
print("\nΔ summary for high‐ERD group:")
print(df[df["erd"] >= ERD_THRESHOLD]["accuracy_delta"].describe().to_string())

# ====== 3. VISUALIZE ERD DISTRIBUTIONS BY CLUSTER ======
plt.figure(figsize=(7,4))
sns.boxplot(x="cluster_label", y="erd", data=df, palette="viridis")
plt.xlabel("Cluster Label")
plt.ylabel("ERD (μ‐band % drop)")
plt.title("ERD by Original K-Means Cluster")
plt.tight_layout()
plt.savefig("erd_by_cluster.png")
plt.close()

# ====== 4. FIT A LINEAR MODEL: Δ ~ baseline + ERD ======
# Prepare design matrix
X = df[["accuracy_pooled", "erd"]].values   # columns: baseline, ERD
y = df["accuracy_delta"].values.reshape(-1, 1)

# Fit linear regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print("\nLinear model Δ = β₀ + β₁·(baseline) + β₂·(ERD) + ε")
print(f"  β₀ (intercept) = {model.intercept_[0]:.4f}")
print(f"  β₁ (baseline coef) = {model.coef_[0][0]:.4f}")
print(f"  β₂ (ERD coef)      = {model.coef_[0][1]:.4f}")
print(f"  R² of model       = {r2:.3f}")

# Scatter 3D‐style / color‐coded by predicted vs. actual Δ
plt.figure(figsize=(6,4))
scatter = sns.scatterplot(
    x=df["accuracy_pooled"],
    y=df["accuracy_delta"],
    hue=y_pred.flatten(),
    palette="coolwarm",
    s=60,
    edgecolor="k"
)
plt.xlabel("Baseline Accuracy")
plt.ylabel("Accuracy Δ")
plt.title("Δ vs. Baseline (colored by Predicted Δ)")

# Grab the mappable and pass it to colorbar()
mappable = scatter.collections[0]
plt.colorbar(mappable, label="Predicted Δ")

plt.tight_layout()
plt.savefig("delta_vs_baseline_with_predicted.png")
plt.close()

# ====== 5. OPTIONAL: TRY AN ALTERNATIVE ERD WINDOW (0.5–1.5 s) ======
# If you want to recompute ERD with a narrower MI window, uncomment below:

with open("dump/preprocessed_data_custom.pkl", "rb") as f:
     preproc = pickle.load(f)

alt_erds = {}
for subj in df["subject"].astype(str):
    entry = preproc.get(subj, None)
    if entry is None or "train" not in entry:
        continue     
    epochs = entry["train"]
    data = epochs.get_data()
    times = epochs.times
    # assume epochs start at 0
    baseline_mask_alt = (times >= 0.0) & (times < 0.5)
    mi_mask_alt       = (times >= 0.5) & (times < 1.5)
    if not (baseline_mask_alt.any() and mi_mask_alt.any()):
        continue
    bpow = (data[..., baseline_mask_alt]**2).mean(axis=(1,2))
    mpow = (data[..., mi_mask_alt]**2).mean(axis=(1,2))
    erd_trials_alt = (bpow - mpow) / bpow
    alt_erds[int(subj)] = erd_trials_alt.mean()

df_alt = df.copy()
df_alt["erd_alt"] = df_alt["subject"].map(alt_erds)
df_alt = df_alt.dropna(subset=["erd_alt"])
r_alt, p_alt = pearsonr(df_alt["erd_alt"], df_alt["accuracy_pooled"])
print(f"\nAlternative ERD (0.5–1.5 s) vs. baseline: r = {r_alt:.2f}, p = {p_alt:.4g}")
# You could then repeat the same scatter/regression-plot procedure for erd_alt.

print("\nAll analyses completed. Plots and statistics are saved in the working directory.")
