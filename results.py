import pandas as pd
import matplotlib.pyplot as plt
import os

# --- LOAD RESULTS ---
df = pd.read_csv("dump/all_model_comparison.csv")  # Adjust path as needed

# --- CONFIG: What to group by? (Add n_clusters if needed) ---
GROUPS = ['model']  # or ['model', 'n_clusters'] if n_clusters in df

# --- 1. PER-SUBJECT MEAN ACCURACY (across runs) ---
subj_means = (
    df.groupby(GROUPS + ['subject'])['accuracy']
      .mean()
      .reset_index()
      .rename(columns={'accuracy': 'subject_mean_accuracy'})
)

# --- 2. MODEL-LEVEL SUMMARY (across subjects) ---
model_summary = (
    subj_means.groupby(GROUPS)['subject_mean_accuracy']
        .agg(['mean', 'std', 'min', 'max', 'median', 'count'])
        .reset_index()
)

print("\nMODEL-LEVEL SUMMARY (mean over subjects):")
print(model_summary)

# --- 3. Optional: Per-subject stats (useful for supplementary material) ---
subj_stats = (
    subj_means.groupby(GROUPS)['subject_mean_accuracy']
        .describe()
        .reset_index()
)
print("\nPER-SUBJECT ACCURACY STATS (describe):")
print(subj_stats)

# --- 4. Optional: Boxplot visualization ---
os.makedirs("plots", exist_ok=True)
for key, subdf in subj_means.groupby(GROUPS):
    plt.figure()
    plt.boxplot(subdf['subject_mean_accuracy'], vert=True)
    plt.title(f"Subject Accuracy Distribution: {dict(zip(GROUPS, key))}")
    plt.ylabel("Accuracy")
    plt.xlabel("Subjects")
    plt.tight_layout()
    # Name plot file based on grouping (e.g. model_TL_nclust_3.png)
    fname = "plots/" + "_".join(f"{g}_{v}" for g,v in zip(GROUPS, key)) + ".png"
    plt.savefig(fname)
    plt.close()
    print(f"Saved boxplot: {fname}")

# --- 5. Optional: Save summaries as CSV for Excel ---
model_summary.to_csv("model_summary.csv", index=False)
subj_stats.to_csv("subject_accuracy_stats.csv", index=False)
subj_means.to_csv("subject_mean_accuracies.csv", index=False)

print("\nCSV summaries saved as 'model_summary.csv', 'subject_accuracy_stats.csv', 'subject_mean_accuracies.csv'.")
