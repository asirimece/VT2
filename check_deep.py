import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ---- Load Features ----
with open("./dump/deep_features.pkl", "rb") as f:
    subject_features = pickle.load(f)

print("Number of subjects:", len(subject_features))
for subj, feat in list(subject_features.items())[:5]:  # Show first 5 subjects
    print(f"Subject: {subj}, Feature shape: {feat.shape}, First 5 vals: {feat[:5]}")

all_feats = np.stack(list(subject_features.values()))  # (n_subjects, n_trials, n_features)
print("Feature array shape:", all_feats.shape)

# ---- Average across trials for each subject ----
if all_feats.ndim == 3:
    # Shape: (n_subjects, n_trials, n_features) -> (n_subjects, n_features)
    all_feats_avg = all_feats.mean(axis=1)
else:
    all_feats_avg = all_feats  # Already averaged

print("Averaged feature shape (should be n_subjects x n_features):", all_feats_avg.shape)
print("Mean (per-feature):", all_feats_avg.mean(axis=0)[:5])
print("Std (per-feature):", all_feats_avg.std(axis=0)[:5])
print("Total std across subjects:", all_feats_avg.std())

# ---- t-SNE Visualization ----
tsne = TSNE(n_components=2, random_state=42)
X_emb = tsne.fit_transform(all_feats_avg)

plt.figure(figsize=(6,5))
plt.scatter(X_emb[:,0], X_emb[:,1])
plt.title("t-SNE of Subject Deep Features (Mean Across Trials)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.tight_layout()
plt.savefig("./dump/deep_features_tsne.png")
plt.close()
print("t-SNE plot saved to ./dump/deep_features_tsne.png")

# ---- Histogram ----
plt.figure(figsize=(6,4))
plt.hist(all_feats_avg.flatten(), bins=50)
plt.title("Histogram of All Deep Feature Values (Mean Across Trials)")
plt.tight_layout()
plt.savefig("./dump/deep_features_hist.png")
plt.close()
print("Histogram plot saved to ./dump/deep_features_hist.png")
