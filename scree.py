import pickle
import numpy as np
import matplotlib.pyplot as plt

features_pkl_path = "dump/features.pkl"
output_path = "scree_plot.png"

# 1. Load features.pkl
with open(features_pkl_path, "rb") as f:
    features_obj = pickle.load(f)

# 2. Collect all features
all_features = []
for subj_id, subj_dict in features_obj.items():
    for split in ['train', 'test']:
        arr = subj_dict[split]['combined']
        all_features.append(arr)
all_features = np.concatenate(all_features, axis=0)  # shape: (total_samples, n_features)

# 3. Compute covariance and eigenvalues
X = all_features - all_features.mean(axis=0, keepdims=True)
cov = np.cov(X, rowvar=False)
eigenvalues, _ = np.linalg.eigh(cov)   # ascending order
eigenvalues = eigenvalues[::-1]        # descending

# 4. Scree plot
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, 'o-', markersize=5)
plt.title("Scree Plot (Eigenvalues of Covariance Matrix)")
plt.xlabel("Principal Component")
plt.ylabel("Eigenvalue")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_path, dpi=150)
print(f"Scree plot saved as {output_path}")
