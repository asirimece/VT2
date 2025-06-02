"""import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ---- Load both feature sets ----
with open("./dump/deep_features.pkl", "rb") as f:
    deep_features = pickle.load(f)

with open("./dump/features.pkl", "rb") as f:
    fbcsp_features = pickle.load(f)

subject_ids = sorted(deep_features.keys())  # assumes same keys in both

# ---- Compute raw feature matrices ----
deep_matrix = np.stack([deep_features[sid].mean(axis=0) for sid in subject_ids])
fbcsp_matrix = np.stack([fbcsp_features[sid]['train']['combined'].mean(axis=0) for sid in subject_ids])

print("Deep feature shape:", deep_matrix.shape)
print("FBCSP feature shape:", fbcsp_matrix.shape)

# ---- Print stats before scaling ----
print("Deep features: mean =", deep_matrix.mean(), ", std =", deep_matrix.std())
print("FBCSP features: mean =", fbcsp_matrix.mean(), ", std =", fbcsp_matrix.std())

# ---
print("Per-feature std in FBCSP:", fbcsp_matrix.std(axis=0))
plt.imshow(fbcsp_matrix, aspect="auto", cmap="viridis")
plt.colorbar()
plt.title("FBCSP Matrix (Subjects × Features)")
plt.savefig("./dump/fbcsp_matrix_visual.png")
plt.close()

print("Per-feature std in Deep Features:", deep_matrix.std(axis=0))
plt.imshow(deep_matrix, aspect="auto", cmap="viridis")
plt.colorbar()
plt.title("Deep Features Matrix (Subjects × Features)")
plt.savefig("./dump/dp_matrix_visual.png")
plt.close()
# ---

# ---- Plot histogram of feature values before scaling ----
plt.hist(deep_matrix.flatten(), bins=100, alpha=0.5, label="Deep")
plt.hist(fbcsp_matrix.flatten(), bins=100, alpha=0.5, label="FBCSP")
plt.legend()
plt.title("Feature Value Distributions (Before Scaling)")
plt.tight_layout()
plt.savefig("./dump/feature_value_distributions.png")
plt.close()
print("Saved histogram: ./dump/feature_value_distributions.png")

# ---- Standardize each separately ----
deep_scaled = StandardScaler().fit_transform(deep_matrix)
fbcsp_scaled = StandardScaler().fit_transform(fbcsp_matrix)

# ---- Concatenate scaled features ----
X = np.concatenate([deep_scaled, fbcsp_scaled], axis=1)  # shape: (85, 416)

print("Concatenated scaled feature shape (n_subjects x n_features):", X.shape)

# ---- Try different n_clusters ----
cluster_range = [2, 3, 4, 5]
results = []

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)

    sil = silhouette_score(X, labels)
    db  = davies_bouldin_score(X, labels)

    print(f"n_clusters={n_clusters}: Silhouette={sil:.3f}, DB index={db:.3f}")
    results.append((n_clusters, sil, db))

    # --- t-SNE plot ---
    tsne = TSNE(n_components=2, random_state=42)
    X_emb = tsne.fit_transform(X)
    plt.figure(figsize=(6,5))
    for i in range(n_clusters):
        plt.scatter(X_emb[labels==i,0], X_emb[labels==i,1], label=f"Cluster {i}", s=30)
    plt.title(f"t-SNE, n_clusters={n_clusters}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./dump/tsne_clusters_concat_scaled_{n_clusters}.png")
    plt.close()
    print(f"Saved t-SNE plot: ./dump/tsne_clusters_concat_scaled_{n_clusters}.png")

# ---- Summary plot of scores ----
sils, dbs = zip(*[(sil, db) for _, sil, db in results])
plt.figure()
plt.plot(cluster_range, sils, marker="o", label="Silhouette Score")
plt.plot(cluster_range, dbs, marker="o", label="Davies-Bouldin Index")
plt.xlabel("n_clusters")
plt.ylabel("Score (higher=sil, lower=DB is better)")
plt.title("Cluster Quality: Concatenated + Scaled")
plt.legend()
plt.tight_layout()
plt.savefig("./dump/cluster_quality_concat_scaled_summary.png")
plt.close()
print("Saved summary plot: ./dump/cluster_quality_concat_scaled_summary.png")
"""


import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ---- Load features ----
with open("./dump/features.pkl", "rb") as f:
    features = pickle.load(f)

subject_ids = list(features.keys())
# Compute mean feature vector per subject from 'train' set
X = np.stack([
    features[sid]['train']['combined'].mean(axis=0)
    for sid in subject_ids
])  # shape: (n_subjects, n_features)

print("Subject feature matrix shape (should be n_subjects x n_features):", X.shape)

# ---- Print stats before scaling ----
print("FBCSP features: mean =", X.mean(), ", std =", X.std())

# ---
print("Per-feature std in FBCSP:", X.std(axis=0))
plt.imshow(X, aspect="auto", cmap="viridis")
plt.colorbar()
plt.title("FBCSP Matrix (Subjects × Features)")
plt.savefig("./dump/fbcsp_matrix_visual.png")
plt.close()

# ---- Try different n_clusters ----
cluster_range = [2, 3, 4, 5]
results = []

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)

    sil = silhouette_score(X, labels)
    db  = davies_bouldin_score(X, labels)

    print(f"n_clusters={n_clusters}: Silhouette={sil:.3f}, DB index={db:.3f}")
    results.append((n_clusters, sil, db))

    # --- t-SNE plot colored by cluster ---
    tsne = TSNE(n_components=2, random_state=42)
    X_emb = tsne.fit_transform(X)
    plt.figure(figsize=(6,5))
    for i in range(n_clusters):
        plt.scatter(X_emb[labels==i,0], X_emb[labels==i,1], label=f"Cluster {i}", s=30)
    plt.title(f"t-SNE, n_clusters={n_clusters}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./dump/tsne_clusters_{n_clusters}.png")
    plt.close()
    print(f"Saved t-SNE plot: ./dump/tsne_clusters_{n_clusters}.png")

# ---- Summary plot of scores ----
sils, dbs = zip(*[(sil, db) for _, sil, db in results])
plt.figure()
plt.plot(cluster_range, sils, marker="o", label="Silhouette Score")
plt.plot(cluster_range, dbs, marker="o", label="Davies-Bouldin Index")
plt.xlabel("n_clusters")
plt.ylabel("Score (higher=sil, lower=DB is better)")
plt.title("Cluster Quality Metrics")
plt.legend()
plt.tight_layout()
plt.savefig("./dump/cluster_quality_summary.png")
plt.close()
print("Saved summary plot: ./dump/cluster_quality_summary.png")
