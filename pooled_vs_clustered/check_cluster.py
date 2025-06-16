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
