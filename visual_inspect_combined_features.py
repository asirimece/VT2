#!/usr/bin/env python
"""
visual_inspection_features.py

This script loads a pickle file containing the combined features for each subject and session,
applies dimensionality reduction using PCA, and uses both 2D and 3D (if requested) projections,
and t‑SNE to visualize the high-dimensional feature vectors. Additionally, it computes quantitative 
metrics (silhouette score and Davies–Bouldin index) to assess cluster separability based on trial labels.

Each trial in the features dictionary should have two keys:
  - 'combined': the combined feature vector for that trial (one row per trial)
  - 'labels': the corresponding label for that trial

Usage:
  python visual_inspection_features.py

Make sure you point 'features_file' to your saved combined features pickle file.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

def load_features(features_file):
    with open(features_file, 'rb') as f:
        features_dict = pickle.load(f)
    return features_dict

def compute_clustering_metrics(X, y):
    """
    Compute quantitative metrics (silhouette score and Davies-Bouldin index)
    using the provided feature matrix X and ground truth labels y.
    """
    sil_score = silhouette_score(X, y, metric='euclidean')
    db_index = davies_bouldin_score(X, y)
    return sil_score, db_index

def plot_tsne(X, y, title, filename, perplexity=30, n_iter=1000):
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', s=10)
    plt.title(f"{title} (t-SNE)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(scatter, label="Class Label")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"t-SNE plot saved as {filename}")

def plot_pca(X, y, title, filename, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    
    if n_components == 2:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=10)
        plt.title(f"{title} (PCA 2D)")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.colorbar(scatter, label="Class Label")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"PCA 2D plot saved as {filename}")
    elif n_components >= 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', s=10)
        ax.set_title(f"{title} (PCA 3D)")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_zlabel("PCA 3")
        fig.colorbar(scatter, ax=ax, label="Class Label")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"PCA 3D plot saved as {filename}")
    else:
        print("Invalid n_components for PCA plotting. Must be at least 2.")

def inspect_features(features_file, method_name, pca_n_components=2):
    features_dict = load_features(features_file)
    
    print("Inspecting combined features...")
    print("Subjects available:", sorted(features_dict.keys()))
    
    # For each subject and session, run dimensionality reduction and compute metrics.
    for subj, sessions in sorted(features_dict.items(), key=lambda x: int(x[0])):
        print(f"\nSubject {subj}:")
        print("  Sessions:", list(sessions.keys()))
        for sess, data in sessions.items():
            keys = list(data.keys())
            print(f"  Session '{sess}' keys: {keys}")
            if method_name not in data:
                print(f"    Method '{method_name}' not found in subject {subj}, session {sess}.")
                continue
            X = data[method_name]
            y = data['labels']
            print(f"    Combined features shape: {X.shape}")
            print(f"    Labels shape: {y.shape}")
            
            # Compute quantitative metrics on the extracted features
            sil_score, db_index = compute_clustering_metrics(X, y)
            print(f"    Silhouette Score: {sil_score:.3f}")
            print(f"    Davies-Bouldin Index: {db_index:.3f}")
            
            title = f"Subj {subj} {sess} - {method_name}"
            tsne_filename = f"tsne_subj{subj}_{sess}_{method_name}.png"
            pca_filename = f"pca_subj{subj}_{sess}_{method_name}.png"
            plot_tsne(X, y, title, tsne_filename)
            plot_pca(X, y, title, pca_filename, n_components=pca_n_components)
    print("Inspection complete.")

if __name__ == "__main__":
    # Set this to your combined features pickle file path
    features_file = "outputs/22ica/22ica_features.pkl"
    # Method name as stored in the features dictionary, e.g., "combined"
    method_name = "combined"
    # Set number of PCA components for visualization (2 for 2D or 3 for 3D)
    pca_n_components = 3  # Change to 2 for 2D plotting if needed
    inspect_features(features_file, method_name, pca_n_components=pca_n_components)
