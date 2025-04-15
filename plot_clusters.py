#!/usr/bin/env python
"""
plot_clusters.py

This script loads the features from a features.pkl file and a clustering configuration
(from a YAML file), runs clustering for each specified method (e.g., kmeans, hierarchical, dbscan)
using your SubjectClusterer class, and produces a scatter plot of subject-level representations
(reduced to 2D via PCA) colored according to cluster assignments.
The resulting plots are saved to disk.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import yaml
import argparse

# Import your SubjectClusterer from your clustering module.
from lib.cluster.cluster import SubjectClusterer

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def plot_clusters(subject_representations, cluster_assignments, method, output_dir="cluster_plots"):
    """
    Plots a 2D scatter of subject-level representations reduced via PCA,
    coloring each point according to its cluster label.
    
    Parameters:
      subject_representations (dict): Mapping subject ID -> representation vector.
      cluster_assignments (dict): Mapping subject ID -> cluster label.
      method (str): The clustering method name (used in the plot title and filename).
      output_dir (str): Directory to save the plot.
    """
    # Order subjects consistently.
    subjects = sorted(subject_representations.keys())
    X = np.array([subject_representations[subj] for subj in subjects])
    labels = [cluster_assignments.get(subj, -1) for subj in subjects]
    
    # Reduce features to 2D using PCA.
    #pca = PCA(n_components=2)
    #X_reduced = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=100)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"Cluster Assignments using {method}")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster Label")
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"clusters_{method}.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Saved cluster plot for {method} to {output_file}")

#!/usr/bin/env python
"""
plot_clusters.py

This script loads the features from a features.pkl file and the clustering configuration
from a YAML file, runs clustering for each specified method (e.g., kmeans, hierarchical, dbscan)
using your SubjectClusterer class, and produces a scatter plot of subject-level representations
(reduced to 2D via PCA) colored by their cluster assignments. Each point is annotated with the subject ID
and its cluster label. The resulting plots (and a printed summary) are saved to disk.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import yaml
import argparse

# Import your SubjectClusterer from your clustering module.
from lib.cluster.cluster import SubjectClusterer

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def plot_clusters(subject_representations, cluster_assignments, method, output_dir="cluster_plots"):
    """
    Plots a 2D scatter of subject-level representations after reducing via PCA.
    Each point is annotated with its subject ID and cluster label.
    
    Parameters:
      subject_representations (dict): Mapping of subject ID -> representation vector.
      cluster_assignments (dict): Mapping of subject ID -> cluster label.
      method (str): Name of the clustering method (for plot title and filename).
      output_dir (str): Directory to save the plot.
    """
    # Sort subjects to ensure consistent order.
    subjects = sorted(subject_representations.keys())
    X = np.array([subject_representations[subj] for subj in subjects])
    # Get the cluster label for each subject (or -1 if not available).
    labels = [cluster_assignments.get(subj, -1) for subj in subjects]
    
    # Reduce the high-dimensional representations to 2D using PCA.
    #pca = PCA(n_components=2)
    #X_reduced = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=100)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"Cluster Assignments using {method}")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster Label")
    
    # Annotate each point with the subject ID and cluster label.
    for i, subj in enumerate(subjects):
        plt.annotate(f"{subj} (cl: {labels[i]})",
                     (X[i, 0], X[i, 1]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"clusters_{method}.png")
    plt.savefig(output_file)
    plt.close()
    print(f"--- Cluster assignments for {method} ---")
    for subj in subjects:
        cl = cluster_assignments.get(subj, -1)
        print(f"Subject: {subj}, Cluster: {cl}")
    print(f"Saved cluster plot for {method} to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Plot clusters for different clustering methods using features.pkl")
    parser.add_argument("--config", type=str, default="/home/ubuntu/VT2/config/experiment/mtl.yaml",
                        help="Path to clustering configuration YAML file")
    parser.add_argument("--features", type=str, default="outputs/features.pkl",
                        help="Path to features.pkl file")
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["kmeans", "hierarchical", "dbscan"],
                        help="List of clustering methods to plot")
    parser.add_argument("--output_dir", type=str, default="cluster_plots",
                        help="Directory to save the cluster plots")
    args = parser.parse_args()
    
    # Load clustering configuration.
    config = load_config(args.config)
    clustering_config = config.get('experiment', {}).get('clustering', {})
    
    for method in args.methods:
        print(f"Processing clustering method: {method}")
        # Instantiate SubjectClusterer and run clustering for the given method.
        subject_clusterer = SubjectClusterer(args.features, clustering_config)
        cluster_wrapper = subject_clusterer.cluster_subjects(method=method)
        subject_representations = cluster_wrapper.subject_representations
        cluster_assignments = cluster_wrapper.labels
        plot_clusters(subject_representations, cluster_assignments, method, output_dir=args.output_dir)

if __name__ == "__main__":
    main()

