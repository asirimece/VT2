#!/usr/bin/env python
import numpy as np
import pickle
import yaml
import matplotlib.pyplot as plt
from lib.cluster.methods import evaluate_k_means, kmeans_clustering, hierarchical_clustering, dbscan_clustering

class ClusterWrapper:
    def __init__(self, subject_ids, labels, model, subject_representations):
        """
        Encapsulates the clustering results.

        Parameters:
            subject_ids (list): List of subject identifiers.
            labels (dict): Dictionary mapping subject IDs to cluster labels.
            model (object): The clustering model instance.
            subject_representations (dict): Mapping subject IDs to their representation vectors.
        """
        # Convert subject IDs and dictionary keys to strings for consistency.
        self.subject_ids = [str(sid) for sid in subject_ids]
        self.labels = {str(k): v for k, v in labels.items()}
        self.model = model
        self.subject_representations = {str(k): v for k, v in subject_representations.items()}

    def get_cluster_for_subject(self, subject_id):
        """Return the cluster label for the given subject ID."""
        return self.labels.get(str(subject_id), None)

    def get_num_clusters(self):
        """Return the number of unique clusters."""
        unique = np.unique(list(self.labels.values()))
        return len(unique)

    def summary(self):
        """Return a summary string describing the cluster distribution."""
        unique, counts = np.unique(list(self.labels.values()), return_counts=True)
        summary_lines = [f"Cluster {u}: {c} subjects" for u, c in zip(unique, counts)]
        return "Clustering Summary:\n" + "\n".join(summary_lines)


class SubjectClusterer:
    def __init__(self, features_file, config):
        """
        Initializes the subject clusterer.

        Parameters:
            features_file (str): Path to the features.pkl file.
            config (dict): Clustering configuration parameters.
        """
        self.features_file = features_file
        self.config = config
        self.features = self.load_features()
        # Compute aggregated representations without using PCA.
        self.subject_representations = self.compute_subject_representation()

    def load_features(self):
        """Load the precomputed features from a pickle file."""
        with open(self.features_file, "rb") as f:
            features = pickle.load(f)
        return features

    def compute_subject_representation(self):
        """
        Compute a subject-level representation by aggregating the 'combined' features across sessions.
        In this version, we do NOT perform PCA.
        """
        subject_repr = {}
        for subj, sessions in self.features.items():
            reps = []
            for sess in sessions.values():
                reps.append(sess['combined'])
            reps_concat = np.concatenate(reps, axis=0)  # shape: (n_trials_total, n_features)
            # Compute the mean feature vector across all trials.
            subject_repr[str(subj)] = np.mean(reps_concat, axis=0)
        return subject_repr

    def cluster_subjects(self, method='hierarchical'):
        """
        Cluster subjects based on their aggregated representation vectors (without PCA).
        """
        subject_ids = list(self.subject_representations.keys())
        X = np.array([self.subject_representations[sid] for sid in subject_ids])
        print("Min:", X.min(), "Max:", X.max(), "Mean:", X.mean(), "Std:", X.std())

        method = method.lower()
        if method == 'kmeans':
            params = self.config.get('kmeans', {})
            print("DEBUG: Clustering using KMeans with parameters:", params)
            labels, model = kmeans_clustering(X, **params)
        elif method == 'hierarchical':
            params = self.config.get('hierarchical', {})
            print("DEBUG: Clustering using Hierarchical Clustering with parameters:", params)
            labels, model = hierarchical_clustering(X, **params)
        elif method == 'dbscan':
            params = self.config.get('dbscan', {})
            print("DEBUG: Clustering using DBSCAN with parameters:", params)
            labels, model = dbscan_clustering(X, **params)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Build a dictionary with subject IDs as keys.
        cluster_labels = {subj: label for subj, label in zip(subject_ids, labels)}
        return ClusterWrapper(subject_ids, cluster_labels, model, self.subject_representations)


if __name__ == "__main__":
    # Load your YAML configuration file.
    config_path = "config/experiment/mtl.yaml"  # Adjust path as needed.
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Configuration is nested under 'experiment'
    clustering_config = config.get('experiment', {}).get('clustering', {})

    # --- DEBUG PRINTS ---
    print("DEBUG: Loaded full configuration:")
    print(config)
    print("DEBUG: Loaded clustering configuration:")
    print(clustering_config)
    # ---------------------

    features_file = "./outputs/features.pkl"  # Path to your features file.
    clusterer = SubjectClusterer(features_file, clustering_config)
    method = clustering_config.get('method', 'hierarchical')
    cluster_result = clusterer.cluster_subjects(method=method)
    print(cluster_result.summary())

    # Evaluate different values of k using the Elbow Method and Silhouette Score.
    # Construct the feature matrix X from the computed subject representations.
    subject_ids = list(clusterer.subject_representations.keys())
    X = np.array([clusterer.subject_representations[sid] for sid in subject_ids])
    
    # Get base KMeans parameters from the config (except n_clusters)
    base_kmeans_params = clustering_config.get("kmeans", {}).copy()
    if "n_clusters" in base_kmeans_params:
        del base_kmeans_params["n_clusters"]
    
    # Define a range of k values to test – for example, from 2 up to 6 (or based on your data).
    k_values = list(range(2, min(10, len(subject_ids)) + 1))
    print("\nEvaluating KMeans clustering for different values of k...")
    evaluate_k_means(X, base_kmeans_params, k_values)
