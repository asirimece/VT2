# lib/pipeline/cluster/cluster.py

import numpy as np
import pickle

from lib.pipeline.cluster.methods import (
    evaluate_k_means,
    kmeans_clustering,
    hierarchical_clustering,
    dbscan_clustering,
)

class ClusterWrapper:
    def __init__(self, subject_ids, labels, model, subject_representations):
        """
        Encapsulates the clustering results.

        Parameters:
            subject_ids (list): List of subject identifiers (same type as raw_dict keys).
            labels (dict): Dictionary mapping subject IDs → cluster labels.
            model (object): The fitted clustering model.
            subject_representations (dict): Mapping subject IDs → their representation vectors.
        """
        # KEEP the original subject_id types!
        self.subject_ids = list(subject_ids)
        self.labels = dict(labels)
        self.model = model
        self.subject_representations = dict(subject_representations)

    def get_cluster_for_subject(self, subject_id):
        """Return the cluster label for the given subject ID."""
        return self.labels.get(subject_id, None)

    def get_num_clusters(self):
        """Return the number of unique clusters."""
        unique = np.unique(list(self.labels.values()))
        return len(unique)

    def summary(self):
        """Return a summary string describing the cluster distribution."""
        unique, counts = np.unique(list(self.labels.values()), return_counts=True)
        lines = [f"Cluster {u}: {c} subjects" for u, c in zip(unique, counts)]
        return "Clustering Summary:\n" + "\n".join(lines)


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
        self.features = self._load_features()
        self.subject_representations = self._compute_subject_representation()

    def _load_features(self):
        """Load the precomputed features from a pickle file."""
        with open(self.features_file, "rb") as f:
            return pickle.load(f)

    def _compute_subject_representation(self):
        """
        Compute a subject‐level representation by aggregating the 'combined' features
        across all sessions, then taking the mean vector.
        """
        subject_repr = {}
        for subj, sessions in self.features.items():
            # KEEP subj in its original type (int, str, etc.)
            reps = [sess['combined'] for sess in sessions.values()]
            reps_concat = np.concatenate(reps, axis=0)  # (n_trials, n_features)
            subject_repr[subj] = np.mean(reps_concat, axis=0)
        return subject_repr

    def cluster_subjects(self, method=None):
        """
        Cluster subjects based on their aggregated representation vectors.
        """
        if method is None:
            method = self.config.get('method', 'kmeans')
        subject_ids = list(self.subject_representations.keys())
        X = np.stack([self.subject_representations[sid] for sid in subject_ids], axis=0)
        method = method.lower()
        if method == 'kmeans':
            params = self.config.get('kmeans', {})
            print("[DEBUG] KMeans params:", params)
            labels, model = kmeans_clustering(X, **params)
        elif method == 'hierarchical':
            params = self.config.get('hierarchical', {})
            print("[DEBUG] Hierarchical params:", params)
            labels, model = hierarchical_clustering(X, **params)
        elif method == 'dbscan':
            params = self.config.get('dbscan', {})
            print("[DEBUG] DBSCAN params:", params)
            labels, model = dbscan_clustering(X, **params)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Build a dict: subject_id → label, preserving original subject_id type
        cluster_labels = {sid: lab for sid, lab in zip(subject_ids, labels)}
        return ClusterWrapper(subject_ids, cluster_labels, model, self.subject_representations)
