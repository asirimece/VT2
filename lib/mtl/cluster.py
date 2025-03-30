import numpy as np
import pickle
from lib.cluster.methods import kmeans_clustering, hierarchical_clustering, dbscan_clustering


class ClusterWrapper:
    def __init__(self, subject_ids, labels, model, subject_representations):
        """
        Encapsulates the clustering results.

        Parameters:
            subject_ids (list): List of subject identifiers.
            labels (dict): Dictionary mapping subject IDs to cluster labels.
            model (object): The clustering model instance (e.g., KMeans, AgglomerativeClustering, DBSCAN).
            subject_representations (dict): Dictionary mapping subject IDs to their representation vectors.
        """
        self.subject_ids = subject_ids
        self.labels = labels
        self.model = model
        self.subject_representations = subject_representations

    def get_cluster_for_subject(self, subject_id):
        """Return the cluster label for the given subject ID."""
        return self.labels.get(subject_id, None)

    def summary(self):
        """Return a summary string describing the cluster distribution."""
        unique, counts = np.unique(list(self.labels.values()), return_counts=True)
        summary_lines = [f"Cluster {u}: {c} subjects" for u, c in zip(unique, counts)]
        return "Clustering Summary:\n" + "\n".join(summary_lines)


class SubjectClusterer:
    def __init__(self, features_file, config):
        """
        features_file: Path to the features.pkl file.
        config: Dictionary of clustering configuration parameters (from your Hydra config).
        """
        self.features_file = features_file
        self.config = config
        self.features = self.load_features()
        self.subject_representations = self.compute_subject_representation()

    def load_features(self):
        with open(self.features_file, "rb") as f:
            features = pickle.load(f)
        return features

    def compute_subject_representation(self):
        """
        Compute a subject-level representation by averaging the "combined" features across sessions.
        """
        subject_repr = {}
        for subj, sessions in self.features.items():
            reps = []
            for sess in sessions.values():
                reps.append(sess['combined'])
            reps = np.concatenate(reps, axis=0)  # shape: (n_trials_total, n_features)
            subject_repr[subj] = np.mean(reps, axis=0)
        return subject_repr

    def cluster_subjects(self, method='kmeans'):
        subject_ids = list(self.subject_representations.keys())
        X = np.array([self.subject_representations[subj] for subj in subject_ids])

        if method.lower() == 'kmeans':
            params = self.config.get('kmeans', {})
            labels, model = kmeans_clustering(X, **params)
        elif method.lower() == 'hierarchical':
            params = self.config.get('hierarchical', {})
            labels, model = hierarchical_clustering(X, **params)
        elif method.lower() == 'dbscan':
            params = self.config.get('dbscan', {})
            labels, model = dbscan_clustering(X, **params)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        cluster_labels = {subj: label for subj, label in zip(subject_ids, labels)}
        # Instead of writing to disk, wrap the results in a ClusterResult instance and return it.
        return ClusterResult(subject_ids, cluster_labels, model, self.subject_representations)

# Example usage within your pipeline:
if __name__ == "__main__":
    import yaml
    # Load your Hydra YAML configuration
    config_path = "config/clustering_config.yaml"  # Adjust as needed
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    clustering_config = config.get('clustering', {})

    features_file = "features.pkl"  # Path to your features file
    clusterer = SubjectClusterer(features_file, clustering_config)
    method = clustering_config.get('method', 'kmeans')
    cluster_result = clusterer.cluster_subjects(method=method)
    print(cluster_result.summary())
    # Now cluster_result can be directly passed to your MTL stage.

