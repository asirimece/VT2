# lib/cluster.py
import numpy as np
from sklearn.cluster import KMeans

def compute_subject_vector(features_dict, session="0train"):
    """
    Compute a subject-level vector vector by averaging the 'combined' features
    across all trials for a given session (typically the training session).
    
    Parameters
    ----------
    features_dict : dict
        Dictionary structured as {subject: {session: {'combined': X, 'labels': y}, ...}, ...}
    session : str, default "0train"
        The session from which to compute the vector.
        
    Returns
    -------
    subject_vector : dict
        Dictionary mapping each subject to its vector feature vector.
    subjects : list
        List of subject keys (as strings) for which a vector was computed.
    """
    subject_vector = {}
    subjects = []
    for subj, sessions in features_dict.items():
        if session in sessions:
            X = sessions[session]['combined']  # shape (n_trials, n_features)
            vector = np.mean(X, axis=0)  # average over trials
            subject_vector[subj] = vector
            subjects.append(subj)
    return subject_vector, subjects

def subject_level_clustering(subject_vector, n_clusters=3):
    """
    Perform k-means clustering on subject-level vector vectors.
    
    Parameters
    ----------
    subject_vector : dict
        Dictionary mapping subject to vector vector (numpy array).
    n_clusters : int, default 3
        Number of clusters to form.
        
    Returns
    -------
    cluster_labels : dict
        Dictionary mapping each subject to its cluster label.
    kmeans_model : KMeans
        The fitted k-means clustering model.
    """
    subjects = list(subject_vector.keys())
    X_vector = np.array([subject_vector[subj] for subj in subjects])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_vector)
    cluster_labels = {subj: label for subj, label in zip(subjects, labels)}
    return cluster_labels, kmeans
