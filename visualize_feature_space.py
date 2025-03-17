#!/usr/bin/env python
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_features(features_file):
    with open(features_file, 'rb') as f:
        features_dict = pickle.load(f)
    return features_dict

def inspect_features(features_dict):
    print("Inspecting loaded features dictionary...")
    subjects = list(features_dict.keys())
    print("Subjects available:", subjects)
    for subj in subjects:
        sessions = list(features_dict[subj].keys())
        print(f"Subject {subj} sessions:", sessions)
        for sess in sessions:
            keys = list(features_dict[subj][sess].keys())
            print(f"  Session '{sess}' keys:", keys)
    print("Inspection complete.\n")

def combine_features_and_labels_for_session(session_data, method_name):
    if method_name in session_data and 'labels' in session_data:
        return session_data[method_name], session_data['labels']
    else:
        raise ValueError(f"Method {method_name} or 'labels' not found in session.")

def plot_tsne(X, y, method_name, subj, sess, perplexity=30, n_iter=1000):
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', s=10)
    plt.title(f"t-SNE of {method_name} Features - Subject {subj} Session {sess}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(scatter, label="Class Label")
    plt.tight_layout()
    filename = f"tsne_{method_name}_subj{subj}_sess{sess}.png"
    plt.savefig(filename)
    plt.close()
    print(f"t-SNE plot saved for subject {subj} session {sess} as {filename}")

if __name__ == "__main__":
    # Change to your specific features file (e.g., features_riemannian_pca.pkl)
    features_file = "outputs/2025-03-12/fbcsp/fbcsp_features.pkl"
    features_dict = load_features(features_file)
    
    inspect_features(features_dict)
    
    method = 'fbcsp'  # Choose the method to visualize
    for subj in sorted(features_dict.keys()):
        for sess in features_dict[subj]:
            try:
                X, y = combine_features_and_labels_for_session(features_dict[subj][sess], method)
                plot_tsne(X, y, method, subj, sess)
            except Exception as e:
                print(f"Error for subject {subj} session {sess}: {e}")
