#!/usr/bin/env python
"""
evaluate_features.py

This script loads saved feature extraction outputs (from your pipeline)
and evaluates each method using a simple Linear Discriminant Analysis (LDA)
classifier with cross-validation. It computes classification accuracy, Cohen's kappa,
confusion matrix, and also provides visualizations (PCA and t-SNE) of the feature space.

Usage:
    python evaluate_features.py --features_file features.pkl --labels_file labels.pkl
"""

import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def evaluate_features(features, labels):
    """
    Evaluate features using LDA with stratified 5-fold cross-validation.
    Returns the accuracy, kappa, and confusion matrix.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lda = LDA()
    predicted = cross_val_predict(lda, features, labels, cv=skf)
    
    acc = accuracy_score(labels, predicted)
    kappa = cohen_kappa_score(labels, predicted)
    cm = confusion_matrix(labels, predicted)
    
    print("Accuracy:", acc)
    print("Kappa:", kappa)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(labels, predicted))
    return acc, kappa, cm

def visualize_features(features, labels, method_name):
    """
    Visualize the feature space using PCA and t-SNE.
    """
    # PCA Visualization
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    
    plt.figure(figsize=(8,6))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(features_pca[idx, 0], features_pca[idx, 1], label=f"Class {label}")
    plt.title(f"PCA Visualization - {method_name}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.savefig(f"{method_name}_pca.png")
    plt.close()
    
    # t-SNE Visualization
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)
    
    plt.figure(figsize=(8,6))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(features_tsne[idx, 0], features_tsne[idx, 1], label=f"Class {label}")
    plt.title(f"t-SNE Visualization - {method_name}")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend()
    plt.savefig(f"{method_name}_tsne.png")
    plt.close()

def main(features_file, labels_file):
    # Load saved features and labels (features should be a dict: {method_name: features_array})
    feature_dict = load_pickle(features_file)
    labels = load_pickle(labels_file)  # labels: np.array of shape (n_epochs,)
    
    # Evaluate each feature extraction method individually
    for method_name, feats in feature_dict.items():
        print(f"\nEvaluating method: {method_name}")
        acc, kappa, cm = evaluate_features(feats, labels)
        visualize_features(feats, labels, method_name)
        # Optionally, save the metrics per method to a file
        # For example: write to CSV or log the results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Feature Extraction Methods")
    parser.add_argument("--features_file", type=str, default="features.pkl", help="Path to pickle file with features")
    parser.add_argument("--labels_file", type=str, default="labels.pkl", help="Path to pickle file with labels")
    args = parser.parse_args()
    main(args.features_file, args.labels_file)
