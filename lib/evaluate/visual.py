#!/usr/bin/env python
"""
visual.py

This module provides functions to create qualitative (visual) evaluations:
  - Plotting learning curves
  - Plotting ROC curves (for multiclass, one-vs-rest)
  - Visualizing high-dimensional features using t-SNE and PCA (2D/3D)

Usage:
  Import the functions in your main evaluation script.
"""

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

def plot_learning_curve(loss_curve, subj, run, mode, output_dir):
    """Plot and save the learning curve."""
    plt.figure(figsize=(8, 4))
    plt.plot(loss_curve, marker='o')
    plt.title(f"Learning Curve - Subject {subj} (Run {run}, Mode: {mode})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    out_path = os.path.join(output_dir, f"learning_curve_subj{subj}_run{run}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Learning curve saved as {out_path}")

def plot_roc_curves(all_true, all_probs, n_classes, subj, run, mode, output_dir):
    """Plot ROC curves for each class and save the figure."""
    y_bin = label_binarize(all_true, classes=list(range(n_classes)))
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC Curves - Subject {subj} (Run {run}, Mode: {mode})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    out_path = os.path.join(output_dir, f"roc_curves_subj{subj}_run{run}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"ROC curves saved as {out_path}")

def plot_tsne(X, y, title, filename, perplexity=30, n_iter=1000):
    """Plot 2D t-SNE visualization and save the figure."""
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
    """Plot PCA visualization (2D or 3D) and save the figure."""
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
