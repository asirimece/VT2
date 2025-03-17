#!/usr/bin/env python
"""
metric.py

This module provides functions for quantitative evaluations:
  - Computing standard classification metrics (accuracy, Cohen's kappa, confusion matrix)
  - Computing clustering metrics (silhouette score and Davies-Bouldin index)
  - Aggregating metrics across multiple runs

Usage:
  Import these functions in your evaluation or training scripts.
"""

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, silhouette_score, davies_bouldin_score

def evaluate_model(preds, true_labels):
    """
    Compute standard classification metrics.
    
    Returns:
      accuracy, kappa, confusion_matrix.
    """
    acc = accuracy_score(true_labels, preds)
    kappa = cohen_kappa_score(true_labels, preds)
    conf_mat = confusion_matrix(true_labels, preds)
    return acc, kappa, conf_mat

def compute_clustering_metrics(X, y):
    """
    Compute clustering metrics: silhouette score and Davies-Bouldin index.
    """
    sil_score = silhouette_score(X, y, metric='euclidean')
    db_index = davies_bouldin_score(X, y)
    return sil_score, db_index

def aggregate_results(all_run_results, mode):
    """
    Aggregate accuracy and kappa metrics across runs.
    For "single" mode, aggregates per subject and then overall average.
    For "pooled" mode, aggregates directly.
    """
    if mode == "single":
        subj_metrics = {}
        for run_key, run_results in all_run_results.items():
            for subj, metrics in run_results.items():
                if subj not in subj_metrics:
                    subj_metrics[subj] = {"accuracy": [], "kappa": []}
                subj_metrics[subj]["accuracy"].append(metrics["accuracy"])
                subj_metrics[subj]["kappa"].append(metrics["kappa"])
        overall_acc = []
        overall_kappa = []
        for subj, met in subj_metrics.items():
            subj_acc = np.mean(met["accuracy"])
            subj_kappa = np.mean(met["kappa"])
            overall_acc.append(subj_acc)
            overall_kappa.append(subj_kappa)
            print(f"Subject {subj} - Mean Accuracy: {subj_acc:.3f}, Mean Kappa: {subj_kappa:.3f}")
        print(f"Overall average accuracy: {np.mean(overall_acc):.3f} (std: {np.std(overall_acc):.3f})")
        print(f"Overall average kappa: {np.mean(overall_kappa):.3f} (std: {np.std(overall_kappa):.3f})")
    else:  # pooled mode
        accuracies = []
        kappas = []
        for run_key, metrics in all_run_results.items():
            accuracies.append(metrics["accuracy"])
            kappas.append(metrics["kappa"])
            print(f"{run_key} - Accuracy: {metrics['accuracy']:.3f}, Kappa: {metrics['kappa']:.3f}")
        print(f"Pooled overall average accuracy: {np.mean(accuracies):.3f} (std: {np.std(accuracies):.3f})")
        print(f"Pooled overall average kappa: {np.mean(kappas):.3f} (std: {np.std(kappas):.3f})")
