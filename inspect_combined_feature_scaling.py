#!/usr/bin/env python
"""
feature_statistics.py

This script loads the combined features from a pickle file (the output of your PCA-LASSO pipeline),
computes summary statistics (mean, standard deviation, minimum, and maximum) for the combined features,
and produces histogram and box plots for visual inspection of the feature distributions.

Usage:
    python feature_statistics.py --features_file <path_to_features_pickle>
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_features(features_file):
    with open(features_file, 'rb') as f:
        features_dict = pickle.load(f)
    return features_dict

def compute_summary_stats(X):
    """
    Compute summary statistics for the given feature matrix X.
    
    Parameters:
        X (np.ndarray): Feature matrix with shape (n_trials, n_features).
        
    Returns:
        stats (dict): Dictionary with mean, std, min, and max.
    """
    stats = {
        "mean": np.mean(X, axis=0),
        "std": np.std(X, axis=0),
        "min": np.min(X, axis=0),
        "max": np.max(X, axis=0)
    }
    return stats

def plot_distribution(X, title, output_prefix):
    """
    Create and save histogram and box plots for the flattened feature distribution.
    
    Parameters:
        X (np.ndarray): Feature matrix with shape (n_trials, n_features).
        title (str): Title for the plots.
        output_prefix (str): Prefix for the output file names.
    """
    # Flatten the matrix to a 1D array for overall distribution
    X_flat = X.flatten()
    
    # Histogram plot
    plt.figure(figsize=(8, 6))
    sns.histplot(X_flat, bins=50, kde=True, color='steelblue')
    plt.title(f"{title} - Histogram")
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    hist_filename = f"{output_prefix}_histogram.png"
    plt.tight_layout()
    plt.savefig(hist_filename)
    plt.close()
    print(f"Histogram saved as {hist_filename}")
    
    # Box plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=X_flat, color='lightgreen')
    plt.title(f"{title} - Box Plot")
    plt.xlabel("Feature Value")
    box_filename = f"{output_prefix}_boxplot.png"
    plt.tight_layout()
    plt.savefig(box_filename)
    plt.close()
    print(f"Box plot saved as {box_filename}")

def inspect_features(features_file):
    features_dict = load_features(features_file)
    
    print("Inspecting combined features...")
    subjects = sorted(features_dict.keys(), key=lambda x: int(x))
    for subj in subjects:
        print(f"\nSubject {subj}:")
        sessions = features_dict[subj]
        print("  Sessions:", list(sessions.keys()))
        for sess, data in sessions.items():
            print(f"  Session '{sess}' keys: {list(data.keys())}")
            if 'combined' not in data:
                print(f"    No combined features found for subject {subj}, session {sess}.")
                continue
            X = data['combined']
            y = data['labels']
            n_trials, n_features = X.shape
            print(f"    Combined features shape: {X.shape}")
            print(f"    Labels shape: {y.shape}")
            
            # Compute summary statistics
            stats = compute_summary_stats(X)
            overall_mean = np.mean(stats['mean'])
            overall_std = np.mean(stats['std'])
            overall_min = np.min(stats['min'])
            overall_max = np.max(stats['max'])
            print(f"    Overall feature statistics (averaged over dimensions):")
            print(f"       Mean: {overall_mean:.3f}, Std: {overall_std:.3f}, Min: {overall_min:.3f}, Max: {overall_max:.3f}")
            
            # Create plots for the overall distribution
            title = f"Subj {subj} {sess} Combined Features"
            output_prefix = f"features_subj{subj}_{sess}_combined"
            plot_distribution(X, title, output_prefix)
    print("Feature inspection complete.")

def main():
    parser = argparse.ArgumentParser(description="Feature Statistics and Visualization")
    parser.add_argument("--features_file", type=str, default="outputs/2025-03-15/16-32-53/features___.pkl",
                        help="Path to the combined features pickle file")
    args = parser.parse_args()
    
    inspect_features(args.features_file)

if __name__ == "__main__":
    main()
