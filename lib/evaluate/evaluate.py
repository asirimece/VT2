#!/usr/bin/env python
"""
evaluate.py

This script performs the final evaluation of your baseline training.
It loads the evaluation results (e.g., SVM classification results) from a pickle file,
prints per-subject quantitative metrics (accuracy, Cohen's kappa, confusion matrices),
and generates qualitative visualizations (TSNE and PCA plots) using the visual module.
It also aggregates results across runs if needed.

Usage:
  python evaluate.py --results_file <path_to_results.pkl> --features_file <path_to_features.pkl>

Parameters:
  --results_file: Path to the pickle file containing evaluation metrics (e.g., from SVM training).
  --features_file: Path to the pickle file containing the combined features (for visualization).
  --pca_components: Number of PCA components for visualization (default: 3).
  --mode: Evaluation mode ("single" for subject-level or "pooled" for combined subjects).  
"""

import argparse
import pickle
from metric import aggregate_results  # module for quantitative metrics aggregation
from visual import plot_tsne, plot_pca  # module for qualitative plots

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def print_individual_results(results):
    print("Individual Subject Results:")
    for subj, metrics in sorted(results.items(), key=lambda x: int(x[0])):
        print(f"Subject {subj}: Accuracy = {metrics['accuracy']:.3f}, Kappa = {metrics['kappa']:.3f}")
        print("Confusion Matrix:")
        print(metrics["confusion_matrix"])
        print("")

def run_visual_inspection(features_file, pca_components):
    features_dict = load_pickle(features_file)
    print("Qualitative evaluation (TSNE and PCA) on the feature representations:")
    for subj, sessions in sorted(features_dict.items(), key=lambda x: int(x[0])):
        for sess, data in sessions.items():
            if 'combined' not in data or 'labels' not in data:
                continue
            X = data['combined']
            y = data['labels']
            title = f"Subject {subj} - Session {sess} - Combined Features"
            tsne_filename = f"tsne_subj{subj}_{sess}_combined.png"
            pca_filename = f"pca_subj{subj}_{sess}_combined.png"
            plot_tsne(X, y, title, tsne_filename)
            plot_pca(X, y, title, pca_filename, n_components=pca_components)

def main():
    parser = argparse.ArgumentParser(description="Final Evaluation Script")
    parser.add_argument("--results_file", type=str, default="svm_results.pkl",
                        help="Path to the pickle file with evaluation metrics")
    parser.add_argument("--features_file", type=str, default="outputs/combined_features.pkl",
                        help="Path to the pickle file with combined features")
    parser.add_argument("--pca_components", type=int, default=3,
                        help="Number of PCA components for qualitative visualization (2 or 3)")
    parser.add_argument("--mode", type=str, default="single",
                        help="Evaluation mode: 'single' for subject-level or 'pooled' for combined subjects")
    
    args = parser.parse_args()
    
    # Load and print quantitative evaluation results
    results = load_pickle(args.results_file)
    print_individual_results(results)
    
    # If you have multiple runs stored in a dict (e.g., {run1: {...}, run2: {...}}), you can aggregate:
    # For example:
    # aggregate_results(results, mode=args.mode)
    
    # Run qualitative visualizations on the feature representations
    run_visual_inspection(args.features_file, args.pca_components)
    
    print("Final evaluation complete.")

if __name__ == "__main__":
    main()
