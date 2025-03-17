#!/usr/bin/env python
"""
crossval_dim_reduction.py

This script loads the combined features (from the training session, e.g., "0train")
from a pickle file and then performs cross‑validation using a linear SVM classifier.
It runs two evaluations:
  1. Without additional dimensionality reduction.
  2. With PCA applied (to retain a specified amount of variance).
  
It computes the accuracy and Cohen’s kappa (using StratifiedKFold) for each subject,
and then prints the per‑subject and overall mean and standard deviation.
This will help decide whether applying PCA (i.e., reducing dimension further) improves stability
and overall performance before you proceed with clustering.
"""

import pickle
import numpy as np
import argparse

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, accuracy_score, cohen_kappa_score

def load_features(features_file):
    """Load the combined features dictionary from a pickle file."""
    with open(features_file, "rb") as f:
        features_dict = pickle.load(f)
    return features_dict

def evaluate_subject(X, y, use_pca=False, explained_variance=0.95, cv_folds=5, random_state=42):
    """
    Evaluate a classifier on subject's data using cross-validation.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The feature matrix.
    y : array-like, shape (n_samples,)
        The labels.
    use_pca : bool
        If True, apply PCA to reduce dimensionality before classification.
    explained_variance : float
        The fraction of variance to retain (if use_pca is True).
    cv_folds : int
        Number of cross-validation folds.
    random_state : int
        Random seed.
    
    Returns
    -------
    cv_results : dict
        Dictionary with test scores for 'accuracy' and 'kappa' over folds.
    """
    # Optionally apply PCA
    if use_pca:
        pca = PCA(n_components=explained_variance, random_state=random_state)
        X = pca.fit_transform(X)
    
    clf = SVC(kernel="linear", random_state=random_state)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Create a custom scorer for Cohen's kappa
    kappa_scorer = make_scorer(cohen_kappa_score)
    scoring = {"accuracy": "accuracy", "kappa": kappa_scorer}
    
    cv_results = cross_validate(clf, X, y, cv=skf, scoring=scoring, return_train_score=False)
    return cv_results

def main():
    parser = argparse.ArgumentParser(description="Cross-validation to decide on applying PCA before clustering")
    parser.add_argument("--features_file", type=str, default="outputs/22ica/22ica_features.pkl",
                        help="Path to the combined features pickle file (training session)")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--explained_variance", type=float, default=0.95,
                        help="Explained variance threshold for PCA")
    args = parser.parse_args()
    
    features_dict = load_features(args.features_file)
    
    # We assume the features dictionary is structured as {subject: {"0train": {'combined': X, 'labels': y}, ...}}
    subjects = sorted(features_dict.keys(), key=lambda x: int(x))
    
    results_no_pca = {}
    results_with_pca = {}
    
    for subj in subjects:
        subj_data = features_dict[subj]
        if "0train" not in subj_data:
            print(f"Subject {subj} has no training data; skipping.")
            continue
        
        X = subj_data["0train"]["combined"]
        y = subj_data["0train"]["labels"]
        print(f"Evaluating subject {subj} with {X.shape[0]} samples and {X.shape[1]} features.")
        
        cv_results_no_pca = evaluate_subject(X, y, use_pca=False, explained_variance=args.explained_variance,
                                             cv_folds=args.cv_folds)
        cv_results_with_pca = evaluate_subject(X, y, use_pca=True, explained_variance=args.explained_variance,
                                               cv_folds=args.cv_folds)
        
        results_no_pca[subj] = cv_results_no_pca
        results_with_pca[subj] = cv_results_with_pca
        
        mean_acc_no_pca = np.mean(cv_results_no_pca["test_accuracy"])
        std_acc_no_pca = np.std(cv_results_no_pca["test_accuracy"])
        mean_kappa_no_pca = np.mean(cv_results_no_pca["test_kappa"])
        std_kappa_no_pca = np.std(cv_results_no_pca["test_kappa"])
        
        mean_acc_with_pca = np.mean(cv_results_with_pca["test_accuracy"])
        std_acc_with_pca = np.std(cv_results_with_pca["test_accuracy"])
        mean_kappa_with_pca = np.mean(cv_results_with_pca["test_kappa"])
        std_kappa_with_pca = np.std(cv_results_with_pca["test_kappa"])
        
        print(f"Subject {subj} - Without PCA: Accuracy {mean_acc_no_pca:.3f} (std {std_acc_no_pca:.3f}), Kappa {mean_kappa_no_pca:.3f} (std {std_kappa_no_pca:.3f})")
        print(f"Subject {subj} - With PCA:    Accuracy {mean_acc_with_pca:.3f} (std {std_acc_with_pca:.3f}), Kappa {mean_kappa_with_pca:.3f} (std {std_kappa_with_pca:.3f})\n")
    
    # Optionally, combine across subjects:
    all_acc_no_pca = np.concatenate([results_no_pca[subj]["test_accuracy"] for subj in subjects if subj in results_no_pca])
    all_acc_with_pca = np.concatenate([results_with_pca[subj]["test_accuracy"] for subj in subjects if subj in results_with_pca])
    all_kappa_no_pca = np.concatenate([results_no_pca[subj]["test_kappa"] for subj in subjects if subj in results_no_pca])
    all_kappa_with_pca = np.concatenate([results_with_pca[subj]["test_kappa"] for subj in subjects if subj in results_with_pca])
    
    print("Overall average performance:")
    print(f"Without PCA: Accuracy {np.mean(all_acc_no_pca):.3f} (std {np.std(all_acc_no_pca):.3f}), Kappa {np.mean(all_kappa_no_pca):.3f} (std {np.std(all_kappa_no_pca):.3f})")
    print(f"With PCA:    Accuracy {np.mean(all_acc_with_pca):.3f} (std {np.std(all_acc_with_pca):.3f}), Kappa {np.mean(all_kappa_with_pca):.3f} (std {np.std(all_kappa_with_pca):.3f})")
    
if __name__ == "__main__":
    main()

"""
Evaluating subject 1 with 288 samples and 292 features.
Subject 1 - Without PCA: Accuracy 0.694 (std 0.028), Kappa 0.593 (std 0.037)
Subject 1 - With PCA:    Accuracy 0.694 (std 0.050), Kappa 0.593 (std 0.066)

Evaluating subject 2 with 288 samples and 292 features.
Subject 2 - Without PCA: Accuracy 0.590 (std 0.063), Kappa 0.453 (std 0.084)
Subject 2 - With PCA:    Accuracy 0.569 (std 0.093), Kappa 0.425 (std 0.124)

Evaluating subject 3 with 288 samples and 292 features.
Subject 3 - Without PCA: Accuracy 0.715 (std 0.032), Kappa 0.620 (std 0.042)
Subject 3 - With PCA:    Accuracy 0.694 (std 0.029), Kappa 0.592 (std 0.039)

Evaluating subject 4 with 288 samples and 292 features.
Subject 4 - Without PCA: Accuracy 0.500 (std 0.050), Kappa 0.333 (std 0.067)
Subject 4 - With PCA:    Accuracy 0.514 (std 0.022), Kappa 0.352 (std 0.030)

Evaluating subject 5 with 288 samples and 292 features.
Subject 5 - Without PCA: Accuracy 0.538 (std 0.050), Kappa 0.384 (std 0.068)
Subject 5 - With PCA:    Accuracy 0.559 (std 0.047), Kappa 0.412 (std 0.063)

Evaluating subject 6 with 288 samples and 292 features.
Subject 6 - Without PCA: Accuracy 0.517 (std 0.084), Kappa 0.357 (std 0.111)
Subject 6 - With PCA:    Accuracy 0.528 (std 0.082), Kappa 0.371 (std 0.109)

Evaluating subject 7 with 288 samples and 292 features.
Subject 7 - Without PCA: Accuracy 0.729 (std 0.087), Kappa 0.639 (std 0.116)
Subject 7 - With PCA:    Accuracy 0.743 (std 0.069), Kappa 0.657 (std 0.093)

Evaluating subject 8 with 288 samples and 292 features.
Subject 8 - Without PCA: Accuracy 0.812 (std 0.055), Kappa 0.749 (std 0.073)
Subject 8 - With PCA:    Accuracy 0.802 (std 0.042), Kappa 0.736 (std 0.056)

Evaluating subject 9 with 288 samples and 292 features.
Subject 9 - Without PCA: Accuracy 0.628 (std 0.018), Kappa 0.505 (std 0.024)
Subject 9 - With PCA:    Accuracy 0.628 (std 0.016), Kappa 0.505 (std 0.021)

Overall average performance:
Without PCA: Accuracy 0.636 (std 0.117), Kappa 0.515 (std 0.156)
With PCA:    Accuracy 0.637 (std 0.111), Kappa 0.516 (std 0.148)
"""
