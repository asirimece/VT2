#!/usr/bin/env python
"""
svm_classifier.py

This script loads the combined features from the pickle file (features.pkl), 
trains a linear SVM classifier for each subject using the training session ("0train") 
and tests on the test session ("1test"). It computes quantitative metrics including 
accuracy, Cohen's kappa, and confusion matrices, then prints and saves the results.
"""

import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import argparse

def load_features(features_file):
    """Load the combined features dictionary from a pickle file."""
    with open(features_file, 'rb') as f:
        features_dict = pickle.load(f)
    return features_dict

def train_evaluate_svm(features, session_train="0train", session_test="1test"):
    """
    For each subject, train a linear SVM on training session features and evaluate on test session features.
    
    The features dictionary should be structured as:
      { subject: { session: {'combined': X, 'labels': y} } }
    
    Returns:
      results: dict of performance metrics for each subject.
    """
    results = {}
    # Sort subjects numerically (assuming subject keys are numbers or strings convertible to int)
    subjects = sorted(features.keys(), key=lambda x: int(x))
    
    for subj in subjects:
        subj_data = features[subj]
        if session_train not in subj_data or session_test not in subj_data:
            print(f"Subject {subj}: Missing training or testing session data.")
            continue
        
        # Extract training and test features and labels
        X_train = subj_data[session_train]['combined']
        y_train = subj_data[session_train]['labels']
        X_test = subj_data[session_test]['combined']
        y_test = subj_data[session_test]['labels']
        
        # Train a linear SVM classifier
        clf = SVC(kernel='linear', random_state=42)
        clf.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = clf.predict(X_test)
        
        # Compute evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)
        
        results[subj] = {
            "accuracy": acc,
            "kappa": kappa,
            "confusion_matrix": conf_mat
        }
        
        print(f"Subject {subj}: Accuracy = {acc:.3f}, Kappa = {kappa:.3f}")
        print(f"Confusion Matrix:\n{conf_mat}\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="SVM Classifier for Combined EEG Features")
    parser.add_argument("--features_file", type=str, default="outputs/robustscaler/robustscaler_features.pkl",
                        help="Path to the combined features pickle file")
    args = parser.parse_args()
    
    # Load features from pickle file
    features = load_features(args.features_file)
    
    # Train and evaluate the SVM classifier for each subject
    results = train_evaluate_svm(features, session_train="0train", session_test="1test")
    
    # Optionally, save the results to a pickle file
    results_file = "svm_results.pkl"
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
    print(f"SVM classification complete. Results saved to {results_file}")

if __name__ == "__main__":
    main()
