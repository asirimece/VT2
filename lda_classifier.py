#!/usr/bin/env python3
"""
lda_classifier.py

This script loads preprocessed sub-epoched data from ./outputs/preprocessed_data.pkl,
flattens the sub-epoch data, trains a Linear Discriminant Analysis (LDA) classifier,
and then aggregates sub-epoch predictions to trial-level predictions.
It finally prints the sub-epoch and trial-level accuracy and the confusion matrix.
"""

import os
import pickle
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

def aggregate_predictions(probabilities, trial_ids):
    """
    Given predicted class probabilities for each sub-epoch and the associated trial IDs,
    aggregate the predictions by averaging probabilities per trial.
    """
    unique_trials = np.unique(trial_ids)
    aggregated_probs = []
    for t in unique_trials:
        idx = np.where(trial_ids == t)[0]
        # Average the predicted probabilities for all sub-epochs of the trial.
        avg_prob = np.mean(probabilities[idx], axis=0)
        aggregated_probs.append(avg_prob)
    aggregated_probs = np.array(aggregated_probs)
    trial_preds = np.argmax(aggregated_probs, axis=1)
    return trial_preds, unique_trials

def main():
    data_file = "./outputs/preprocessed_data.pkl"
    with open(data_file, "rb") as f:
        preprocessed_data = pickle.load(f)

    results = {}
    for subj in sorted(preprocessed_data.keys()):
        print(f"\n=== Subject {subj} ===")
        # Load training and testing epochs (these are sub-epoched, e.g. 2 s each)
        train_ep = preprocessed_data[subj]["0train"]
        test_ep  = preprocessed_data[subj]["1test"]

        # Get data, labels and trial IDs
        X_train = train_ep.get_data()  # shape: (n_subepochs, n_channels, n_times)
        y_train = train_ep.events[:, -1]
        trial_ids_train = train_ep.events[:, 1]

        X_test = test_ep.get_data()
        y_test = test_ep.events[:, -1]
        trial_ids_test = test_ep.events[:, 1]

        # Flatten each sub-epoch into a 1D feature vector.
        n_train, n_channels, n_times = X_train.shape
        X_train_flat = X_train.reshape(n_train, n_channels * n_times)
        n_test = X_test.shape[0]
        X_test_flat = X_test.reshape(n_test, n_channels * n_times)

        # Train LDA on sub-epochs
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train_flat, y_train)

        train_preds = lda.predict(X_train_flat)
        test_preds = lda.predict(X_test_flat)

        sub_train_acc = accuracy_score(y_train, train_preds)
        sub_test_acc = accuracy_score(y_test, test_preds)
        sub_kappa = cohen_kappa_score(y_test, test_preds)
        sub_cm = confusion_matrix(y_test, test_preds)
        print(f"Sub-epoch Train Accuracy: {sub_train_acc:.4f}")
        print(f"Sub-epoch Test Accuracy: {sub_test_acc:.4f}")
        print("Sub-epoch Confusion Matrix:\n", sub_cm)

        # Get predicted probabilities to aggregate predictions to the trial level.
        test_probs = lda.predict_proba(X_test_flat)
        trial_preds, unique_trials = aggregate_predictions(test_probs, trial_ids_test)

        # For trial-level ground truth, assume each trial's label is the label
        # of its first sub-epoch.
        trial_labels = []
        for t in unique_trials:
            idx = np.where(trial_ids_test == t)[0]
            trial_labels.append(y_test[idx[0]])
        trial_labels = np.array(trial_labels)

        trial_acc = accuracy_score(trial_labels, trial_preds)
        trial_kappa = cohen_kappa_score(trial_labels, trial_preds)
        trial_cm = confusion_matrix(trial_labels, trial_preds)
        print(f"Trial-level Test Accuracy: {trial_acc:.4f}, Kappa: {trial_kappa:.4f}")
        print("Trial-level Confusion Matrix:\n", trial_cm)

        results[subj] = {
            "sub_epoch_test_acc": sub_test_acc,
            "trial_level_test_acc": trial_acc,
            "trial_level_kappa": trial_kappa
        }

    print("\nFinal results per subject:", results)

if __name__ == "__main__":
    main()
