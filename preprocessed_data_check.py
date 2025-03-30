#!/usr/bin/env python
"""
inspect_preprocessing.py

This script inspects the preprocessed EEG epochs to verify that:
1. Data/Label assignment is correct.
2. The features (i.e. raw time‐series) show some degree of class discrimination.
3. There is no unexpected class imbalance or mislabeling.

It prints data shapes and label distributions, plots the average time‐series per class,
performs a PCA on flattened data to visualize potential clustering, and computes frequency spectra.
All plots are saved to an output folder.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import welch

def inspect_preprocessed_data(preprocessed_data_file, output_dir="inspection_plots"):
    # Load the preprocessed data (expected to be a dict: { subject: { "0train": epochs, "1test": epochs } })
    with open(preprocessed_data_file, "rb") as f:
        preprocessed_data = pickle.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for subj, sessions in preprocessed_data.items():
        print(f"\n=== Subject {subj} ===")
        if "0train" not in sessions or "1test" not in sessions:
            print("Missing training or test session for this subject.")
            continue

        # Get data from epochs (assumes that each epoch object has a get_data() method and events attribute)
        train_epochs = sessions["0train"]
        test_epochs = sessions["1test"]
        
        # Get the raw numpy data and labels
        X_train = train_epochs.get_data()  # shape: (n_trials, n_channels, n_times)
        y_train_raw = train_epochs.events[:, -1]  # raw labels (typically 1-4)
        y_train = y_train_raw - 1  # convert to 0–3 if needed
        
        X_test = test_epochs.get_data()
        y_test_raw = test_epochs.events[:, -1]
        y_test = y_test_raw - 1
        
        # Print shapes and unique labels
        print("Training data shape:", X_train.shape)
        print("Test data shape:", X_test.shape)
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        print("Unique training labels (raw):", np.unique(y_train_raw))
        print("Unique training labels (converted):", dict(zip(unique_train, counts_train)))
        print("Unique test labels (raw):", np.unique(y_test_raw))
        print("Unique test labels (converted):", dict(zip(unique_test, counts_test)))
        
        # Check that the number of channels matches your expected value (e.g., 22 EEG channels)
        expected_channels = 22
        if X_train.shape[1] != expected_channels or X_test.shape[1] != expected_channels:
            print(f"Warning: Expected {expected_channels} channels but found {X_train.shape[1]} (train) or {X_test.shape[1]} (test).")
        
        # Plot average time-series per class (for training data)
        for label in unique_train:
            class_indices = np.where(y_train == label)[0]
            if len(class_indices) == 0:
                continue
            avg_signal = np.mean(X_train[class_indices], axis=0)  # shape: (n_channels, n_times)
            plt.figure(figsize=(10, 5))
            for ch in range(avg_signal.shape[0]):
                plt.plot(avg_signal[ch, :], label=f"Ch {ch+1}", alpha=0.7)
            plt.title(f"Subject {subj} - Average Signal for Class {label}")
            plt.xlabel("Time (samples)")
            plt.ylabel("Amplitude")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            filename = os.path.join(output_dir, f"subject_{subj}_class_{label}_avg_signal.png")
            plt.savefig(filename)
            plt.close()
            print(f"Saved average signal plot for subject {subj}, class {label} to {filename}")
        
        # Perform PCA on flattened training data to inspect clustering
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_train_flat)
        plt.figure(figsize=(8, 6))
        for label in unique_train:
            indices = np.where(y_train == label)[0]
            plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=f"Class {label}", alpha=0.6)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"Subject {subj} - PCA of Training Data")
        plt.legend()
        filename = os.path.join(output_dir, f"subject_{subj}_pca.png")
        plt.savefig(filename)
        plt.close()
        print(f"Saved PCA plot for subject {subj} to {filename}")
        
        # Optionally, inspect frequency content using Welch's method for one trial per class
        fs = train_epochs.info["sfreq"] if hasattr(train_epochs, "info") and "sfreq" in train_epochs.info else 250.0
        for label in unique_train:
            # take the first trial for this label
            idx = np.where(y_train == label)[0][0]
            trial_data = X_train[idx]  # shape: (n_channels, n_times)
            plt.figure(figsize=(10, 5))
            for ch in range(trial_data.shape[0]):
                f, Pxx = welch(trial_data[ch, :], fs=fs, nperseg=256)
                plt.semilogy(f, Pxx, label=f"Ch {ch+1}", alpha=0.5)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power")
            plt.title(f"Subject {subj} - Frequency Spectrum for Class {label} (Trial {idx})")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            filename = os.path.join(output_dir, f"subject_{subj}_class_{label}_trial_{idx}_freq.png")
            plt.savefig(filename)
            plt.close()
            print(f"Saved frequency spectrum plot for subject {subj}, class {label}, trial {idx} to {filename}")

if __name__ == "__main__":
    preprocessed_data_file = "./outputs/preprocessed_data.pkl"
    print(f"Using preprocessed data file: {preprocessed_data_file}")
    inspect_preprocessed_data(preprocessed_data_file)
