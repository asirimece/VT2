#!/usr/bin/env python
"""
verify_epochs.py

This script loads the preprocessed data (assumed to be saved in a pickle file),
and then checks for each subject in the training session ("0train"):
  - The shape of the epoch data.
    (We expect the shape to be (n_epochs, n_channels, 500) if the crops are 2 s long at 250 Hz.)
  - The events array: the second column should contain the original trial IDs,
    and the third column should contain class labels in {0, 1, 2, 3}.
"""

import os
import pickle
import numpy as np

def check_epochs(preprocessed_file, subject):
    with open(preprocessed_file, "rb") as f:
        preprocessed_data = pickle.load(f)

    if subject not in preprocessed_data:
        print(f"Subject {subject} not found in preprocessed data.")
        return

    # Get the training session epochs for the subject.
    epochs = preprocessed_data[subject]["0train"]
    data = epochs.get_data()  # Expected shape: (n_epochs, n_channels, n_samples)
    print(f"\nSubject {subject} - '0train' epoch data shape: {data.shape}")

    # Check that each epoch has 500 samples.
    n_samples = data.shape[-1]
    if n_samples == 500:
        print("✓ Each epoch has 500 samples (2 s at 250 Hz).")
    else:
        print(f"✗ Each epoch has {n_samples} samples; expected 500. Check your epoching parameters (crop_window_length, etc.).")

    # Now check the events array.
    events = epochs.events  # Expected shape: (n_epochs, 3)
    print(f"Events shape: {events.shape}")

    # Column meanings: [sample_index, trial_id, event_code]
    unique_trial_ids = np.unique(events[:, 1])
    unique_class_labels = np.unique(events[:, 2])
    print(f"Unique trial IDs (middle column): {unique_trial_ids}")
    print(f"Unique class labels (last column): {unique_class_labels}")

    # Check that the class labels are in {0,1,2,3}
    if set(unique_class_labels) == {0, 1, 2, 3}:
        print("✓ Class labels are correctly encoded as {0, 1, 2, 3}.")
    else:
        print("✗ Class labels are not as expected. Please check your labeling logic.")

def main():
    # Modify the path to your preprocessed data file.
    preprocessed_file = "./outputs/preprocessed_data.pkl"
    
    # If you want to check multiple subjects, list them here.
    subjects = list()  # e.g., subjects = ["subj1", "subj2", "subj3"]
    # If the preprocessed data is keyed by subject names, you can list them.
    # For demonstration, we'll iterate over all keys in the pickle file.
    with open(preprocessed_file, "rb") as f:
        preprocessed_data = pickle.load(f)
    subjects = list(preprocessed_data.keys())
    
    print("Verifying epochs in the training session ('0train') for each subject:")
    for subj in subjects:
        check_epochs(preprocessed_file, subj)

if __name__ == "__main__":
    main()
