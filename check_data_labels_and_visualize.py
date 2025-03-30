#!/usr/bin/env python
"""
check_data_labels_and_visualize.py

This script verifies:
1. That the epoch length (input_window_samples) is as expected.
2. That label conversion (subtracting 1) is correct.
3. Visualizes a few epochs for each class and saves the plots to a folder.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import mne

def verify_epoch_length(epochs):
    """
    Verifies that the number of time samples in each epoch is as expected.
    Prints the tmin, tmax, duration, sampling frequency, and both the expected
    and actual number of samples.
    """
    # Retrieve the time vector and sampling frequency from the epochs info
    times = epochs.times  # time axis (in seconds)
    tmin = times[0]
    tmax = times[-1]
    duration = tmax - tmin
    sfreq = epochs.info.get("sfreq", None)
    
    if sfreq is None:
        print("Sampling frequency not found in epochs.info!")
        return

    # Calculation may vary; here we assume:
    # expected_samples = duration * sfreq, plus 1 if the time vector includes both endpoints
    expected_samples = int(duration * sfreq) + 1
    actual_samples = epochs.get_data().shape[2]
    
    print(f"tmin: {tmin:.2f}s, tmax: {tmax:.2f}s, Duration: {duration:.2f}s")
    print(f"Sampling frequency: {sfreq} Hz")
    print(f"Expected number of samples: {expected_samples}")
    print(f"Actual number of samples: {actual_samples}")
    
    if actual_samples == 1000:
        print("Epoch length is 1000 samples.\n")
    else:
        print("Epoch length is NOT 1000 samples.\n")

def verify_labels(epochs):
    """
    Checks the unique event markers in the epochs and verifies the label conversion.
    Assumes that raw labels are in the last column of epochs.events and that subtracting 1
    converts them from [1,2,3,4] to [0,1,2,3].
    """
    raw_labels = epochs.events[:, -1]
    converted_labels = raw_labels - 1
    print("Unique raw labels:", np.unique(raw_labels))
    print("Unique converted labels (raw - 1):", np.unique(converted_labels))
    
    for label in np.unique(raw_labels):
        count_raw = np.sum(raw_labels == label)
        count_converted = np.sum(converted_labels == (label - 1))
        print(f"Raw label {label} appears {count_raw} times; converted label {label - 1} appears {count_converted} times.")
    print("")

def visualize_epochs(epochs, output_dir="plots", n_samples=3):
    """
    Visualizes up to n_samples epochs for each class.
    For each class, it plots the first few channels (up to 4) against time.
    The plots are saved to the specified output_dir.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert event labels (subtract 1 to match 0-indexed labels)
    labels = epochs.events[:, -1] - 1
    unique_labels = np.unique(labels)
    data = epochs.get_data()  # shape: (n_trials, n_channels, n_times)
    times = epochs.times
    
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        n_to_plot = min(n_samples, len(indices))
        for i in range(n_to_plot):
            epoch = data[indices[i]]  # shape: (n_channels, n_times)
            plt.figure(figsize=(10, 6))
            # Plot the first up to 4 channels for visualization
            n_channels_to_plot = min(4, epoch.shape[0])
            for ch in range(n_channels_to_plot):
                plt.plot(times, epoch[ch, :], label=f"Channel {ch}")
            plt.title(f"Epoch sample for class {int(label)} (Epoch index: {indices[i]})")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()
            # Save the plot to a file
            filename = os.path.join(output_dir, f"epoch_class_{int(label)}_sample_{i}.png")
            plt.savefig(filename)
            plt.close()
            print(f"Saved plot for class {int(label)} (sample {i}) to {filename}")

if __name__ == "__main__":
    # Adjust the path as needed. Here we assume the preprocessed data is saved as a pickle file.
    data_file = "./outputs/preprocessed_data.pkl"
    
    if not os.path.exists(data_file):
        print(f"Preprocessed data file not found at {data_file}")
        exit(1)
    
    # Load the preprocessed data (assuming a dict: {subject: {"0train": epochs, "1test": epochs}})
    with open(data_file, "rb") as f:
        preprocessed_data = pickle.load(f)
    
    # For demonstration, use subject '1' and the training session ("0train")
    subject = list(preprocessed_data.keys())[0]
    epochs = preprocessed_data[subject]["0train"]
    
    print("Verifying epoch length...")
    verify_epoch_length(epochs)
    
    print("Verifying labels...")
    verify_labels(epochs)
    
    print("Visualizing sample epochs and saving plots...")
    visualize_epochs(epochs, output_dir="plots", n_samples=3)
    
    print("All checks completed.")
