#!/usr/bin/env python
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def compute_channel_stats(epochs_obj):
    """
    Compute summary statistics (mean, standard deviation, min, max)
    for the data contained in an MNE Epochs object.
    
    If the input is a dictionary (with key 'combined'), it assumes that
    the value is already a NumPy array.
    """
    # If epochs_obj is an MNE Epochs object, call get_data()
    if hasattr(epochs_obj, 'get_data'):
        data = epochs_obj.get_data()  # shape: (n_epochs, n_channels, n_times)
    # If it's a dict, try to get the 'combined' key (adjust if your structure is different)
    elif isinstance(epochs_obj, dict) and 'combined' in epochs_obj:
        data = epochs_obj['combined']  # assumed to be 2D: (n_trials, n_features)
    else:
        raise ValueError("Input object does not have get_data() and is not a dict with 'combined' key.")

    # For simplicity, we compute stats across epochs and time, channel by channel
    # If data is 3D, average over epochs and times for each channel
    if data.ndim == 3:
        # Shape: (n_epochs, n_channels, n_times)
        channel_means = data.mean(axis=(0, 2))
        channel_stds  = data.std(axis=(0, 2))
        channel_min   = data.min(axis=(0, 2))
        channel_max   = data.max(axis=(0, 2))
    elif data.ndim == 2:
        # If data is already 2D (e.g., after feature extraction), treat columns as features.
        channel_means = data.mean(axis=0)
        channel_stds  = data.std(axis=0)
        channel_min   = data.min(axis=0)
        channel_max   = data.max(axis=0)
    else:
        raise ValueError("Unexpected data shape.")

    stats = {
        'mean': np.mean(channel_means),
        'std': np.mean(channel_stds),
        'min': np.min(channel_min),
        'max': np.max(channel_max)
    }
    return stats

def plot_histogram_box(data, subj, sess_label, output_dir="."):
    """
    Plot histogram and box plot for the provided data (flattened) and save the figures.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Flatten the data
    flat_data = data.flatten()
    
    # Histogram
    plt.figure(figsize=(8, 4))
    plt.hist(flat_data, bins=50, color='skyblue', edgecolor='black')
    plt.title(f"Histogram - Subject {subj}, Session {sess_label}")
    hist_path = os.path.join(output_dir, f"features_subj{subj}_{sess_label}_histogram.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"Histogram saved as {hist_path}")
    
    # Box plot
    plt.figure(figsize=(6, 4))
    plt.boxplot(flat_data, vert=False)
    plt.title(f"Box Plot - Subject {subj}, Session {sess_label}")
    box_path = os.path.join(output_dir, f"features_subj{subj}_{sess_label}_boxplot.png")
    plt.savefig(box_path)
    plt.close()
    print(f"Box plot saved as {box_path}")

def main():
    features_file = "outputs/22ica/22ica_features.pkl"  # Change the path as needed
    with open(features_file, "rb") as f:
        data_dict = pickle.load(f)
    
    # data_dict is assumed to be structured as: {subject: {session: epochs_obj}}
    output_dir = "scaling_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    for subj, sessions in data_dict.items():
        for sess_label, epochs_obj in sessions.items():
            print(f"--- Subject: {subj}, Session: {sess_label} ---")
            try:
                stats = compute_channel_stats(epochs_obj)
                print("Overall feature statistics (averaged over dimensions):")
                print(f"   Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}, Min: {stats['min']:.3f}, Max: {stats['max']:.3f}")
                
                # If epochs_obj is an MNE Epochs object, use get_data(); if dict, use its 'combined' value.
                if hasattr(epochs_obj, 'get_data'):
                    data = epochs_obj.get_data()
                elif isinstance(epochs_obj, dict) and 'combined' in epochs_obj:
                    data = epochs_obj['combined']
                else:
                    continue
                
                # Plot histogram and box plot
                plot_histogram_box(data, subj, sess_label, output_dir=output_dir)
            except Exception as e:
                print(f"Error processing subject {subj}, session {sess_label}: {e}")

if __name__ == "__main__":
    main()
