#!/usr/bin/env python
"""
inspect_subepochs.py

This script loads preprocessed (cropped) epochs from a pickle file (e.g. preprocessed_data.pkl),
selects a specific subject and session, and then plots the time courses of a few channels for several
sub-epochs to visually inspect that the sliding-window cropping is working as intended.
"""

import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt

def load_preprocessed_data(path):
    """Load the preprocessed_data.pkl file."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_subepoch_channels(epochs, subepoch_indices, channel_indices):
    """
    Plots selected channels over time for given sub-epoch indices.
    
    Parameters
    ----------
    epochs : mne.EpochsArray
        The cropped epochs (sub-epochs).
    subepoch_indices : list of int
        The indices of sub-epochs to inspect.
    channel_indices : list of int
        The indices of channels to plot.
    """
    # Get data from epochs: shape (n_subepochs, n_channels, n_times)
    data = epochs.get_data()
    n_subepochs, n_channels, input_window_samples = data.shape
    
    # Create a time axis using tmin and tmax from the epochs (assumed to start at 0)
    time_axis = np.linspace(epochs.tmin, epochs.tmax, input_window_samples)
    
    for idx in subepoch_indices:
        if idx >= n_subepochs:
            print(f"Sub-epoch index {idx} is out of range (only {n_subepochs} sub-epochs available).")
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        for ch in channel_indices:
            if ch >= n_channels:
                print(f"Channel index {ch} is out of range (only {n_channels} channels available).")
                continue
            ax.plot(time_axis, data[idx, ch, :], label=f"Channel {ch}")
        ax.set_title(f"Sub-epoch {idx}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        plt.tight_layout()
        plt.show()

def main():
    # Update this path to where your preprocessed_data.pkl is saved.
    preproc_path = "./outputs/preprocessed_data.pkl"
    
    # Load preprocessed data (expected structure: {subject: {'0train': epochs, '1test': epochs}})
    preprocessed_data = load_preprocessed_data(preproc_path)
    
    # Print available subject keys to help debug
    print("Available subject keys:", list(preprocessed_data.keys()))
    
    # Choose a subject and session to inspect.
    # Use an integer subject ID if your keys are integers.
    subject_id = 1  
    # If keys are strings, you might need to convert, e.g., str(subject_id)
    if subject_id not in preprocessed_data:
        if str(subject_id) in preprocessed_data:
            subject_id = str(subject_id)
        else:
            print(f"Subject {subject_id} not found in preprocessed data.")
            return

    session_key = '0train'
    
    if session_key not in preprocessed_data[subject_id]:
        print(f"Session {session_key} not found for subject {subject_id}.")
        return
    
    epochs = preprocessed_data[subject_id][session_key]
    print(f"Loaded epochs for subject {subject_id} session {session_key} with {len(epochs.events)} sub-epochs.")
    
    # Define which sub-epochs and channels to inspect
    subepoch_indices = [0, 10, 20, 30, 40]  # Adjust as needed
    channel_indices = [0, 1, 2]  # For example, first three channels
    
    plot_subepoch_channels(epochs, subepoch_indices, channel_indices)

if __name__ == "__main__":
    main()
