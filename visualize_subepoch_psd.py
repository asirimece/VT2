#!/usr/bin/env python
"""
determine_tmin_tmax_save_plots.py

This script loads preprocessed data from a pickle file (preprocessed_data.pkl), selects
a specific subject and session (e.g., training data for subject 1), filters the epochs to a
given class, and then computes:
  1) A PSD using Welch’s method (averaged across sub-epochs) for a chosen channel.
  2) A time–frequency representation using Morlet wavelets on one sub-epoch.

Both plots are saved to disk.
"""

import os
import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt

def load_preprocessed_data(path):
    """Load preprocessed data from a pickle file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def main():
    # Set up output directory for plots
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load preprocessed data (structure: {subject: {'0train': epochs, '1test': epochs}})
    data_file = "./outputs/preprocessed_data.pkl"
    preprocessed_data = load_preprocessed_data(data_file)
    
    # Choose subject and session to inspect
    subject = 1
    session_key = "0train"
    
    # Check available subjects (here keys are integers)
    if subject not in preprocessed_data:
        print(f"Subject {subject} not found in preprocessed data.")
        return
    
    epochs = preprocessed_data[subject][session_key]
    print(f"Subject {subject}, session={session_key}")
    print(f"Epochs shape: {epochs.get_data().shape}")
    
    # Filter sub-epochs by class label (last column of events, 0-based)
    label_of_interest = 0  # For example, left hand (after cropping, raw 1 becomes 0)
    sub_idxs = np.where(epochs.events[:, 2] == label_of_interest)[0]
    if len(sub_idxs) == 0:
        print(f"No sub-epochs found for label={label_of_interest}")
        return
    
    # Create a subset of epochs for the given class
    class_epochs = mne.EpochsArray(
        data=epochs.get_data()[sub_idxs],
        info=epochs.info,
        events=epochs.events[sub_idxs],
        tmin=epochs.tmin,
        baseline=None
    )
    
    # --- 1) PSD using Welch's method ---
    psds_welch, freqs = class_epochs.compute_psd('welch', fmin=4, fmax=40).get_data(return_freqs=True)
    # psds_welch shape: (n_epochs, n_channels, n_freqs)
    # Average across all sub-epochs
    avg_psd = psds_welch.mean(axis=0)  # shape: (n_channels, n_freqs)
    # Choose a channel index (e.g., channel 0)
    ch_idx = 0
    plt.figure()
    plt.semilogy(freqs, avg_psd[ch_idx], label=f"Channel {ch_idx}, label={label_of_interest}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (uV^2/Hz)")
    plt.title(f"Subject {subject}, Class {label_of_interest}, Channel {ch_idx} PSD")
    plt.legend()
    psd_filename = os.path.join(output_dir, f"Subject_{subject}_Class_{label_of_interest}_Channel_{ch_idx}_PSD.png")
    plt.savefig(psd_filename)
    plt.close()
    print(f"Saved PSD plot to: {psd_filename}")
    
    # --- 2) Morlet time-frequency representation on one sub-epoch ---
    # Choose the first sub-epoch index from the filtered indices
    ep_idx = sub_idxs[0]
    single_data = epochs.get_data()[ep_idx:ep_idx+1]  # shape: (1, n_channels, n_times)
    single_epochs = mne.EpochsArray(
        single_data, epochs.info,
        events=epochs.events[ep_idx:ep_idx+1],
        tmin=epochs.tmin,
        baseline=None
    )
    # Define frequency range and cycles for Morlet wavelets
    freq_range = np.arange(8, 31, 2)
    n_cycles = freq_range / 2.0
    power = mne.time_frequency.tfr_morlet(
        single_epochs,
        freqs=freq_range,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        verbose=False
    )
    # Save the TFR plot instead of showing it interactively.
    tfr_fig = power.plot([ch_idx], baseline=None, mode='logratio',
                          title=f"Subject {subject}, Class {label_of_interest}, Sub-epoch {ep_idx} TFR",
                          show=False)
    # tfr_fig might be a list or a single figure. Handle both cases.
    if isinstance(tfr_fig, list):
        for i, fig in enumerate(tfr_fig):
            tfr_filename = os.path.join(output_dir, f"Subject_{subject}_Class_{label_of_interest}_Subepoch_{ep_idx}_TFR_page{i}.png")
            fig.savefig(tfr_filename)
            plt.close(fig)
            print(f"Saved TFR plot page {i} to: {tfr_filename}")
    else:
        tfr_filename = os.path.join(output_dir, f"Subject_{subject}_Class_{label_of_interest}_Subepoch_{ep_idx}_TFR.png")
        tfr_fig.savefig(tfr_filename)
        plt.close(tfr_fig)
        print(f"Saved TFR plot to: {tfr_filename}")

if __name__ == "__main__":
    main()
