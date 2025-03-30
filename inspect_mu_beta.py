#!/usr/bin/env python
"""
inspect_mu_beta.py

This script loads preprocessed epochs from a pickle file (e.g., preprocessed_data.pkl),
selects a given subject and session (e.g., "0train"), computes the power spectral density (PSD)
using Welch’s method for each epoch, and then averages the power in the mu (8–12 Hz)
and beta (13–30 Hz) bands for each class. The results are plotted as a bar graph and saved to disk.

Usage:
    python inspect_mu_beta.py --subject 1 --session 0train
"""

import os
import pickle
import numpy as np
import mne
import matplotlib.pyplot as plt
import argparse

def load_preprocessed_data(path):
    """Load preprocessed data from a pickle file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def get_event_mapping(epochs):
    """
    Derive an event mapping from the events array in epochs.
    Instead of creating keys like "class_0", this function returns keys that are simply the string
    representation of the numeric event labels. For example, if events are [0,1,2,3], the mapping will be:
       {"0": 0, "1": 1, "2": 2, "3": 3}
    This allows you to index the epochs using these keys.
    """
    unique_labels = np.unique(epochs.events[:, 2])
    event_id = {str(int(lbl)): int(lbl) for lbl in unique_labels}
    return event_id

def compute_band_power(epochs, band, method='welch', fmin=4, fmax=40):
    """
    Compute PSD using Welch's method and average the power in the given frequency band.
    """
    psds, freqs = epochs.compute_psd(method=method, fmin=fmin, fmax=fmax, verbose=False).get_data(return_freqs=True)
    
    print("PSD shape:", psds.shape)
    print("Frequency range:", freqs[0], "to", freqs[-1])
    print("PSD min, max, mean (all epochs):", psds.min(), psds.max(), psds.mean())
    
    band_min, band_max = band
    band_idx = np.where((freqs >= band_min) & (freqs <= band_max))[0]
    avg_power = psds[:, :, band_idx].mean(axis=-1).mean(axis=0).mean()
    return avg_power


def plot_band_power_by_class(epochs, output_dir="mu_beta_plots"):
    """
    For each class in the epochs, compute and plot the average mu and beta band power.
    The plot is saved to disk.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Derive event mapping with keys as strings of the event numbers.
    event_id = get_event_mapping(epochs)
    print("Derived event mapping:", event_id)
    
    # Sort the keys (e.g., ["0", "1", "2", "3"])
    classes = sorted(event_id.keys(), key=lambda x: int(x))
    
    mu_powers = []
    beta_powers = []
    
    for cls in classes:
        try:
            # Use the string key to select epochs.
            epochs_cls = epochs[cls]
        except Exception as e:
            print(f"Could not select epochs for class {cls}: {e}")
            continue
        
        mu_power = compute_band_power(epochs_cls, band=(8, 12))
        beta_power = compute_band_power(epochs_cls, band=(13, 30))
        mu_powers.append(mu_power)
        beta_powers.append(beta_power)
        print(f"Class {cls}: mu power = {mu_power:.4f}, beta power = {beta_power:.4f}")
    
    # Ensure that we have values for all classes before plotting.
    if len(classes) != len(mu_powers):
        print("Warning: Some classes were not selected. Skipping plot.")
        return

    x = np.arange(len(classes))
    width = 0.35
    plt.figure(figsize=(8, 6))
    plt.bar(x - width/2, mu_powers, width, label="Mu (8–12 Hz)")
    plt.bar(x + width/2, beta_powers, width, label="Beta (13–30 Hz)")
    plt.xticks(x, classes)
    plt.xlabel("Class")
    plt.ylabel("Average Power (a.u.)")
    plt.title("Average Mu and Beta Band Power by Class")
    plt.legend()
    out_filename = os.path.join(output_dir, "mu_beta_power_by_class.png")
    plt.savefig(out_filename)
    plt.close()
    print(f"Saved band power plot to: {out_filename}")

def main(subject, session):
    data_file = "./outputs/preprocessed_data.pkl"
    data = load_preprocessed_data(data_file)
    
    if subject not in data:
        raise ValueError(f"Subject {subject} not found in preprocessed data.")
    if session not in data[subject]:
        raise ValueError(f"Session {session} not found for subject {subject}.")
    
    epochs = data[subject][session]
    print(f"Subject {subject}, session {session} epochs shape: {epochs.get_data().shape}")
    
    plot_band_power_by_class(epochs, output_dir="mu_beta_plots")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect mu/beta band power differences by class.")
    parser.add_argument("--subject", type=int, default=1, help="Subject number to inspect")
    parser.add_argument("--session", type=str, default="0train", help="Session key (e.g., '0train')")
    args = parser.parse_args()
    main(subject=args.subject, session=args.session)
