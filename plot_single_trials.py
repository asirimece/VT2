#!/usr/bin/env python3
"""
Script 2: plot_single_trials.py

This script loads raw EEG data, creates macro epochs using tmin=2.0 and tmax=6.0,
and saves plots of a few individual epochs. This allows you to visually inspect that
the epoch window (2–6 s) captures the relevant MI period.
Expected:
    - The printed shape of the epochs should indicate 4 s epochs.
    - Saved epoch plots (e.g., "epoch_trial_0.png", etc.) should show EEG activity from 2 to 6 s.
"""

import mne
import matplotlib.pyplot as plt

def main():
    # Update this path with your raw file location.
    raw_fname = "./vt2/data/bci_iv2a/A01T.gdf"
    raw = mne.io.read_raw_gdf(raw_fname, preload=True)
    
    # Define event mapping as per your dataset documentation.
    event_id = {'feet': 1, 'left_hand': 2, 'right_hand': 3, 'tongue': 4}
    events, _ = mne.events_from_annotations(raw)
    
    # Create macro epochs from 2.0 to 6.0 s
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=2.0, tmax=6.0, baseline=None, preload=True)
    print("Created epochs with shape:", epochs.get_data().shape)
    
    # Save plots for the first 5 epochs
    for i in range(5):
        fig = epochs[i].plot(show=False)
        out_fname = f"epoch_trial_{i}.png"
        fig.savefig(out_fname, dpi=150)
        plt.close(fig)
        print(f"Saved epoch plot for trial {i} to '{out_fname}'.")
    
if __name__ == "__main__":
    main()
