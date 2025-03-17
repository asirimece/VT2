#!/usr/bin/env python
"""
plot_preprocessed.py

This script loads preprocessed epochs (the output of the preprocessing pipeline)
and generates diagnostic plots:
  - Power Spectral Density (PSD) plot
  - Raw signal plot (from one epoch or a subset)
  - Event markers plot

The plots are saved to disk for inspection.
"""

import os
import mne
import matplotlib.pyplot as plt
import pickle

def save_fig(fig, out_dir, fname):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.savefig(os.path.join(out_dir, fname))
    plt.close(fig)

def load_preprocessed_data(pickle_file):
    """
    Load preprocessed data from a pickle file.
    The data should be a dictionary with structure:
      {subject: {session: epochs, ...}, ...}
    """
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    return data

def plot_preprocessed_data(subject_epochs, subject, session, save_figures=True, out_dir="./plots"):
    """
    Generate and save plots for preprocessed epochs.
    
    Parameters:
        subject_epochs (mne.Epochs): Preprocessed epochs for a given subject and session.
        subject (int): Subject number.
        session (str): Session label.
        save_figures (bool): Whether to save the figures.
        out_dir (str): Directory where to save the figures.
    """
    # Plot the power spectral density (PSD)
    psd = subject_epochs.compute_psd(method='welch', fmax=40, average='mean')
    psd_fig = psd.plot(dB=True, show=False)
    if save_figures:
        save_fig(psd_fig, out_dir, f"psd_subject{subject}_{session}.png")
    
    # Plot a few raw epochs for inspection (non-interactive, saved to file)
    # Here we plot the first 5 epochs (adjust as needed)
    raw_fig = subject_epochs.copy().plot(n_epochs=5, n_channels=10, show=False, block=False)
    if save_figures:
        save_fig(raw_fig, out_dir, f"raw_epochs_subject{subject}_{session}.png")
    
    # Plot event markers using mne.find_events and mne.viz.plot_events.
    # For this, we need to create a figure manually since plot_events returns a list of figures.
    try:
        events = mne.find_events(subject_epochs, verbose=False)
        if len(events) > 0:
            fig_events = mne.viz.plot_events(events, subject_epochs.info, show=False)
            if save_figures:
                save_fig(fig_events, out_dir, f"events_subject{subject}_{session}.png")
        else:
            print("No events found in the preprocessed epochs.")
    except Exception as e:
        print("Error plotting events:", e)

if __name__ == "__main__":
    # Path to the pickle file containing preprocessed data.
    # This file should have been created by your pipeline (e.g., via pickle.dump(results, ...))
    pickle_file = "./outputs/preprocessed_data.pkl"
    
    # Load the preprocessed data
    data = load_preprocessed_data(pickle_file)
    
    # For demonstration, choose subject 1 and session "0train"
    subject = 1
    session = "0train"
    
    if subject in data and session in data[subject]:
        epochs = data[subject][session]
        print(f"Loaded preprocessed epochs for subject {subject}, session {session}.")
        plot_preprocessed_data(epochs, subject, session, save_figures=True, out_dir="./plots")
        print("Plots saved.")
    else:
        print(f"Data for subject {subject} session {session} not found.")
