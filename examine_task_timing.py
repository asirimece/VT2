#!/usr/bin/env python
"""
examine_task_timing.py

This script loads raw EEG data from a GDF file (e.g., from BNCI2014_001),
extracts epochs in the window from -0.5 s (pre-cue baseline) to 6.0 s (post-cue),
and produces two sets of plots for each condition:
  1. The average ERP (time–domain) signal.
  2. A time–frequency representation (using Morlet wavelets).

These plots help verify that motor imagery signals (e.g., event-related desynchronization or
spectral modulations) are present in the chosen epoch window.

Note: We use event_repeated='merge' to handle non-unique event time samples.
"""

import os
import mne
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
# Define the epoch window relative to the cue: pre-cue baseline (-0.5 s) and post-cue until 6.0 s.
EPOCH_TMIN = -0.5    # seconds: start 500 ms before the cue
EPOCH_TMAX = 4.5     # seconds: end at 6.0 s after the cue (total duration = 6.5 s)

# Parameters for time-frequency analysis (Morlet wavelets)
FREQS = np.arange(8, 30, 1)       # frequencies from 8 Hz to 29 Hz (alpha-beta band)
N_CYCLES = FREQS / 2.0            # number of cycles for each frequency
TFR_DECIM = 3                   # decimation factor to speed up plotting

# Path to the raw GDF file (update as needed)
RAW_FNAME = './vt2/data/bci_iv2a/A01E.gdf'

# Directory where plots will be saved
PLOT_DIR = 'inspection_plots'
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

def extract_epochs(raw, tmin, tmax):
    """
    Extract epochs from raw data using its annotations.
    Uses event_repeated='merge' to handle non-unique event time samples.
    """
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose=False,
                        event_repeated='merge')
    return epochs, event_id

def plot_average_erp(epochs, condition, subject_id):
    """
    Plot the average ERP (time-domain) for a given condition.
    If there are no epochs for the condition, print a message and skip plotting.
    """
    # Check if there are any epochs for the condition
    cond_epochs = epochs[condition]
    if len(cond_epochs) == 0:
        print(f"Condition {condition} has no epochs; skipping ERP plot.")
        return
    # Compute the evoked response
    evoked = cond_epochs.average()
    # Plot the evoked response (no 'return_fig' argument)
    evoked.plot(show=False)
    # Grab the current figure, add a title, and save it
    fig = plt.gcf()
    fig.suptitle(f'Subject {subject_id} - Condition {condition} ERP ({EPOCH_TMIN} to {EPOCH_TMAX} s)')
    fname = os.path.join(PLOT_DIR, f'subject_{subject_id}_{condition}_ERP.png')
    fig.savefig(fname)
    plt.close(fig)

def plot_tfr(epochs, condition, subject_id):
    """
    Compute and plot a time-frequency representation (using Morlet wavelets)
    for a given condition. If no epochs exist, skip plotting.
    """
    cond_epochs = epochs[condition]
    if len(cond_epochs) == 0:
        print(f"Condition {condition} has no epochs; skipping TFR plot.")
        return
    # Use the legacy tfr_morlet (or update to compute_tfr if preferred)
    power = mne.time_frequency.tfr_morlet(cond_epochs, freqs=FREQS,
                                          n_cycles=N_CYCLES, use_fft=True,
                                          return_itc=False, decim=TFR_DECIM,
                                          n_jobs=1, verbose=False)
    # Plot TFR for the first channel and add a title if possible
    power.plot([0], baseline=None, title=f'Subject {subject_id} - Condition {condition} TFR ({EPOCH_TMIN} to {EPOCH_TMAX} s)', show=False)
    fig = plt.gcf()
    fname = os.path.join(PLOT_DIR, f'subject_{subject_id}_{condition}_TFR.png')
    fig.savefig(fname)
    plt.close(fig)

def main():
    # --- Load raw data ---
    try:
        # For GDF files, use read_raw_gdf
        raw = mne.io.read_raw_gdf(RAW_FNAME, preload=True, verbose=False)
    except Exception as e:
        print("Error loading raw data. Please check RAW_FNAME and file format.")
        raise e

    # Optionally, set the montage (if required):
    # montage = mne.channels.make_standard_montage('standard_1020')
    # raw.set_montage(montage)

    # --- Extract epochs ---
    epochs, event_id = extract_epochs(raw, EPOCH_TMIN, EPOCH_TMAX)
    print(f"Extracted {len(epochs)} epochs in the {EPOCH_TMIN} to {EPOCH_TMAX} s window.")

    # List condition names from event_id dictionary
    conditions = list(event_id.keys())
    subject_id = 1  # Modify as needed

    # --- Plot average ERP and TFR for each condition ---
    for cond in conditions:
        print(f"Plotting condition: {cond}")
        plot_average_erp(epochs, cond, subject_id)
        plot_tfr(epochs, cond, subject_id)

    print(f"All plots saved in the folder: {PLOT_DIR}")

if __name__ == '__main__':
    main()
