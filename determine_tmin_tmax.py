#!/usr/bin/env python
"""
determine_tmin_tmax.py

Loads raw EEG data from the BNCI2014_001 dataset for a selected subject,
concatenates all runs from all sessions into a single Raw object, then extracts
epochs for multiple (tmin, tmax) candidates. Finally, it plots and saves average
ERP plots for each class to help determine which epoch window is best.

Usage:
    python determine_tmin_tmax.py --subject 1
"""

import argparse
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014_001

def plot_erps(epochs, event_id, title, out_filename):
    """
    Compute and plot the average ERP for each class, then save the plot.
    """
    # Create an Evoked object per class label, then plot the mean across channels
    evokeds = {key: epochs[key].average() for key in event_id}

    plt.figure(figsize=(10, 6))
    for label, ev in evokeds.items():
        # Plot the mean across channels for clarity
        plt.plot(ev.times, ev.data.mean(axis=0), label=label)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_filename)
    plt.close()
    print(f"Saved plot: {out_filename}")

def main(subject):
    # 1) Load dataset
    dataset = BNCI2014_001()
    all_data = dataset.get_data()  
    # => {subject_id: {"session_0": {"run_0": raw, ...}, "session_1": {...}}, ...}

    # 2) Check if subject is present
    if subject not in all_data:
        raise ValueError(f"Subject {subject} not found in dataset.")
    
    # 3) Flatten the runs from all sessions into a single list of Raw objects
    subject_data = all_data[subject]  # e.g. {"session_0": {...}, "session_1": {...}}
    run_list = []
    for session_name, runs_dict in subject_data.items():
        for run_name, raw_obj in runs_dict.items():
            run_list.append(raw_obj)

    # 4) Concatenate all runs into one Raw
    if len(run_list) == 0:
        raise ValueError(f"No runs found for subject {subject}.")
    raw = mne.concatenate_raws(run_list)
    print(f"Loaded subject {subject} raw data with channels:\n{raw.ch_names}")

    # 5) Create output directory for saving plots
    out_dir = "tmin_tmax_plots"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 6) Extract events and event_id from annotations
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    print("Event mapping:", event_id)

    # 7) Candidate (tmin, tmax) windows to explore
    candidate_windows = [
        (0, 4),
        (0.5, 4.5),
        (1.0, 4.5),
        (2.0, 6.0)
    ]

    # 8) For each candidate window, create epochs, then plot/save ERPs
    for (tmin, tmax) in candidate_windows:
        epochs_candidate = mne.Epochs(
            raw, events, event_id=event_id,
            tmin=tmin, tmax=tmax,
            baseline=None, preload=True, verbose=False
        )
        n_ep = len(epochs_candidate.events)
        title = f"Subject {subject} ERP - tmin={tmin}, tmax={tmax} (n={n_ep})"
        out_filename = os.path.join(out_dir, f"Subject_{subject}_ERP_tmin_{tmin}_tmax_{tmax}.png")
        print(f"Extracted {n_ep} epochs with tmin={tmin}, tmax={tmax}")
        
        if n_ep > 0:
            plot_erps(epochs_candidate, event_id, title, out_filename)
        else:
            print("No epochs found for this window—skipping plot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Determine optimal tmin and tmax for epoching.")
    parser.add_argument("--subject", type=int, default=1, help="Subject number to inspect")
    args = parser.parse_args()
    main(args.subject)
