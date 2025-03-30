#!/usr/bin/env python3
"""
check_mi_onset.py

This script loads one subject’s raw EEG data from the BNCI2014001 (BCI‐IV 2a) dataset
using MOABB, extracts event annotations, and creates two plots:
  1. A raw-data plot with event markers overlaid.
  2. An evoked (average epoch) plot from epochs created with a wide window (e.g. from –1 to 8 s).

You can inspect these plots (saved as "raw_with_events.png" and "evoked_epochs.png")
to see if the event marker (time=0) corresponds to the actual MI onset or to the cue onset.
If the large evoked deflection (e.g. cue response) appears outside your expected MI period,
you may need to adjust your epoching window (or shift the event markers).
"""

import mne
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014001

def main():
    # ---------------------------
    # 1) Load Raw Data via MOABB
    # ---------------------------
    print("Loading BNCI2014001 dataset via MOABB ...")
    dataset = BNCI2014001()
    all_data = dataset.get_data()  # Returns a dictionary with subjects as keys
    subject = 1  # Change this to the desired subject number

    if subject not in all_data:
        raise ValueError(f"Subject {subject} not found in dataset.")
    
    subj_data = all_data[subject]
    
    # Use the training session ("0train"). If there are multiple runs, concatenate them.
    if isinstance(subj_data.get("0train"), dict):
        raw = mne.concatenate_raws(list(subj_data["0train"].values()))
    else:
        raw = subj_data["0train"]
    
    print(f"Loaded raw data for subject {subject}.")
    print("Channel names:", raw.ch_names)
    
    # ----------------------------------
    # 2) Extract Events from Annotations
    # ----------------------------------
    events, event_id = mne.events_from_annotations(raw, verbose=True)
    print("Extracted events (first 10 events):")
    print(events[:10])
    print("Event ID mapping from annotations:", event_id)
    
    # For BNCI2014001 the events (annotations) usually map to:
    #   'left_hand': 1, 'right_hand': 2, 'feet': 3, 'tongue': 4
    # We define our own mapping accordingly:
    my_event_dict = {'left_hand': 1, 'right_hand': 2, 'feet': 3, 'tongue': 4}
    
    # ----------------------------------
    # 3) Plot Raw Data with Event Markers and Save
    # ----------------------------------
    # Plot raw data segment (10 s) with events overlaid.
    # Note: We use mne.viz.plot_events() to get a non-interactive plot.
    fig_events = mne.viz.plot_events(events, event_id=my_event_dict, sfreq=raw.info['sfreq'])
    plt.savefig("raw_events.png")
    plt.close(fig_events)
    print("Saved events plot to 'raw_events.png'")
    
    # Alternatively, overlay events on a raw-data plot:
    fig_raw = raw.plot(n_channels=10, duration=10, show=False)
    # Save the raw data plot (this will not be interactive)
    fig_raw.savefig("raw_with_events.png")
    plt.close(fig_raw)
    print("Saved raw data plot with events to 'raw_with_events.png'")
    
    # ----------------------------------
    # 4) Create Epochs with a Wide Window to Inspect Timing
    # ----------------------------------
    # Here we create epochs from -1 s to 8 s relative to the event marker.
    # This wide window lets you see activity before and after the event.
    tmin, tmax = -1.0, 8.0
    epochs = mne.Epochs(raw, events, event_id=my_event_dict, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose=True)
    print(f"Created epochs with tmin={tmin} s and tmax={tmax} s. Number of epochs: {len(epochs)}")
    
    # ----------------------------------
    # 5) Plot the Average (Evoked) Response and Save
    # ----------------------------------
    evoked = epochs.average()
    fig_evoked = evoked.plot(show=False, titles="Average Epoch (Evoked)")
    # Save the evoked plot so you can inspect the timing of the cue/MI response.
    fig_evoked.savefig("evoked_epochs.png")
    plt.close(fig_evoked)
    print("Saved evoked (average epoch) plot to 'evoked_epochs.png'")
    
if __name__ == "__main__":
    main()
