#!/usr/bin/env python
import mne
import numpy as np
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014001

def plot_and_save_raw_data(subject=1, session="0train", picks=["Fz", "FC3", "FC1", "FCz", "FC2"],
                           duration=10):
    # Initialize the dataset
    dataset = BNCI2014001()
    data = dataset.get_data(subjects=[subject])
    
    if subject not in data:
        print(f"Subject {subject} not found in dataset.")
        return
    
    if session not in data[subject]:
        print(f"Session {session} not found for subject {subject}.")
        return
    
    sess_data = data[subject][session]
    
    # Extract the Raw object from session data
    if isinstance(sess_data, dict):
        if "raw" in sess_data:
            raw = sess_data["raw"]
        else:
            raw = list(sess_data.values())[0]
    else:
        raw = sess_data

    # Check and print basic information
    try:
        sfreq = raw.info["sfreq"]
        print(f"Sampling rate: {sfreq} Hz")
    except Exception as e:
        print("Could not access raw.info:", e)
        return

    # ---- Raw Time-Series Plot ----
    picks_idx = mne.pick_channels(raw.info["ch_names"], include=picks)
    data_array = raw.get_data(picks=picks_idx)
    times = raw.times
    idx = np.where(times <= duration)[0]
    
    plt.figure(figsize=(12, 6))
    offset = 0
    for i, ch in enumerate(picks):
        plt.plot(times[idx], data_array[i, idx] + offset, label=ch)
        offset += np.ptp(data_array[i, idx]) + 10  # Add vertical offset
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV) + offset")
    plt.title(f"Subject {subject} Session {session} - Raw EEG Data")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("raw_data_plot.png", dpi=300)
    plt.close()
    print("Saved raw time-series plot as 'raw_data_plot.png'.")

    # ---- Power Spectral Density (PSD) Plot ----
    # Compute PSD using Welch's method with 'mean' averaging.
    psd_result = raw.compute_psd(method='welch', fmax=40, average='mean')
    fig_psd = psd_result.plot(show=False)
    fig_psd.savefig("raw_data_psd.png", dpi=300)
    plt.close(fig_psd)
    print("Saved PSD plot as 'raw_data_psd.png'.")

    # ---- Event Markers Plot ----
    try:
        events = mne.find_events(raw, verbose=False)
        unique_events = np.unique(events[:, 2])
        print("Unique Event Codes:", unique_events)
        print("Number of Events/Trials:", events.shape[0])
        fig_events = mne.viz.plot_events(events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, show=False)
        fig_events.savefig("raw_data_events.png", dpi=300)
        plt.close(fig_events)
        print("Saved events plot as 'raw_data_events.png'.")
    except Exception as e:
        print("Error extracting or plotting events:", e)

    # ---- Artifact Inspection via Epoch Image ----
    try:
        event_id = {'left_hand': 1, 'right_hand': 2, 'both_feet': 3, 'tongue': 4}
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0.0, tmax=3.5, preload=True)
        # Use plot_image to generate a static image of the epochs
        fig_epochs = epochs.plot_image(picks=picks, sigma=0.1, vmin=-10, vmax=10, show=False)
        fig_epochs.savefig("epochs_image.png", dpi=300)
        plt.close(fig_epochs)
        print("Saved epochs image as 'epochs_image.png'.")
    except Exception as e:
        print("Error creating or plotting epochs image:", e)

if __name__ == "__main__":
    plot_and_save_raw_data(subject=1, session="0train")
