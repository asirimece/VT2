#!/usr/bin/env python
import mne
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014001

def main():
    # Initialize the dataset and load data for subject 1
    dataset = BNCI2014001()
    subject = 1
    data = dataset.get_data(subjects=[subject])
    
    # Pick a session key (e.g., "0train" for training)
    subj_data = data[subject]
    session_key = "0train"
    raw_data = subj_data[session_key]
    
    # If multiple runs exist, pick the first run (or concatenate runs as needed)
    if isinstance(raw_data, dict):
        raw = list(raw_data.values())[0]
    else:
        raw = raw_data

    print(f"Loaded raw data for subject {subject}, session {session_key}.")
    
    # Select a couple of channels for visualization (e.g., 'Fz' and 'Cz')
    channels_to_plot = ['Fz', 'Cz']
    
    # Plot PSD BEFORE filtering
    fig, ax = plt.subplots()
    raw.plot_psd(fmax=50, picks=channels_to_plot, ax=ax, show=False)
    ax.set_title("PSD BEFORE Filtering")
    fig.savefig("psd_before.png")
    plt.close(fig)
    print("Saved PSD BEFORE filtering as 'psd_before.png'.")

    # Apply bandpass filtering (e.g., 4-38 Hz)
    raw_filtered = raw.copy().filter(l_freq=4, h_freq=38, method='iir', verbose=False)
    
    # Plot PSD AFTER filtering
    fig, ax = plt.subplots()
    raw_filtered.plot_psd(fmax=50, picks=channels_to_plot, ax=ax, show=False)
    ax.set_title("PSD AFTER Filtering (4-38 Hz)")
    fig.savefig("psd_after.png")
    plt.close(fig)
    print("Saved PSD AFTER filtering as 'psd_after.png'.")

if __name__ == '__main__':
    main()
