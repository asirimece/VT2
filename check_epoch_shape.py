#!/usr/bin/env python3
"""
check_epoch_shape.py

Loads preprocessed MNE Epochs from a pickle file and prints important info
such as shape, sampling frequency, tmin/tmax, etc.
"""

import mne
import pickle

def main():
    # Load preprocessed data from the pickle file
    with open("./outputs/preprocessed_data.pkl", "rb") as f:
        all_data = pickle.load(f)
    
    # Select epochs for a specific subject and session
    # Change the subject key or session name if needed.
    subject_key = 1
    session_key = "0train"
    epochs = all_data[subject_key][session_key]
    
    # Retrieve the data array from the epochs
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    
    print("Epochs array shape:", data.shape, "=> (n_epochs, n_channels, n_times)")
    print("Sampling frequency (Hz):", epochs.info['sfreq'])
    print("tmin, tmax (s):", epochs.tmin, epochs.tmax)
    print("Epoch duration (s):", epochs.tmax - epochs.tmin)
    print("Number of time points per epoch:", len(epochs.times))
    print("First few 'times' values:", epochs.times[:5], "...")
    print("Last few 'times' values:", epochs.times[-5:], "...")
    
    # If you expect 4-second epochs at 250 Hz, you'd want:
    # n_times == 1000  (i.e., 4 s * 250 Hz)

if __name__ == "__main__":
    main()
