#!/usr/bin/env python
"""
examine_epoch_boundaries.py

This script loads the preprocessed data (assumed to be saved as a pickle file containing MNE Epochs objects)
and examines the epoch boundaries for each subject’s training session.
It prints the epoch start (tmin), end (tmax), the duration, and compares the expected versus actual
number of samples. In addition, it plots a few example epochs (from one channel) with vertical lines
at tmin and tmax, saving the plots to a folder.
"""

import os
import pickle
import mne
import numpy as np
import matplotlib.pyplot as plt

def examine_epoch_boundaries(preprocessed_data_file, save_dir='epoch_boundaries_inspection'):
    print(f"Using preprocessed data file: {os.path.abspath(preprocessed_data_file)}")
    
    # Load preprocessed data
    with open(preprocessed_data_file, 'rb') as f:
        preprocessed_data = pickle.load(f)
    
    # Create a folder to save the plots if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for subj, sessions in preprocessed_data.items():
        print(f"\n=== Subject {subj} ===")
        if "0train" in sessions:
            epochs = sessions["0train"]
            # Print epoch timing information
            print(f"tmin: {epochs.tmin:.2f} s, tmax: {epochs.tmax:.2f} s, Duration: {epochs.tmax - epochs.tmin:.2f} s")
            sfreq = epochs.info['sfreq']
            expected_samples = int((epochs.tmax - epochs.tmin) * sfreq)
            data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
            actual_samples = data.shape[2]
            print(f"Sampling frequency: {sfreq:.1f} Hz")
            print(f"Expected number of samples: {expected_samples}")
            print(f"Actual number of samples: {actual_samples}")
            
            # Print event marker information
            events = epochs.events
            print(f"Events shape: {events.shape}")
            print("First 5 events:")
            print(events[:5])
            
            # Plot a few sample epochs from the first channel to visually inspect epoch boundaries
            times = epochs.times  # time vector in seconds
            n_epochs_to_plot = min(5, data.shape[0])
            for i in range(n_epochs_to_plot):
                plt.figure(figsize=(8, 4))
                plt.plot(times, data[i, 0, :], label='Channel 1')
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
                plt.title(f"Subject {subj} - Epoch {i} (Channel 1)")
                # Mark the epoch boundaries
                plt.axvline(x=epochs.tmin, color='r', linestyle='--', label='tmin')
                plt.axvline(x=epochs.tmax, color='g', linestyle='--', label='tmax')
                plt.legend()
                plot_filename = os.path.join(save_dir, f"subject_{subj}_epoch_{i}_boundary.png")
                plt.savefig(plot_filename)
                plt.close()
                print(f"Saved plot for subject {subj}, epoch {i} to {plot_filename}")
        else:
            print("No training session '0train' found for this subject.")
    
    print("\nEpoch boundary inspection completed.")

if __name__ == '__main__':
    # Adjust the path to your preprocessed data file as needed
    preprocessed_data_file = './outputs/preprocessed_data.pkl'
    examine_epoch_boundaries(preprocessed_data_file)
