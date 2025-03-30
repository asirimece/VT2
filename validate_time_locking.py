#!/usr/bin/env python
import pickle
import numpy as np

def main():
    filename = "outputs/22ica/22ica_features.pkl"  # adjust the path as needed
    print(f"Loading preprocessed data from {filename}...")
    with open(filename, "rb") as f:
        preprocessed_data = pickle.load(f)
    
    # These parameters should match those used in your epoching configuration.
    window_length = 4.0  # sliding window duration in seconds
    step_size = 0.5      # step between windows in seconds
    tmin_event = -0.5    # start time relative to event (for original epoch)
    tmax_event = 3.5     # end time relative to event (for original epoch)
    
    # Calculate expected number of samples per original epoch and per sliding window.
    original_duration = tmax_event - tmin_event  # total duration of a time-locked epoch
    sample_sfreq = None
    # Get the sampling frequency from the first available epochs object.
    for subj, sessions in preprocessed_data.items():
        for session, epochs in sessions.items():
            sample_sfreq = epochs.info.get('sfreq', None)
            if sample_sfreq is not None:
                break
        if sample_sfreq is not None:
            break
    if sample_sfreq is None:
        print("Error: Unable to determine sampling frequency from data.")
        return
    
    n_samples_window = int(round(window_length * sample_sfreq))
    step_samples = int(round(step_size * sample_sfreq))
    original_samples = int(round(original_duration * sample_sfreq))
    
    # Expected number of sub-epochs per original epoch:
    expected_sub_epochs = ((original_samples - n_samples_window) // step_samples) + 1
    print("----- Validation of Time-Locking -----")
    print(f"Sampling frequency: {sample_sfreq} Hz")
    print(f"Original epoch duration: {original_duration} s ({original_samples} samples)")
    print(f"Sliding window length: {window_length} s ({n_samples_window} samples)")
    print(f"Step size: {step_size} s ({step_samples} samples)")
    print(f"Expected sub-epochs per original epoch: {expected_sub_epochs}\n")
    
    # Now validate: for each subject/session, the total number of sliding-window epochs
    # divided by the expected number per original epoch should roughly equal the number of trials.
    for subj, sessions in preprocessed_data.items():
        print(f"\n--- Subject: {subj} ---")
        for session, epochs in sessions.items():
            n_sub_epochs = len(epochs.events)
            # Estimated number of original (time-locked) epochs/trials:
            estimated_trials = n_sub_epochs / expected_sub_epochs
            print(f"\nSession: {session}")
            print(f"Total sliding-window epochs: {n_sub_epochs}")
            print(f"Estimated number of original epochs (trials): {estimated_trials:.2f}")
            
if __name__ == "__main__":
    main()
