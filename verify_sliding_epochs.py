import numpy as np
import mne
import pickle

def verify_sliding_epochs(epochs, crop_window_length, sfreq):
    # Get data shape: (n_epochs, n_channels, n_samples)
    data = epochs.get_data()
    n_epochs, n_channels, n_samples = data.shape
    expected_samples = int(round(crop_window_length * sfreq))
    print(f"Each epoch should have {expected_samples} samples; found {n_samples} samples.")
    print(f"Total number of epochs: {n_epochs}")

def main():
    # For this example, assume you load a preprocessed EpochsArray from your file
    preproc_file = "./outputs/preprocessed_data.pkl"
    with open(preproc_file, "rb") as f:
        preprocessed_data = pickle.load(f)
    
    # Pick one subject's training session (assuming sliding window was applied)
    # (e.g., sessions["0train"] should now be the sliding windowed epochs)
    subject = list(preprocessed_data.keys())[0]
    epochs = preprocessed_data[subject]["0train"]
    
    sfreq = epochs.info["sfreq"]
    crop_window_length = 2.0  # in seconds, as set in your config
    verify_sliding_epochs(epochs, crop_window_length, sfreq)

if __name__ == "__main__":
    main()
