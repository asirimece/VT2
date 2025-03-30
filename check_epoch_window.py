#!/usr/bin/env python

import pickle
import numpy as np

def main():
    data_file = "./outputs/preprocessed_data.pkl"
    print(f"Loading data from {data_file}")
    with open(data_file, "rb") as f:
        preprocessed_data = pickle.load(f)
    
    subj = 1
    session_key = "0train"  # or "1test"
    epochs = preprocessed_data[subj][session_key]
    
    # MNE epochs have: epochs.tmin, epochs.tmax, and epochs.info["sfreq"]
    print(f"Subject {subj} - Session {session_key}")
    print(f"Time range: tmin={epochs.tmin:.3f}, tmax={epochs.tmax:.3f}")
    duration = epochs.tmax - epochs.tmin
    print(f"Duration: {duration:.3f} seconds per epoch")
    
    sfreq = epochs.info["sfreq"]
    print(f"Sampling frequency: {sfreq} Hz")
    
    # check number of time samples
    n_samples = epochs.get_data().shape[-1]
    print(f"Epoch data shape: {epochs.get_data().shape} => {n_samples} samples in time dimension")
    
    # If you used tmin=-0.5 and tmax=4.5 => total = 5.0s.
    # at 250 Hz, we expect 5.0 * 250 = 1250 samples (maybe ±1, depending on rounding).
    
    # If you see 2.0-second sub-epochs, that implies your forced sliding-window was applied,
    # so you might see only 500 samples. That is not the original big window, but the cropped sub-epochs.

if __name__ == "__main__":
    main()
