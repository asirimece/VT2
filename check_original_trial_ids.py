#!/usr/bin/env python

import pickle
import numpy as np

def main():
    data_file = "./outputs/preprocessed_data.pkl"  # Adjust path as needed
    print(f"Loading preprocessed data from: {data_file}")
    with open(data_file, "rb") as f:
        preprocessed_data = pickle.load(f)

    subject = 1  # or whichever subject you want
    session_key = "0train"  # or "1test"
    epochs = preprocessed_data[subject][session_key]
    
    # events array shape: (n_subepochs, 3)
    #   events[i, 0] -> sample index
    #   events[i, 1] -> original trial ID
    #   events[i, 2] -> class label (0..3)
    print(f"Epochs shape: {epochs.get_data().shape} => {len(epochs)} sub-epochs")
    print("First 10 rows of events array:")
    print(epochs.events[:10])
    
    # Check the distribution of the middle column
    trial_ids = epochs.events[:, 1]
    unique_ids = np.unique(trial_ids)
    print(f"\nNumber of unique trial IDs in the middle column: {len(unique_ids)}")
    print("Sample of unique trial IDs:", unique_ids[:20], "...")
    
    # If the aggregator is correct, we expect as many unique trial IDs as real full trials.
    # For BCI IV-2a, typically each session has 288 full trials (6 runs x 48 triggers).
    # So we might expect 288 unique IDs if we sub-epoched everything.
    
    # If you see, for example, only [0], or 2 IDs, that indicates aggregator is broken.
    
    # Also check the distribution of the last column (class labels).
    class_labels = epochs.events[:, 2]
    print("\nClass label distribution (last column):", np.bincount(class_labels))
    # Should be a balanced distribution across {0,1,2,3} if everything is correct.

if __name__ == "__main__":
    main()
