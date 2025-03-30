#!/usr/bin/env python
"""
verify_epoching.py

This script inspects the MNE Epochs in a preprocessed_data.pkl file to confirm:
1) The event codes (1,2,3,4 or 769..772) match the intended motor imagery classes.
2) The epochs are capturing the correct time window for motor imagery (e.g., from 2-6s post-cue).

It prints:
- The MNE epochs' tmin/tmax or times array, so you see if you're using 0-4s or something else.
- For each subject and each session (0train/1test), the number of epochs for each event code.
- The raw event codes if they exist (like 769..772).
- Possibly a quick check that code 1 means left hand, etc. (you can adapt to your class labeling).
"""

import os
import pickle
import numpy as np

def verify_epoching(preprocessed_file):
    with open(preprocessed_file, 'rb') as f:
        preprocessed_data = pickle.load(f)
    
    for subj, sessions in preprocessed_data.items():
        print(f"\n=== Subject {subj} ===")
        
        for sess_label in ["0train", "1test"]:
            if sess_label not in sessions:
                print(f"  No {sess_label} session for subject {subj}, skipping.")
                continue
            
            epochs = sessions[sess_label]
            print(f"  Session: {sess_label}")
            print(f"    Number of epochs: {len(epochs)}")
            
            # Print the time window from the MNE Epochs object
            # Usually epochs have attributes tmin, tmax
            if hasattr(epochs, 'tmin') and hasattr(epochs, 'tmax'):
                print(f"    MNE epoch time range: tmin={epochs.tmin}, tmax={epochs.tmax}")
            else:
                print("    Warning: This epochs object does not have tmin/tmax. Possibly custom epoching?")
            
            # Check event codes
            # Typically stored in epochs.events[:, -1]
            if hasattr(epochs, 'events'):
                raw_codes = epochs.events[:, -1]
                unique_codes, counts = np.unique(raw_codes, return_counts=True)
                print("    Unique event codes:", unique_codes)
                print("    Counts per code:", dict(zip(unique_codes, counts)))
                
                # If you have a known mapping (e.g. code 1->left, 2->right, 3->foot, 4->tongue),
                # you can do a quick check or print. 
                # E.g.:
                code_mapping = {
                    1: "Left_Hand",
                    2: "Right_Hand",
                    3: "Feet",
                    4: "Tongue"
                }
                # Print how many trials for each code -> label
                for c in unique_codes:
                    if c in code_mapping:
                        print(f"      Code {c} = {code_mapping[c]}, # trials={counts[np.where(unique_codes==c)][0]}")
                    else:
                        print(f"      Code {c} is unknown or outside [1..4]? # trials={counts[np.where(unique_codes==c)][0]}")
            else:
                print("    No epochs.events attribute found. Can't inspect codes.")

            # Optionally, print the actual times array for the first epoch
            # to confirm your epoch window
            if len(epochs) > 0:
                example_data = epochs[0].get_data()  # shape (n_channels, n_times)
                # The times array is typically epochs.times
                if hasattr(epochs, 'times'):
                    times_array = epochs.times  # e.g. from -1.0 to 4.0
                    print(f"    Example epoch times array (length={len(times_array)}): from {times_array[0]} to {times_array[-1]}")
                else:
                    print("    This epochs object has no 'times' attribute for the window. Possibly a custom object.")
            
            # You could do more checks if needed, e.g. verifying if 2-6s is used.

def main():
    preprocessed_data_file = "./outputs/preprocessed_data.pkl"
    print(f"Inspecting preprocessed data: {preprocessed_data_file}")
    if not os.path.exists(preprocessed_data_file):
        print("File does not exist. Exiting.")
        return
    
    verify_epoching(preprocessed_data_file)

if __name__ == "__main__":
    main()
