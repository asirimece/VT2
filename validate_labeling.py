#!/usr/bin/env python
import pickle
import numpy as np

def main():
    filename = "./outputs/2025-03-12/11-39-55/outputs/preprocessed_data.pkl"  # adjust the path as needed
    print(f"Loading preprocessed data from {filename}...")
    with open(filename, "rb") as f:
        preprocessed_data = pickle.load(f)
    
    # Iterate over subjects and sessions
    for subj, sessions in preprocessed_data.items():
        print(f"\n--- Subject: {subj} ---")
        for session, epochs in sessions.items():
            event_labels = epochs.events[:, 2]  # labels are stored in the third column
            unique, counts = np.unique(event_labels, return_counts=True)
            print(f"\nSession: {session}")
            print("Label distribution:")
            for label, count in zip(unique, counts):
                print(f"  Event code {label}: {count} epochs")
                
if __name__ == "__main__":
    main()
