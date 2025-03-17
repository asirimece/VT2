#!/usr/bin/env python
"""
inspect_raw_annotations.py

This script demonstrates how to inspect raw annotations from an MNE Raw object.
It prints the annotations, extracts events from them, and shows the distribution
of event codes.

Usage:
    python inspect_raw_annotations.py
"""

import mne
import numpy as np
from collections import Counter

def inspect_raw_annotations(raw):
    """
    Print raw annotations and display the distribution of event codes.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data with annotations.
    """
    # Print all raw annotations
    print("Raw Annotations:")
    print(raw.annotations)
    
    # Extract events from annotations
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    print("\nFirst 10 events extracted from annotations:")
    print(events[:10])
    
    # Get distribution of event codes
    unique_codes, counts = np.unique(events[:, 2], return_counts=True)
    print("\nEvent code distribution:")
    for code, count in zip(unique_codes, counts):
        print(f"  Event code {code}: {count} occurrence(s)")
    
def load_sample_raw():
    """
    Loads raw data from a sample file.
    Replace this function with your own data loading logic if needed.
    """
    # Example: read a sample raw file (modify the file path as needed)
    raw = mne.io.read_raw_fif("sample_raw.fif", preload=True)
    return raw

if __name__ == "__main__":
    # Option 1: Load a sample raw file
    try:
        raw = load_sample_raw()
    except FileNotFoundError:
        print("Sample raw file not found. Please modify load_sample_raw() with your own data path.")
        exit(1)
    
    # Option 2 (for MOABB dataset):
    # from moabb.datasets import BNCI2014001
    # dataset = BNCI2014001()
    # all_data = dataset.get_data()
    # # For example, take subject 1's training data and concatenate if necessary:
    # subj = list(all_data.keys())[0]
    # subj_data = all_data[subj]
    # train_data = subj_data.get("0train")
    # if isinstance(train_data, dict):
    #     raw = mne.concatenate_raws(list(train_data.values()))
    # else:
    #     raw = train_data

    inspect_raw_annotations(raw)
