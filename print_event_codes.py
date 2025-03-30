#!/usr/bin/env python3
"""
print_event_codes.py

Script to load BNCI2014001 (BCI IV 2a) via MOABB and print the unique
event codes for each subject's sessions. This helps you see the actual
trigger IDs (e.g., 769..772 or 0..3) present in the raw annotations.
"""

import numpy as np
import mne
from moabb.datasets import BNCI2014001

def main():
    # 1) Instantiate the dataset
    dataset = BNCI2014001()

    # 2) Get all_data => dict { subject_id: { session_label: runs } }
    all_data = dataset.get_data()

    # 3) Loop over subjects
    for subj_id in sorted(all_data.keys()):
        print(f"\n--- Subject {subj_id} ---")
        subj_data = all_data[subj_id]

        # 4) Each subject typically has "0train" and "1test" sessions in BCI IV 2a
        for sess_label, runs_or_raw in subj_data.items():
            # The session might be multiple runs in a dict, or a single Raw
            if isinstance(runs_or_raw, dict):
                # multiple runs: e.g. run_0, run_1, ...
                run_list = list(runs_or_raw.values())
                raw = mne.concatenate_raws(run_list)
            else:
                # single raw
                raw = runs_or_raw

            # 5) Convert annotations to events, then print the unique trigger codes
            events, _ = mne.events_from_annotations(raw)
            print(events)
            unique_codes = np.unique(events[:, 2])
            print(f"  Session: {sess_label}, unique event codes = {unique_codes}")

if __name__ == "__main__":
    main()
