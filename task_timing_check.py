#!/usr/bin/env python

"""
verify_task_timing.py

This script demonstrates how to load raw data from BNCI2014_001 (BCI IV 2a)
via MOABB or direct MNE, extract events, and print out each event's onset time
and type. It helps confirm that the 'cue beep' (or trial start) occurs at the expected time,
and that the motor imagery period is indeed around 2s after beep, continuing until ~6s.
"""

import os

def verify_task_timing(subject_ids=[1,2,3,4,5,6,7,8,9], data_path="./data/BCI_IV2a"):
    """
    Loads the BNCI2014_001 (BCI IV 2a) dataset via MOABB, for each subject:
      1) Retrieve raw data
      2) Extract events with mne.events_from_annotations
      3) Print each event's code and onset time in seconds
    """

    # Option A: If using MOABB
    from moabb.datasets import BNCI2014001
    import mne

    dataset = BNCI2014001()  # BCI IV 2a
    # by default, BNCI2014001() in moabb automatically downloads to a local folder
    # or you can specify dataset.data_path

    for subj in subject_ids:
        print(f"\n=== Subject {subj} ===")
        # dataset.get_data(subjects=[subj]) returns a dict { subject: {session: raw} }
        # We'll do get_data()[subj] to retrieve all sessions (train/test runs)
        subj_data = dataset.get_data([subj])[subj]
        # subj_data is something like { "session_T": { "run_0": raw0, "run_1": raw1, ...},
        #                              "session_E": {...} }
        
        for sess_label, run_dict in subj_data.items():
            print(f"  Session: {sess_label}")
            # run_dict might be multiple runs (e.g., 'run_0', 'run_1', etc.)
            for run_name, raw in run_dict.items():
                print(f"    Run: {run_name}")

                # We can extract events from annotations
                events, event_id = mne.events_from_annotations(raw)
                print(f"    Found {len(events)} events.")
                for e in events:
                    # e[0] is sample index, e[2] is event code
                    onset_time_sec = e[0] / raw.info['sfreq']
                    event_code = e[2]
                    print(f"      Onset={onset_time_sec:.3f}s, Code={event_code}")

                # You can also check the raw annotations directly
                # raw.annotations can hold annotations with onset/duration/description
                # but BCI IV 2a usually uses events, not annotation descriptions

                # Optional: print the raw times between consecutive events
                # This helps see if there's a 2s gap between "cue beep" and "motor imagery" marker.

    # If you're not using MOABB but local MNE raw files, you'd do:
    # raw = mne.io.read_raw_gdf("somefile.gdf", preload=True)
    # events, event_id = mne.events_from_annotations(raw)
    # print them similarly.

if __name__ == "__main__":
    verify_task_timing(subject_ids=[1,2], data_path="./data/BCI_IV2a")
