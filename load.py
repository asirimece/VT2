#!/usr/bin/env python
import mne
from moabb.datasets import BNCI2014_001

def inspect_raw_data():
    # Initialize the dataset
    dataset = BNCI2014_001()
    print("Dataset Name:", dataset.__class__.__name__)
    
    # Get the list of subjects
    subjects = dataset.subject_list
    print("Subjects:", subjects)
    
    # Loop over each subject and session to print channel names
    for subject in subjects:
        print("\n--- Subject:", subject, "---")
        # Get the data for the subject (this returns a dictionary: session -> raw or dict of runs)
        data = dataset.get_data(subjects=[subject])
        
        for session, session_data in data[subject].items():
            print(f"Session: {session}")
            
            if isinstance(session_data, dict):
                # If there are multiple runs in the session, loop over each run.
                for run_key, raw in session_data.items():
                    print(f"  Run {run_key}:")
                    print("    Channels:", raw.info['ch_names'])
            else:
                # If session_data is a single Raw object
                print("  Channels:", session_data.info['ch_names'])

if __name__ == "__main__":
    inspect_raw_data()
