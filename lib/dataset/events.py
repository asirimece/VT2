#!/usr/bin/env python3
"""
events.py

This module handles the unification of raw annotation codes/strings into
a consistent set of labels as defined in your configuration.

It uses two dictionaries:
  1. unify_annotations: maps raw annotation codes (as strings) or synonyms
     to a unified label. For example:
         {"769": "left_hand", "770": "right_hand", "771": "feet", "772": "tongue",
          "foot": "feet", "left": "left_hand", "hand_left": "left_hand",
          "right": "right_hand", "hand_right": "right_hand"}
  2. event_markers: maps the unified label to the final numeric code. For example:
         {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}

This function prints debug messages to show what each raw event code is mapped to.
"""

import mne

def unify_events(raw, unify_annotations, event_markers):
    """
    Unify raw event annotations and map them to final numeric codes.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    unify_annotations : dict
        Dictionary mapping raw annotation codes (as strings) or synonyms to unified labels.
    event_markers : dict
        Dictionary mapping unified labels to final numeric codes.

    Returns
    -------
    new_events : numpy.ndarray
        The modified events array with unified and mapped event codes.
    new_event_id : dict
        A dictionary mapping unified label names to numeric codes.
    """
    # Extract events using MNE’s events_from_annotations
    events, _ = mne.events_from_annotations(raw, verbose=False)
    new_events = events.copy()

    for i, evt in enumerate(new_events):
        raw_code = evt[2]
        # Convert raw code to string
        code_str = str(raw_code)
        if code_str in unify_annotations:
            unified_label = unify_annotations[code_str]
            if unified_label in event_markers:
                mapped_value = event_markers[unified_label]
                new_events[i, 2] = mapped_value
                print(f"[DEBUG] Event {i}: raw code {raw_code} -> unified '{unified_label}' -> mapped {mapped_value}")
            else:
                print(f"[DEBUG] Event {i}: unified label '{unified_label}' not found in event_markers. Keeping original code {raw_code}.")
        else:
            print(f"[DEBUG] Event {i}: raw code {raw_code} not found in unify_annotations. Keeping original.")
    
    # Construct new event_id from the event_markers (e.g., {'left_hand':1, ...})
    new_event_id = {label: code for label, code in event_markers.items()}
    return new_events, new_event_id

