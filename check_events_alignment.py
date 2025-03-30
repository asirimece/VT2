#!/usr/bin/env python3
"""
check_events_alignment.py

Loads EEG data from vt2/data/bci_iv2a/A01E.gdf into MNE, creates static (non‐interactive)
plots that show the event markers and the average (evoked) response for a chosen tmin/tmax.
These saved plots help verify that the epochs capture the intended motor imagery period.
"""

import os
import mne
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 1) LOAD RAW DATA
# ----------------------------------------------------------------------------
raw_fname = os.path.join("vt2", "data", "bci_iv2a", "A01E.gdf")
print(f"Loading raw file from: {raw_fname}")
raw = mne.io.read_raw_gdf(raw_fname, preload=True)

# ----------------------------------------------------------------------------
# 2) EXTRACT EVENTS & DEFINE EVENT IDS
# ----------------------------------------------------------------------------
# Extract events from the raw file.
events, _ = mne.events_from_annotations(raw, verbose=True)

# Update event mapping: use IDs 1, 2, 3, 4.
my_event_dict = {
    'left_hand':  1,
    'right_hand': 2,
    'feet':       3,
    'tongue':     4
}
print("Event mapping:")
print(my_event_dict)

# ----------------------------------------------------------------------------
# 3) PLOT EVENTS (Static) AND SAVE FIGURE
# ----------------------------------------------------------------------------
fig_events = mne.viz.plot_events(events, event_id=my_event_dict,
                                   sfreq=raw.info['sfreq'], show=False)
events_plot_fname = "raw_events.png"
fig_events.savefig(events_plot_fname)
plt.close(fig_events)
print(f"Saved events plot to: {events_plot_fname}")

# ----------------------------------------------------------------------------
# 4) CREATE EPOCHS WITH SPECIFIED TMIN/TMAX
# ----------------------------------------------------------------------------
# Here, tmin = -0.5 s and tmax = 4.5 s.
tmin, tmax = 2.0, 6.0
epochs = mne.Epochs(raw, events, event_id=my_event_dict,
                    tmin=tmin, tmax=tmax,
                    baseline=None, preload=True, verbose=True)
print(f"Created {len(epochs)} epochs from {tmin} to {tmax} s.")

# ----------------------------------------------------------------------------
# 5) PLOT THE AVERAGE (EVOKED) RESPONSE AND SAVE FIGURE
# ----------------------------------------------------------------------------
evoked = epochs.average()
fig_evoked = evoked.plot(show=False, time_unit='s')
evoked_plot_fname = "epochs_evoked.png"
fig_evoked.savefig(evoked_plot_fname)
plt.close(fig_evoked)
print(f"Saved evoked (average epoch) plot to: {evoked_plot_fname}")
