#!/usr/bin/env python

import mne
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    # Path to your GDF (or other raw) file
    raw_fname = "./vt2/data/bci_iv2a/A01T.gdf"  # Adjust as needed
    raw = mne.io.read_raw_gdf(raw_fname, preload=True)
    print(f"Loaded raw with {len(raw.ch_names)} channels, {raw.n_times} samples.")

    # Extract events from annotations
    events, event_id = mne.events_from_annotations(raw, verbose=True)
    print("Extracted events (first 10):")
    print(events[:10])
    print("Event ID dictionary:", event_id)

    # Convert sample indices to seconds for clarity
    sfreq = raw.info['sfreq']
    for i, e in enumerate(events[:10]):
        sample_idx = e[0]
        t_sec = sample_idx / sfreq
        code  = e[2]
        print(f"Event {i}: sample={sample_idx}, time={t_sec:.3f}s, code={code}")

    # Plot raw + events
    fig = raw.plot(
        events=events,
        duration=10.0,
        scalings='auto',
        event_id=event_id,
        title="Raw data + Events (first 10s shown)",
        block=False
    )
    fig.canvas.draw()
    out_path = "raw_events.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved raw events plot to '{out_path}'.")

if __name__ == "__main__":
    main()
