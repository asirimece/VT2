import os
import mne
import matplotlib.pyplot as plt

data_dir = "data/"
out_dir = "event_plots"
os.makedirs(out_dir, exist_ok=True)

fif_files = [f for f in os.listdir(data_dir) if f.endswith(".fif")]

print(f"Found {len(fif_files)} .fif files in {data_dir}\n{'='*50}")

for i, fname in enumerate(sorted(fif_files)):
    fpath = os.path.join(data_dir, fname)
    print(f"\n[{i+1}/{len(fif_files)}] Inspecting: {fname}")
    try:
        raw = mne.io.read_raw_fif(fpath, preload=True, verbose=False)
        print(f"  - n_channels: {raw.info['nchan']}")
        print(f"  - Channel names: {raw.ch_names}")
        print(f"  - Duration (s): {raw.n_times / raw.info['sfreq']:.2f}")
        print(f"  - Sample rate: {raw.info['sfreq']} Hz")
        
        # Check annotations
        if raw.annotations is not None and len(raw.annotations) > 0:
            print(f"  - Found {len(raw.annotations)} annotations.")
            print(f"    Event labels: {set(raw.annotations.description)}")
            # Show first 5 annotations
            for ann in raw.annotations[:5]:
                print(f"    > At {ann['onset']}s: {ann['description']}")
        else:
            print("  - No annotations found!")
        
        # Try to create events from annotations and save plot
        try:
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            print(f"  - Number of events: {len(events)}")
            print(f"  - Event IDs: {event_id}")

            # Save event raster plot
            times_sec = events[:, 0] / raw.info['sfreq']
            plt.figure(figsize=(12, 2))
            plt.eventplot(times_sec, lineoffsets=1, colors='b')
            plt.xlabel('Time (s)')
            plt.title(f"Events: {fname}")
            plt.tight_layout()
            out_path = os.path.join(out_dir, fname.replace('.fif', '_events.png'))
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"  - Event plot saved to {out_path}")
        except Exception as e:
            print(f"  - Could not parse events from annotations: {e}")

    except Exception as e:
        print(f"  !! Error loading file: {e}")

print("\nInspection complete.")


import os
import mne
import numpy as np
import matplotlib.pyplot as plt

data_dir = "data/"
output_dir = "inspection_results"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "psd_plots"), exist_ok=True)

fif_files = [f for f in os.listdir(data_dir) if f.endswith(".fif")]

print(f"Found {len(fif_files)} .fif files in {data_dir}")

stats_report = []

for i, fname in enumerate(sorted(fif_files)):
    fpath = os.path.join(data_dir, fname)
    print(f"\n[{i+1}/{len(fif_files)}] Inspecting: {fname}")

    try:
        raw = mne.io.read_raw_fif(fpath, preload=True, verbose=False)
        # ---- Save PSD Plot ----
        psd_fig = raw.plot_psd(show=False, fmax=60)
        psd_path = os.path.join(output_dir, "psd_plots", f"{fname.replace('.fif', '')}_psd.png")
        psd_fig.savefig(psd_path, bbox_inches="tight")
        plt.close(psd_fig)
        print(f"  - PSD plot saved: {psd_path}")

        # ---- Standardization stats ----
        data = raw.get_data()
        print(f"    Data shape: {data.shape}, min: {data.min()}, max: {data.max()}")
        print(f"    First 5 values of first channel: {data[0, :5]}")
        means = data.mean(axis=1)
        stds = data.std(axis=1)
        overall_mean = means.mean()
        overall_std = stds.mean()

        stats_report.append({
            "file": fname,
            "mean_per_channel": means,
            "std_per_channel": stds,
            "overall_mean": overall_mean,
            "overall_std": overall_std,
        })
        print(f"  - Mean per channel: {np.round(means, 6)}")
        print(f"  - Std per channel:  {np.round(stds, 6)}")
        print(f"  - Overall mean:     {overall_mean:.3f}")
        print(f"  - Overall std:      {overall_std:.3f}")

    except Exception as e:
        print(f"  !! Error loading file: {e}")

# ---- Save stats summary ----
stats_file = os.path.join(output_dir, "standardization_stats.txt")
with open(stats_file, "w") as f:
    for item in stats_report:
        f.write(f"File: {item['file']}\n")
        f.write(f"  Mean per channel: {np.round(item['mean_per_channel'], 6)}\n")
        f.write(f"  Std per channel:  {np.round(item['std_per_channel'], 6)}\n")
        f.write(f"  Overall mean:     {item['overall_mean']:.3f}\n")
        f.write(f"  Overall std:      {item['overall_std']:.3f}\n")
        f.write("\n")

print("\nInspection complete.")
print(f"Standardization stats summary saved at: {stats_file}")
print(f"All PSD plots saved to: {os.path.join(output_dir, 'psd_plots')}")
