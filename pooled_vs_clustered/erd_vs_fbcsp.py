# erd_vs_fbcsp_inspect_fixed_all8.py
#
# Stand‐alone script to visualize per‐subject μ/β ERD (8–12 Hz, 13–30 Hz)
# time‐courses and topographies **for all eight EEG channels**—
# so you can judge whether a simple ERD/ERS feature set is stable enough (and low‐dimensional enough)
# for your pipeline, versus needing FBCSP.
#
# Usage: python erd_vs_fbcsp_inspect_fixed_all8.py
# Requirements:
#   • Python 3.7+ with the following packages installed:
#       mne, scipy, seaborn, matplotlib, pandas, numpy
#   • A “dump/preprocessed_data_custom.pkl” file where each key is a string subject ID
#     and each value is a dict with at least a `"train"` key whose value is an mne.EpochsArray.
#   • The EEG channels in each EpochsArray must include at least
#       ['Fz', 'C3', 'Cz', 'C4', 'PO7', 'Oz', 'PO8', 'Pz']  (order doesn’t matter).
#
# This script will create an “erd_inspection/” folder (if it doesn’t exist) and save:
#   • One PNG of μ‐band time‐courses (%Δ from baseline) for all 8 channels, per subject
#   • One PNG of β‐band time‐courses (%Δ from baseline) for all 8 channels, per subject
#   • One μ‐band ERD topomap (fractional ERD) for all 8 channels, per subject
#   • One β‐band ERD topomap (fractional ERD) for all 8 channels, per subject
#   • A CSV “erd_summary_stats.csv” summarizing, per subject:
#       – mean μ‐ERD (averaged over all channels)
#       – std μ‐ERD (across channels)
#       – mean β‐ERD (averaged over all channels)
#       – std β‐ERD (across channels)
#
# You can then inspect these plots to decide if a fixed ERD/ERS pipeline (using a subset of channels)
# is viable, or if you need to stick with FBCSP (which will automatically pick subject‐specific spatial filters).

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from scipy.signal import hilbert

# =============================================================================
# PARAMETERS (edit as needed)
# =============================================================================

PREPROC_PKL    = "dump/preprocessed_data_custom.pkl"
OUT_DIR        = "erd_inspection"
MI_START       = 0.0      # cue at 0.0 s
MI_END         = 2.0      # end of trial window (for MI)

BASELINE_START = 0.0      # baseline window start (e.g. 0.0–0.5 s)
BASELINE_END   = 0.5

# Frequency bands to inspect (mu and beta)
BANDS = {
    "mu":   (8.0, 12.0),
    "beta": (13.0, 30.0)
}

# **ALL eight EEG channels** you wish to inspect (order must match the channels in your EpochsArray):
EEG_CHANNELS = ['Fz', 'C3', 'Cz', 'C4', 'PO7', 'Oz', 'PO8', 'Pz']

# =============================================================================
# MAKE OUTPUT DIRECTORY
# =============================================================================

os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# 1) LOAD PREPROCESSED DATA
# =============================================================================

with open(PREPROC_PKL, "rb") as f:
    preproc_dict = pickle.load(f)

# We expect preproc_dict to be a dict where each key is a subject ID string,
# and each value is a dict containing at least `"train": mne.EpochsArray`.
subjects = sorted(preproc_dict.keys(), key=lambda s: int(s))

print(f"Found {len(subjects)} subjects in preprocessed data.\n")


# =============================================================================
# 2) FOR EACH SUBJECT: COMPUTE & PLOT ERD/ERS FOR μ & β ON ALL 8 CHANNELS
# =============================================================================

summary_stats = []  # will hold one dict per subject

for subj in subjects:
    entry = preproc_dict[subj]
    if not isinstance(entry, dict) or "train" not in entry:
        print(f"Skipping subject {subj}: no 'train' key present.")
        continue

    epochs = entry["train"]  # mne.EpochsArray
    times  = epochs.times    # e.g. array([-0.500, -0.490, …, 1.990])
    data   = epochs.get_data()      # shape: (n_epochs, n_channels, n_times)
    ch_names = epochs.ch_names.copy()  # list of channel names

    # Verify that all eight EEG_CHANNELS are present
    missing = [ch for ch in EEG_CHANNELS if ch not in ch_names]
    if missing:
        print(f"  • Subject {subj}: missing channels {missing} → skip this subject entirely.")
        continue

    subj_stats = {
        "subject": subj,
        "n_trials": data.shape[0],
        "n_channels": data.shape[1],
        "mu_mean_erd": np.nan,
        "mu_std_erd_channels": np.nan,
        "beta_mean_erd": np.nan,
        "beta_std_erd_channels": np.nan
    }

    # For each frequency band, compute envelope → ERD and plot time‐courses & topomap
    for band_name, (fmin, fmax) in BANDS.items():
        # ----- 2.1 Filter in the band for the entire Epochs set -----
        epochs_band = epochs.copy().filter(
            l_freq=fmin, h_freq=fmax, method="iir",
            iir_params=dict(ftype='butter', order=4),
            picks=EEG_CHANNELS,  # only filter those 8 channels
            verbose=False
        )

        # ----- 2.2 Compute Hilbert envelope on filtered data -----
        raw_data = epochs_band.get_data()  
        # shape = (n_epochs, n_chan_selected=8, n_times)
        analytic = hilbert(raw_data, axis=-1)  
        envelope = np.abs(analytic)  # (n_epochs, 8, n_times)

        # ----- 2.3 Build baseline & MI masks -----
        baseline_mask = (times >= BASELINE_START) & (times < BASELINE_END)
        mi_mask       = (times >= MI_START)       & (times < MI_END)

        if not baseline_mask.any():
            print(f"  • Subject {subj}, band {band_name}: no baseline samples → skipping ERD for this band.")
            continue

        # ----- 2.4 Compute mean power per trial & channel in both windows -----
        # baseline_power shape = (n_epochs, 8)
        baseline_power = envelope[..., baseline_mask].mean(axis=2)
        mi_power       = envelope[..., mi_mask].mean(axis=2)

        # ----- 2.5 Trial‐by‐trial ERD per channel = (B − MI) / B -----
        # shape of erd_trials = (n_epochs, 8)
        with np.errstate(divide='ignore', invalid='ignore'):
            erd_trials = (baseline_power - mi_power) / baseline_power

        # ----- 2.6 Mean ERD across trials for each channel -----
        mean_erd_channels = np.nanmean(erd_trials, axis=0)  # shape = (8,)

        # Save summary stats (mean+std across channels):
        subj_stats[f"{band_name}_mean_erd"] = np.nanmean(mean_erd_channels)
        subj_stats[f"{band_name}_std_erd_channels"] = np.nanstd(mean_erd_channels)

        # ----- 2.7 Plot time‐course (%Δ from baseline) for all 8 channels -----
        plt.figure(figsize=(7, 3.5))
        for ch in EEG_CHANNELS:
            idx = ch_names.index(ch)
            # average envelope across all trials, for that channel
            mean_env = envelope[:, EEG_CHANNELS.index(ch), :].mean(axis=0)
            # compute percent-change relative to baseline average
            baseline_avg = mean_env[baseline_mask].mean()
            if baseline_avg == 0:
                pct_change = np.zeros_like(mean_env)
            else:
                pct_change = 100.0 * (mean_env - baseline_avg) / baseline_avg

            plt.plot(times, pct_change, label=ch, linewidth=1.2)

        plt.axvline(0.0, color="k", linestyle="--", alpha=0.7)
        plt.axhline(0.0, color="gray", linestyle=":", alpha=0.7)
        plt.legend(loc="upper right", fontsize=6, ncol=4, framealpha=0.7)
        plt.xlabel("Time (s)")
        plt.ylabel(f"{band_name}-power %Δ")
        plt.title(f"Subj {subj} — {band_name} Envelope (%Δ) Across 8 Channels")
        plt.tight_layout()
        fname_tc = os.path.join(OUT_DIR, f"{subj}_{band_name}_timecourse_all8.png")
        plt.savefig(fname_tc, dpi=150)
        plt.close()

        # ----- 2.8 Topomap of mean ERD (averaged over trials) for all 8 channels -----
        erd_evoked = mean_erd_channels  # length = 8, in the same order as EEG_CHANNELS

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        # We rely on epochs_band.info for the channel locations / montage.
        # MNE’s plot_topomap expects erd_evoked to align with info['ch_names'] picks.
        mne.viz.plot_topomap(
            erd_evoked,
            epochs_band.info,
            axes=ax,
            show=False,
            names=EEG_CHANNELS,
            mask=None,
            cmap="RdBu_r",
            contours=0
        )
        ax.set_title(f"Subj {subj} — {band_name} ERD (fraction)")
        cbar = fig.colorbar(ax.images[0], ax=ax, orientation="vertical", shrink=0.75)
        cbar.set_label("ERD (fraction)")

        fname_tp = os.path.join(OUT_DIR, f"{subj}_{band_name}_topo_all8.png")
        plt.savefig(fname_tp, dpi=150)
        plt.close()

    # Append this subject’s summary to the list
    summary_stats.append(subj_stats)

# =============================================================================
# 3) SAVE SUMMARY STATS TO CSV
# =============================================================================

df_stats = pd.DataFrame(summary_stats)
csv_path = os.path.join(OUT_DIR, "erd_summary_stats.csv")
df_stats.to_csv(csv_path, index=False)

print("\nDone. All ERD inspection plots (timecourses + topomaps) and summary CSV")
print(f"are saved under the folder: '{OUT_DIR}/'\n")
