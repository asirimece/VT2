import mne

fif_raw = "/home/ubuntu/VT2/data/04_sampleFreq200_80_events/recording_subject_309_session_1_raw.fif"
raw = mne.io.read_raw_fif(fif_raw, preload=False)

# 1) Extract events; if your raw has annotations, convert them:
events, event_id = mne.events_from_annotations(raw)

# 2) Define your epoch window (as in your config)
tmin, tmax = -1.0, 2.0  # seconds
# This will yield exactly (tmax - tmin) * sfreq = 3.0 * 200 = 600 samples

epochs = mne.Epochs(raw, events, event_id,
                    tmin=tmin, tmax=tmax,
                    baseline=None,
                    preload=False)

print("Sampling rate (Hz):", epochs.info['sfreq'])
print("Time‐points per epoch:", epochs.get_data().shape[-1])
print(f"  → {epochs.get_data().shape[-1] / epochs.info['sfreq']} seconds per epoch")
