import numpy as np
import mne
from omegaconf import OmegaConf

def create_macro_epochs(raw: mne.io.Raw, dataset_config) -> mne.Epochs:
    """
    Create macro MNE Epochs from tmin to tmax around each event.
    (Unchanged—kept here in case you still need it elsewhere.)
    """
    tmin = dataset_config.epoching.kwargs.tmin
    tmax = dataset_config.epoching.kwargs.tmax
    
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=tmin, tmax=tmax,
        baseline=None, preload=True, verbose=False
    )
    return epochs

def extract_time_locked_epochs(raw, tmin, tmax):
    """
    1) Read in annotations
    2) Prune event_id to only classes 1 & 3
    3) Create an Epochs that only knows about those two classes
    """
    # Resolve OmegaConf nodes if present
    tmin = float(OmegaConf.to_container(tmin, resolve=True)) if hasattr(tmin, "keys") else float(tmin)
    tmax = float(OmegaConf.to_container(tmax, resolve=True)) if hasattr(tmax, "keys") else float(tmax)

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    # Keep only codes 1 and 3
    event_id = {name: code for name, code in event_id.items() if code in (1, 3)}
    # Build epochs with only those IDs
    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=tmin, tmax=tmax,
        baseline=None, preload=True, verbose=False
    )
    return epochs

def crop_subepochs(epochs, window_length, step_size):
    """
    1) Filter to only trials with original event codes 1 or 3
    2) Slide a window of length `window_length` (s) by `step_size` (s)
    3) Map 1→0, 3→1 in the new events array
    """
    sfreq = epochs.info['sfreq']
    data = epochs.get_data()      # (n_trials, n_chans, n_times)
    events = epochs.events        # (n_trials, 3)

    # 1) keep only trials with code 1 or 3
    mask = np.isin(events[:, 2], [1, 3])
    data = data[mask]
    events = events[mask]

    window_samples = int(round(window_length * sfreq))
    step_samples   = int(round(step_size   * sfreq))

    all_crops = []
    all_events = []
    counter   = 0

    # mapping for your binary labels
    label_map = {1: 0, 3: 1}

    for trial_idx, raw_label in enumerate(events[:, 2]):
        mapped_label = label_map[raw_label]
        trial_data   = data[trial_idx]         # (n_chans, n_times)
        n_sub        = 1 + (trial_data.shape[1] - window_samples) // step_samples

        for j in range(n_sub):
            start = j * step_samples
            end   = start + window_samples
            all_crops.append(trial_data[:, start:end])
            all_events.append([counter, trial_idx, mapped_label])
            counter += 1

    all_crops  = np.stack(all_crops)        # (n_subepochs, n_chans, window_samples)
    all_events = np.array(all_events, int)  # (n_subepochs, 3)

    new_info = epochs.info.copy()
    new_tmin = 0.0

    new_epochs = mne.EpochsArray(
        data=all_crops,
        info=new_info,
        events=all_events,
        tmin=new_tmin,
        baseline=None,
        verbose=False
    )
    return new_epochs

def time_lock_and_slide_epochs(raw, tmin, tmax, window_length, step_size):
    """
    Combines everything:
      - extract only 1&3 time-locked epochs
      - crop into windowed subepochs with labels 0/1
    """
    epochs = extract_time_locked_epochs(raw, tmin, tmax)
    new_epochs = crop_subepochs(epochs, window_length, step_size)
    return new_epochs
