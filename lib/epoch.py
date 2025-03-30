import numpy as np
import mne
from omegaconf import OmegaConf

# NOT USED
def extract_time_locked_epochs(raw, tmin, tmax):
    # Ensure tmin and tmax are floats (resolve if coming from a config)
    tmin = float(OmegaConf.to_container(tmin, resolve=True)) if hasattr(tmin, "keys") else float(tmin)
    tmax = float(OmegaConf.to_container(tmax, resolve=True)) if hasattr(tmax, "keys") else float(tmax)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(raw, events, event_id=event_id,
                        tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose=False)
    return epochs, event_id

def force_sliding_window_cropping(epochs, window_length, step_size):
    """
    Takes an MNE Epochs object (macro epochs) and re-crops each trial into overlapping
    sub-epochs of length window_length seconds, stepping every step_size seconds.
    The second column of the event array is set to the original trial index.
    """
    sfreq = epochs.info['sfreq']
    original_data = epochs.get_data()    # shape: (n_trials, n_channels, n_times)
    original_events = epochs.events      # shape: (n_trials, 3)
    
    n_trials, n_channels, input_window_samples = original_data.shape
    window_samples = int(round(window_length * sfreq))
    step_samples = int(round(step_size * sfreq))
    
    all_crops = []
    all_events = []
    sub_epoch_counter = 0  # Initialize the sub-epoch counter
    
    unique_labels = np.unique(original_events[:, 2])
    subtract_one = (np.min(unique_labels) == 1 and np.max(unique_labels) == 4)
    
    for i in range(n_trials):
        trial_data = original_data[i]  # shape: (n_channels, n_times)
        raw_label  = original_events[i, 2]
        label_zb = raw_label - 1 if subtract_one else raw_label
        if label_zb < 0:
            raise ValueError(f"Invalid label {label_zb} found. Original was {raw_label}.")
        
        n_sub = 1 + (input_window_samples - window_samples) // step_samples
        for j in range(n_sub):
            start = j * step_samples
            end   = start + window_samples
            crop_data = trial_data[:, start:end]
            all_crops.append(crop_data)
            # Use 'i' (the current trial index) as the trial ID
            evt_row = [sub_epoch_counter, i, label_zb]
            all_events.append(evt_row)
            sub_epoch_counter += 1
    
    all_crops = np.array(all_crops)       # shape: (total_subepochs, n_channels, window_samples)
    all_events = np.array(all_events, int) # shape: (total_subepochs, 3)
    
    new_info = epochs.info.copy()
    new_tmin = 0.0  # Each sub-epoch now starts at 0 seconds relative to its crop
    new_epochs = mne.EpochsArray(
        data=all_crops,
        info=new_info,
        events=all_events,
        tmin=new_tmin,
        baseline=None,
        verbose=False
    )
    print("[DEBUG] Unique sub-epoch labels (should match macro labels):", np.unique(all_events[:, 2]))
    print("[DEBUG] Total number of sub-epochs:", all_crops.shape[0])
    
    # New debug lines:
    print("[DEBUG] First 10 lines of new_epochs.events:")
    print(new_epochs.events[:10])
    
    return new_epochs

def time_lock_and_slide_epochs(raw, tmin, tmax, window_length, step_size):
    epochs, event_id = extract_time_locked_epochs(raw, tmin, tmax)
    new_epochs = force_sliding_window_cropping(epochs, window_length, step_size)
    return new_epochs
