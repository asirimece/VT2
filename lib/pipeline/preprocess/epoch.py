import numpy as np
import mne
from omegaconf import OmegaConf

def create_macro_epochs(raw: mne.io.Raw, dataset_config) -> mne.Epochs:
    """
    Create macro MNE Epochs from tmin to tmax around each event.
    """
    tmin = dataset_config.epoching.kwargs.tmin
    tmax = dataset_config.epoching.kwargs.tmax
    
    events, new_event_id = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(
        raw, events, event_id=new_event_id,
        tmin=tmin, tmax=tmax,
        baseline=None, preload=True, verbose=False
    )
    return epochs

def extract_time_locked_epochs(raw, tmin, tmax):
    tmin = float(OmegaConf.to_container(tmin, resolve=True)) if hasattr(tmin, "keys") else float(tmin)
    tmax = float(OmegaConf.to_container(tmax, resolve=True)) if hasattr(tmax, "keys") else float(tmax)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(raw, events, event_id=event_id,
                        tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose=False)
    return epochs, event_id

def crop_subepochs(epochs, window_length, step_size):
    """
    Create subepochs.
    """
    sfreq = epochs.info['sfreq']
    original_data = epochs.get_data()   
    original_events = epochs.events     
    
    window_samples = int(round(window_length * sfreq))
    step_samples = int(round(step_size * sfreq))
    
    all_crops = []
    all_events = []
    sub_epoch_counter = 0  
    
    unique_labels = np.unique(original_events[:, 2])
    subtract_one = (np.min(unique_labels) == 1 and np.max(unique_labels) == 4)
    
    ##
    # Only allow events 1 and 2 (left_hand and right_hand)
    allowed_labels = [1, 2]
    mask = np.isin(original_events[:, 2], allowed_labels)
    original_data = original_data[mask]
    original_events = original_events[mask]
    
    n_trials, n_channels, n_times = original_data.shape
    ##
    
    for i in range(n_trials):
        trial_data = original_data[i] 
        raw_label  = original_events[i, 2]
        #label_zb = raw_label - 1 if subtract_one else raw_label
        ## 1vs2
        label_map = {1: 0, 2: 1}
        if raw_label not in label_map:
            continue  # Skip unwanted classes
        label_zb = label_map[raw_label]
        ##
        if label_zb < 0:
            raise ValueError(f"Invalid label {label_zb} found. Original was {raw_label}.")
        
        n_sub = 1 + (n_times - window_samples) // step_samples
        for j in range(n_sub):
            start = j * step_samples
            end   = start + window_samples
            crop_data = trial_data[:, start:end]
            all_crops.append(crop_data)
            
            evt_row = [sub_epoch_counter, i, label_zb]
            all_events.append(evt_row)
            sub_epoch_counter += 1
    
    all_crops = np.array(all_crops)      
    all_events = np.array(all_events, int) 
    
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
    epochs, event_id = extract_time_locked_epochs(raw, tmin, tmax)
    new_epochs = crop_subepochs(epochs, window_length, step_size)
    return new_epochs