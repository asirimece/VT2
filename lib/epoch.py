# epoch.py
import numpy as np
import mne

def extract_time_locked_epochs(raw, tmin, tmax):
    """
    Extracts time-locked epochs from raw data using annotations.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data with annotations.
    tmin : float
        Start time (in seconds) relative to each event.
    tmax : float
        End time (in seconds) relative to each event.
        
    Returns
    -------
    epochs : mne.Epochs
        The time-locked epochs.
    event_id : dict
        Mapping from annotation description to event code.
    """
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, preload=True, verbose=False)
    return epochs, event_id

def sliding_window_on_epoch(epoch_data, sfreq, window_length, step_size):
    """
    Splits a single epoch (n_channels x n_times) into overlapping sliding windows.
    
    Parameters
    ----------
    epoch_data : np.ndarray, shape (n_channels, n_times)
        Data for one epoch.
    sfreq : float
        Sampling frequency.
    window_length : float
        Duration (in seconds) of each sliding window.
    step_size : float
        Time (in seconds) between consecutive windows.
        
    Returns
    -------
    windows : np.ndarray, shape (n_windows, n_channels, n_window_samples)
        Array of sliding-window segments.
    """
    window_samples = int(round(window_length * sfreq))
    step_samples = int(round(step_size * sfreq))
    n_times = epoch_data.shape[1]
    windows = []
    for start in range(0, n_times - window_samples + 1, step_samples):
        windows.append(epoch_data[:, start:start + window_samples])
    return np.array(windows)

def time_lock_and_slide_epochs(raw, tmin_event, tmax_event, window_length, step_size, preload=True, strategy='first'):
    """
    Combines time-locking (trial segmentation) and sliding-window cropping.
    
    First, extracts time-locked epochs from the raw data (using annotations),
    then subdivides each epoch into overlapping sub-epochs (crops) using a sliding window.
    
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    tmin_event : float
        Start time (in seconds) relative to each event (e.g., -0.5).
    tmax_event : float
        End time (in seconds) relative to each event (e.g., 3.5).
    window_length : float
        Duration (in seconds) of each sliding-window crop (e.g., 2.0).
    step_size : float
        Time (in seconds) between the start of consecutive crops (e.g., 0.01).
    preload : bool
        Whether to preload the epoch data into memory.
    strategy : str
        (Placeholder for multiple-event strategies; currently only 'first' is implemented.)
        
    Returns
    -------
    new_epochs : mne.EpochsArray
        An EpochsArray containing all the sliding-window crops from all trials.
        Each sub-epoch’s label is inherited from its parent trial.
    """
    # Extract time-locked epochs using annotations
    epochs, event_id = extract_time_locked_epochs(raw, tmin_event, tmax_event)
    sfreq = epochs.info['sfreq']
    
    sub_epochs_list = []
    new_labels = []
    
    # For each trial (epoch), create sliding-window crops
    for i, epoch_data in enumerate(epochs.get_data()):
        n_times = epoch_data.shape[1]
        # Number of sub-epochs that fit in this trial
        n_sub = (n_times - int(round(window_length * sfreq))) // int(round(step_size * sfreq)) + 1
        for j in range(n_sub):
            start = j * int(round(step_size * sfreq))
            sub_epoch = epoch_data[:, start:start + int(round(window_length * sfreq))]
            sub_epochs_list.append(sub_epoch)
            # Inherit the label from the parent trial
            new_labels.append(epochs.events[i, 2])
    
    # Convert list to a NumPy array: shape (total_sub_epochs, n_channels, n_samples)
    sub_epochs_array = np.array(sub_epochs_list)
    
    # Create a new events array for the sub-epochs.
    n_sub_epochs = sub_epochs_array.shape[0]
    new_events = np.zeros((n_sub_epochs, 3), dtype=int)
    for i in range(n_sub_epochs):
        # The sample index is arbitrary here; tmin is set to 0.
        new_events[i, 0] = i * int(round(window_length * sfreq))
        new_events[i, 2] = new_labels[i]
    
    # Copy the info from the original epochs (do not modify nchan manually)
    new_info = epochs.info.copy()
    
    # Create a new EpochsArray from the sub-epoch data.
    new_epochs = mne.EpochsArray(sub_epochs_array, new_info, events=new_events, tmin=0, verbose=False)
    return new_epochs
