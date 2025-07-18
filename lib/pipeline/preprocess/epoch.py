"""
FOR NEW DATASET
"""
# epoch.py
import numpy as np
import mne
from lib.logging import logger

logger = logger.get()

def _build_event_id(dataset_config):
    """
    From dataset_config.event_markers (which has keys both
    like '0': 'left_hand' and left_hand: 0), build a dict
    {'0': 0, '1': 1} including only supervised.classes.
    """
    em = dataset_config.event_markers

    # 1) map annotation‐string → event‐name, e.g. '0' → 'left_hand'
    desc2name = {
        k: v for k, v in em.items()
        if isinstance(v, str) and k.isdigit()
    }
    # 2) map event‐name → integer code,  e.g. 'left_hand' → 0
    name2code = {
        k: v for k, v in em.items()
        if isinstance(v, int)
    }

    sup = dataset_config.supervised
    keep = set(sup.classes)
    drop = set(sup.ignore_labels)

    event_id = {}
    for desc, name in desc2name.items():
        code = name2code.get(name)
        if code is None:
            continue
        if code in keep and code not in drop:
            event_id[desc] = code
    return event_id

def create_macro_epochs(raw: mne.io.Raw, dataset_config) -> mne.Epochs:
    """
    Create macro MNE Epochs from tmin to tmax around each event,
    using exactly the event_id mapping from your config.
    """
    tmin = float(dataset_config.preprocessing.epoching.kwargs.tmin)
    tmax = float(dataset_config.preprocessing.epoching.kwargs.tmax)

    # DEBUG: log raw annotations
    logger.info(f"RAW ANNOTATIONS: {raw.annotations}")
    for desc in np.unique(raw.annotations.description):
        cnt = np.sum(raw.annotations.description == desc)
        logger.info(f"    description={desc!r}, count={cnt}")

    # build your exact event_id
    event_id = _build_event_id(dataset_config)
    logger.info(f"Using event_id={event_id!r}")

    # let MNE pull only those annotations
    events, _ = mne.events_from_annotations(
        raw,
        event_id=event_id,
        verbose=False
    )
    logger.info(f" all raw event codes in data = {np.unique(events[:, -1])}")

    return mne.Epochs(
        raw, events,
        event_id=event_id,
        tmin=tmin, tmax=tmax,
        baseline=None, preload=True, verbose=False
    )

def crop_subepochs(epochs: mne.Epochs,
                   dataset_config,
                   window_length: float,
                   step_size: float) -> mne.EpochsArray:
    """
    Slide-window cropping of a macro-epoch into sub-epochs.
    Remaps raw codes → 0..N-1 in the order of keep_codes.
    """
    # same keep_codes as in macro
    keep_codes = tuple(sorted(_build_event_id(dataset_config).values()))

    sfreq = epochs.info['sfreq']
    data  = epochs.get_data()    # shape (n_epochs, n_ch, n_times)
    events= epochs.events       # array [[idx, raw_trial, code], ...]

    # only keep the codes we built
    mask   = np.isin(events[:, -1], keep_codes)
    data   = data[mask]
    events = events[mask]

    w_samps = int(round(window_length * sfreq))
    s_samps = int(round(step_size   * sfreq))

    # prepare a mapping code → 0..(len-1)
    label_map = {code: idx for idx, code in enumerate(keep_codes)}

    all_crops = []
    all_events= []
    counter   = 0

    for trial_idx, raw_label in enumerate(events[:, -1]):
        mapped = label_map[raw_label]
        trial_data = data[trial_idx]  # shape (n_ch, n_times)
        n_sub = 1 + (trial_data.shape[1] - w_samps)//s_samps
        for j in range(n_sub):
            start = j*s_samps
            end   = start + w_samps
            if end > trial_data.shape[1]:
                break
            all_crops.append(trial_data[:, start:end])
            # events for EpochsArray: [new_epoch_id, original_epoch_idx, new_label]
            all_events.append([counter, trial_idx, mapped])
            counter += 1

    all_crops = np.stack(all_crops)           # (n_subepochs, n_ch, w_samps)
    all_events= np.array(all_events, int)      # (n_subepochs, 3)

    new_info = epochs.info.copy()
    return mne.EpochsArray(
        data=all_crops,
        info=new_info,
        events=all_events,
        tmin=0.0,
        baseline=None,
        verbose=False
    )

def time_lock_and_slide_epochs(raw, dataset_config):
    """
    Convenience: do both steps in one call.
    """
    pre = create_macro_epochs(raw, dataset_config)
    wl = dataset_config.preprocessing.epoching.kwargs.crop_window_length
    ss = dataset_config.preprocessing.epoching.kwargs.crop_step_size
    return crop_subepochs(pre, dataset_config, wl, ss)


"""
FOR OLD DATASET
"""
"""
import numpy as np
import mne
from omegaconf import OmegaConf

def create_macro_epochs(raw: mne.io.Raw, dataset_config) -> mne.Epochs:

    #Create macro MNE Epochs from tmin to tmax around each event.
    #(Unchanged—kept here in case you still need it elsewhere.)

    tmin = dataset_config.epoching.kwargs.tmin
    tmax = dataset_config.epoching.kwargs.tmax
    
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=tmin, tmax=tmax,
        baseline=None, preload=True, verbose=False
    )
    return epochs

def extract_time_locked_epochs(raw, tmin, tmax, keep_codes=(1,2,4)):
    
    #Extract only epochs for left hand (1), right hand (2), and tongue (4) events.
    
    tmin = float(OmegaConf.to_container(tmin, resolve=True)) if hasattr(tmin, "keys") else float(tmin)
    tmax = float(OmegaConf.to_container(tmax, resolve=True)) if hasattr(tmax, "keys") else float(tmax)

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    event_id = {name: code for name, code in event_id.items() if code in keep_codes}
    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=tmin, tmax=tmax,
        baseline=None, preload=True, verbose=False
    )
    return epochs

def crop_subepochs(epochs, window_length, step_size, keep_codes=(1,2,4)):
    
    #Crops epochs to subepochs and remaps event codes 1/2/4 → 0/1/2.
    
    sfreq = epochs.info['sfreq']
    data = epochs.get_data()
    events = epochs.events

    mask = np.isin(events[:, 2], keep_codes)
    data = data[mask]
    events = events[mask]

    window_samples = int(round(window_length * sfreq))
    step_samples   = int(round(step_size   * sfreq))

    all_crops = []
    all_events = []
    counter   = 0

    # Map 1→0, 2→1, 4→2
    label_map = {code: idx for idx, code in enumerate(keep_codes)}

    for trial_idx, raw_label in enumerate(events[:, 2]):
        mapped_label = label_map[raw_label]
        trial_data   = data[trial_idx]
        n_sub        = 1 + (trial_data.shape[1] - window_samples) // step_samples

        for j in range(n_sub):
            start = j * step_samples
            end   = start + window_samples
            if end > trial_data.shape[1]:
                break
            all_crops.append(trial_data[:, start:end])
            all_events.append([counter, trial_idx, mapped_label])
            counter += 1

    all_crops  = np.stack(all_crops)
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

def time_lock_and_slide_epochs(raw, tmin, tmax, window_length, step_size, keep_codes=(1,2,4)):
    
    #Combines steps for 3-class MI (left hand, right hand, tongue)
    
    epochs = extract_time_locked_epochs(raw, tmin, tmax, keep_codes=keep_codes)
    new_epochs = crop_subepochs(epochs, window_length, step_size, keep_codes=keep_codes)
    return new_epochs
"""