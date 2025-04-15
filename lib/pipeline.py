"""
pipeline.py
1) Loads BNCI2014_001 data using MOABB.
2) Bandpass-filters the data.
3) (Optionally) removes EOG artifacts with ICA.
4) Creates 4 s “macro” epochs from tmin..tmax (e.g. [2.0, 6.0] s).
5) Applies sliding-window cropping (subepoching) to create 2 s sub-epochs.
6) Saves the resulting sub-epochs (i.e. preprocessed_data.pkl) to disk.
"""
import os
import pickle
import mne
from omegaconf import OmegaConf
from moabb.datasets import BNCI2014001
from lib.preprocessors import exponential_moving_standardization
from lib.epoch import force_sliding_window_cropping
from lib.dataset.events import unify_events
from lib.preprocessors import bandpass_filter, remove_eog_artifacts_ica, data_split_concatenate
from lib.utils.utils import to_float

def time_lock_epochs(raw, tmin, tmax, unify_annot=None, event_markers=None):
    """
    Create macro MNE Epochs from tmin to tmax around each event.
    If unify_annot and event_markers are provided, unify the events first.
    """
    # Resolve tmin and tmax as floats if needed (using your helper to_float)
    tmin_val = to_float(tmin, 2.0)
    tmax_val = to_float(tmax, 6.0)
    
    if unify_annot is not None and event_markers is not None:
        # Use our custom event unification:
        events, _ = mne.events_from_annotations(raw, verbose=False)
        # (Alternatively, you might pass raw directly to unify_events)
        events, new_event_id = unify_events(raw, unify_annot, event_markers)
    else:
        events, new_event_id = mne.events_from_annotations(raw, verbose=False)

    epochs = mne.Epochs(
        raw, events, event_id=new_event_id,
        tmin=tmin_val, tmax=tmax_val,
        baseline=None, preload=True, verbose=False
    )
    return epochs

def save_preprocessed_data(results, filename="./outputs/preprocessed_data.pkl"):
    out_dir = os.path.dirname(filename)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Preprocessed data saved to {filename}")

def run_preprocessing_pipeline(cfg,
                                tmin_event=2.0,
                                tmax_event=6.0,
                                low=4.0,
                                high=38.0,
                                method='iir',
                                remove_eog=True,
                                eog_ch_list=['EOG1', 'EOG2', 'EOG3'],
                                train_session="0train",
                                test_session="1test",
                                apply_ems=True,
                                smoothing_factor=0.1,
                                window_length_sub=2.0,  # length (in seconds) of sub-epochs
                                step_size_sub=0.5,      # step size (in seconds) for sliding window
                                output_file="./outputs/preprocessed_data.pkl"
                            ):
    # 1) Load dataset
    dataset = BNCI2014001()
    all_data = dataset.get_data()

    results = {}
    # Get event mapping dictionaries from the config (assumed to be under cfg.dataset)
    #config_unify_annot = cfg.dataset.get("unify_annotations", None)
    config_event_markers = cfg.get("event_markers", None)
    
    for subj in sorted(all_data.keys()):
        print(f"\n--- Subject: {subj} ---")
        subj_data = all_data[subj]

        # 2) Concatenate runs for train/test sessions
        train_raw, test_raw = data_split_concatenate(subj_data, train_session, test_session)

        for sess_label, raw in zip([train_session, test_session], [train_raw, test_raw]):
            print(f"Processing session: {sess_label}")

            # 3) Bandpass filtering
            raw = bandpass_filter(raw, low=low, high=high, method=method)

            # 4) Optional: Remove EOG artifacts
            if remove_eog:
                raw = remove_eog_artifacts_ica(raw, eog_ch=eog_ch_list,
                                               n_components=22, method='fastica',
                                               random_state=42, show_ica_plots=False,
                                               save_ica_plots=False)

            # 5) Pick EEG channels only
            raw.pick_types(eeg=True, stim=False, exclude=[])
            
            # 6) Create macro epochs (e.g. 2–6 s)
            """macro_epochs = time_lock_epochs(raw, tmin=tmin_event, tmax=tmax_event,
                                            unify_annot=config_unify_annot,
                                            event_markers=config_event_markers)"""
            macro_epochs = time_lock_epochs(raw, tmin=tmin_event, tmax=tmax_event,
                                            event_markers=config_event_markers)
            print(f"  Created macro epochs: shape={macro_epochs.get_data().shape}, "
                  f"tmin={macro_epochs.tmin}, tmax={macro_epochs.tmax}")

            # 7) Apply sliding-window cropping to obtain sub-epochs
            sub_epochs = force_sliding_window_cropping(macro_epochs, window_length_sub, step_size_sub)
            print(f"  After sliding, sub-epochs shape: {sub_epochs.get_data().shape}, "
                  f"tmin={sub_epochs.tmin}, tmax={sub_epochs.tmax}")

            # 8) Apply exponential moving standardization on training set only
            #if apply_ems and sess_label == train_session:
            if apply_ems:
                # Here EMS is applied on the sub-epochs.
                # (Assume exponential_moving_standardization is defined in pipeline.py)
                sub_epochs = exponential_moving_standardization(sub_epochs, smoothing_factor=smoothing_factor)
                print(f"  Applied exponential moving standardization on session: {sess_label}")

            # Store the processed sub-epochs
            if subj not in results:
                results[subj] = {}
            results[subj][sess_label] = sub_epochs

    save_preprocessed_data(results, output_file)
    return results

"""if __name__ == "__main__":
    run_preprocessing_pipeline(
        tmin_event=2.0,
        tmax_event=6.0,
        low=4.0,
        high=38.0,
        remove_eog=True,
        output_file="./outputs/preprocessed_data.pkl",
        window_length_sub=2.0,
        step_size_sub=0.5
    )"""
    
if __name__ == "__main__":
    cfg = OmegaConf.load("config/config.yaml")
    _ = run_preprocessing_pipeline(cfg)
    



"""# lib/pipeline.py

import os
import pickle
import mne
from moabb.datasets import BNCI2014001

from lib.preprocessors import (apply_notch_filter, bandpass_filter,
                               data_split_concatenate, exponential_moving_standardization,
                               remove_eog_artifacts_ica)

def run_preprocessing_pipeline(config):
    dataset = BNCI2014001()
    all_data = dataset.get_data()
    
    results = {}
    
    # Retrieve relevant config options
    train_session = config.dataset.preprocessing.data_split.kwargs.train_session
    test_session = config.dataset.preprocessing.data_split.kwargs.test_session
    
    bp_kwargs = config.dataset['preprocessing']['raw_preprocessors']['bandpass_filter']['kwargs']
    low = bp_kwargs.get('low', 4)
    high = bp_kwargs.get('high', 38)
    method = bp_kwargs.get('method', 'iir')
    
    remove_eog = config.dataset['preprocessing'].get('remove_eog_artifacts', False)
    show_ica_plots = config.dataset['preprocessing'].get('show_ica_plots', False)
    save_ica_plots = config.dataset['preprocessing'].get('save_ica_plots', False)
    ica_plots_dir = config.dataset['preprocessing'].get('ica_plots_dir', "./ica_plots")
    
    epoch_kwargs = config.dataset['preprocessing']['epoching']['kwargs']
    window_length = epoch_kwargs.get('window_length', 4.0)
    step_size = epoch_kwargs.get('step_size', 0.5)
    tmin_event = epoch_kwargs.get('tmin', 2)
    tmax_event = epoch_kwargs.get('tmax', 6)
    
    ems_kwargs = config.dataset['preprocessing']['exponential_moving_standardization']['kwargs']
    smoothing_factor = ems_kwargs.get('smoothing_factor', 0.1)
    
    for subj in sorted(all_data.keys()):
        print(f"\n--- Subject: {subj} ---")
        subj_data = all_data[subj]
        subj_results = {}
        
        train_raw, test_raw = data_split_concatenate(subj_data, train_session, test_session)
        
        for sess_label, raw in zip([train_session, test_session], [train_raw, test_raw]):
            print(f"Processing session: {sess_label}")
            
            # Optional: Notch
            # raw = apply_notch_filter(raw, notch_freq=50)

            # Bandpass
            raw = bandpass_filter(raw, low=low, high=high, method=method)
            print(f"  After bandpass: {raw}")
            
            # EOG artifact removal if enabled
            if remove_eog:
                raw = remove_eog_artifacts_ica(
                    raw,
                    eog_ch=['EOG1','EOG2','EOG3'],
                    n_components=22, 
                    method='fastica', 
                    random_state=42,
                    show_ica_plots=show_ica_plots,
                    save_ica_plots=save_ica_plots,
                    plots_output_dir=ica_plots_dir,
                    subj_id=subj,
                    sess_label=sess_label
                )
            
            # Keep only EEG + stim
            raw.pick_types(eeg=True, stim=False, exclude=[])
            
            # Epoch with sliding window
            from lib.epoch import time_lock_and_slide_epochs
            epochs = time_lock_and_slide_epochs(raw, tmin_event, tmax_event, window_length, step_size)
            print(f"  Created {len(epochs.events)} epochs from session {sess_label}")
            
            # Exponential moving standardization (train only)
            if sess_label == train_session:
                epochs = exponential_moving_standardization(epochs, smoothing_factor=smoothing_factor)
                print("  Applied exponential moving standardization on training data.")
            
            subj_results[sess_label] = epochs
        
        results[subj] = subj_results
    
    return results

def save_preprocessed_data(results, filename="./preprocessed_data.pkl"):
    out_dir = os.path.dirname(filename)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Preprocessed data saved to {filename}")
"""