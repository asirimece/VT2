# lib/pipeline.py

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
    tmin_event = epoch_kwargs.get('tmin', -0.5)
    tmax_event = epoch_kwargs.get('tmax', 3.5)
    
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
            raw.pick_types(eeg=True, stim=True, exclude=[])
            
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
