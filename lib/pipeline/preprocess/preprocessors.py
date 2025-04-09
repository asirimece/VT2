# preprocessors.py
import os
from lib.logging import logger
import mne
import numpy as np
import matplotlib.pyplot as plt

logger = logger.get()

def bandpass_filter(raw, low=4, high=38, method='iir'):
    raw_filtered = raw.copy().filter(l_freq=low, h_freq=high, method=method, verbose=False)
    return raw_filtered

def apply_notch_filter(raw, notch_freq=50):
    raw_notched = raw.copy().notch_filter(freqs=[notch_freq], verbose=False)
    return raw_notched

def remove_eog_artifacts_ica(raw, 
                             eog_ch=('EOG1','EOG2','EOG3'),
                             n_components=22, 
                             method='fastica', 
                             random_state=42,
                             show_ica_plots=False,
                             save_ica_plots=False,
                             plots_output_dir="./ica_plots",
                             subj_id=None,
                             sess_label=None):

    raw.load_data()

    if save_ica_plots:
        abs_plots_output_dir = os.path.abspath(plots_output_dir)
        os.makedirs(abs_plots_output_dir, exist_ok=True)
    else:
        abs_plots_output_dir = plots_output_dir  # Not used if not saving

    # Create and fit ICA
    ica = mne.preprocessing.ICA(n_components=n_components, 
                                method=method,
                                random_state=random_state, 
                                verbose=False)
    print(f"[ICA] Fitting ICA with {n_components} components (method={method}).")
    ica.fit(raw)
    
    # Identify EOG components
    eog_inds_total = []
    for ch in eog_ch:
        if ch in raw.ch_names:
            eog_inds, _scores = ica.find_bads_eog(raw, ch_name=ch)
            eog_inds_total.extend(eog_inds)
            print(f"[ICA] Detected EOG components {eog_inds} correlating with channel {ch}.")
    eog_inds_total = list(set(eog_inds_total))
    print(f"[ICA] Total EOG-related components to exclude: {eog_inds_total}")
    ica.exclude.extend(eog_inds_total)
    
    fig_label = f"subj{subj_id}_{sess_label}" if subj_id else ""
    
    figs_components = ica.plot_components(show=show_ica_plots)
    
    # Save ICA components plots if enabled
    if save_ica_plots and figs_components:
        if isinstance(figs_components, list):
            for i, fig in enumerate(figs_components):
                out_path = os.path.join(abs_plots_output_dir, f"ica_components_{fig_label}_page{i}.png")
                fig.savefig(out_path)
                logger.info(f"Saved ICA components figure to {out_path}")
        else:
            out_path = os.path.join(abs_plots_output_dir, f"ica_components_{fig_label}.png")
            figs_components.savefig(out_path)
            logger.info(f"Saved ICA components figure to {out_path}")
    
    fig_sources = ica.plot_sources(raw, show=show_ica_plots)
    if save_ica_plots and fig_sources:
        out_path = os.path.join(abs_plots_output_dir, f"ica_sources_{fig_label}.png")
        fig_sources.savefig(out_path)
        logger.info(f"Saved ICA sources figure to {out_path}")
    
    # Apply ICA to remove EOG artifacts
    ica.apply(raw)
    logger.info(f"[ICA] Applied ICA. Excluded {len(ica.exclude)} components.")
    
    return raw


def data_split_concatenate(subj_data, train_session="0train", test_session="1test"):
    """
    If a session contains multiple runs, all runs are concatenated.
    """
    train_data = subj_data.get(train_session)
    if isinstance(train_data, dict):
        runs = list(train_data.values())
        if len(runs) > 1:
            logger.info(f"Found multiple runs {list(train_data.keys())} for training session '{train_session}'. Concatenating all runs.")
            train_raw = mne.concatenate_raws(runs)
        else:
            train_raw = runs[0]
    else:
        train_raw = train_data
        
    test_data = subj_data.get(test_session)
    if isinstance(test_data, dict):
        runs = list(test_data.values())
        if len(runs) > 1:
            logger.info(f"Found multiple runs {list(test_data.keys())} for testing session '{test_session}'. Concatenating all runs.")
            test_raw = mne.concatenate_raws(runs)
        else:
            test_raw = runs[0]
    else:
        test_raw = test_data

    print("[DEBUG] Training session events summary:")
    print(mne.events_from_annotations(train_raw)[0][:5])
    print("[DEBUG] Testing session events summary:")
    print(mne.events_from_annotations(test_raw)[0][:5])

    return train_raw, test_raw

def exponential_moving_standardization(epochs, smoothing_factor=0.1, eps=1e-5):
    data = epochs.get_data()
    n_epochs, n_channels, n_times = data.shape
    standardized_data = np.zeros_like(data)
    
    for ep in range(n_epochs):
        for ch in range(n_channels):
            x = data[ep, ch, :]
            running_mean = x[0]
            running_var = 0.0
            standardized_signal = np.zeros_like(x)
            for t in range(len(x)):
                if t == 0:
                    running_mean = x[t]
                    running_var = 0.0
                else:
                    running_mean = smoothing_factor * x[t] + (1 - smoothing_factor) * running_mean
                    running_var = smoothing_factor * (x[t] - running_mean) ** 2 + (1 - smoothing_factor) * running_var
                standardized_signal[t] = (x[t] - running_mean) / np.sqrt(running_var + eps)
            standardized_data[ep, ch, :] = standardized_signal

    standardized_epochs = epochs.copy().load_data()
    standardized_epochs._data = standardized_data
    return standardized_epochs
