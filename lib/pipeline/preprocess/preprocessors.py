import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from lib.logging import logger

logger = logger.get()


def bandpass_filter(raw, low, high, method):
    raw_filtered = raw.copy().filter(l_freq=low, h_freq=high, method=method, verbose=False)
    return raw_filtered


def apply_notch_filter(raw, notch_freq):
    raw_notched = raw.copy().notch_filter(freqs=[notch_freq], verbose=False)
    return raw_notched


def remove_eog_artifacts_ica(raw, eog_ch, n_components, method, random_state,
                             show_ica_plots, save_ica_plots, plots_output_dir,
                             subj_id=None, sess_label=None):
    """
    Remove EOG artifacts using ICA.
    """
    raw.load_data()

    if save_ica_plots:
        abs_plots_output_dir = os.path.abspath(plots_output_dir)
        os.makedirs(abs_plots_output_dir, exist_ok=True)
    else:
        abs_plots_output_dir = plots_output_dir

    ica = mne.preprocessing.ICA(n_components=n_components,
                                method=method,
                                random_state=random_state,
                                verbose=False)
    ica.fit(raw)

    eog_inds_total = []
    for ch in eog_ch:
        if ch in raw.ch_names:
            eog_inds, _ = ica.find_bads_eog(raw, ch_name=ch)
            eog_inds_total.extend(eog_inds)
            logger.info(f"[ICA] Detected EOG components {eog_inds} correlating with channel {ch}.")
    eog_inds_total = list(set(eog_inds_total))
    logger.info(f"[ICA] Total EOG-related components to exclude: {eog_inds_total}")
    ica.exclude.extend(eog_inds_total)

    fig_label = f"subj{subj_id}_{sess_label}" if subj_id else ""

    figs_components = ica.plot_components(show=show_ica_plots)
    if save_ica_plots and figs_components:
        if isinstance(figs_components, list):
            for i, fig in enumerate(figs_components):
                out_path = os.path.join(abs_plots_output_dir, f"ica_components_{fig_label}_page{i}.png")
                fig.savefig(out_path)
        else:
            out_path = os.path.join(abs_plots_output_dir, f"ica_components_{fig_label}.png")
            figs_components.savefig(out_path)

    fig_sources = ica.plot_sources(raw, show=show_ica_plots)
    if save_ica_plots and fig_sources:
        out_path = os.path.join(abs_plots_output_dir, f"ica_sources_{fig_label}.png")
        fig_sources.savefig(out_path)

    ica.apply(raw)
    logger.info(f"[ICA] Applied ICA. Excluded {len(ica.exclude)} components.")

    return raw


def data_split_concatenate(subj_data, train_session, test_session):
    train_data = subj_data.get(train_session)
    if isinstance(train_data, dict):
        runs = list(train_data.values())
        if len(runs) > 1:
            train_raw = mne.concatenate_raws(runs)
        else:
            train_raw = runs[0]
    else:
        train_raw = train_data

    test_data = subj_data.get(test_session)
    if isinstance(test_data, dict):
        runs = list(test_data.values())
        if len(runs) > 1:
            test_raw = mne.concatenate_raws(runs)
        else:
            test_raw = runs[0]
    else:
        test_raw = test_data

    return train_raw, test_raw


def exponential_moving_standardization(epochs, smoothing_factor, eps, esm_params=None, return_params=False):
    data = epochs.get_data()
    n_epochs, n_channels, n_times = data.shape
    standardized_data = np.zeros_like(data)

    if esm_params is None:
        initial_means = np.mean(data[:, :, 0], axis=0)
        initial_vars = np.var(data, axis=(0, 2))
        esm_params = {"initial_means": initial_means, "initial_vars": initial_vars}

    initial_means = esm_params["initial_means"]
    initial_vars = esm_params["initial_vars"]

    for ep in range(n_epochs):
        for ch in range(n_channels):
            x = data[ep, ch, :]
            running_mean = initial_means[ch]
            running_var = initial_vars[ch]
            standardized_signal = np.zeros_like(x)
            for t in range(len(x)):
                if t == 0:
                    running_mean = initial_means[ch]
                    running_var = initial_vars[ch]
                else:
                    running_mean = smoothing_factor * x[t] + (1 - smoothing_factor) * running_mean
                    running_var = smoothing_factor * ((x[t] - running_mean) ** 2) + (1 - smoothing_factor) * running_var
                standardized_signal[t] = (x[t] - running_mean) / np.sqrt(running_var + eps)
            standardized_data[ep, ch, :] = standardized_signal

    standardized_epochs = epochs.copy().load_data()
    standardized_epochs._data = standardized_data
    if return_params:
        return standardized_epochs, esm_params
    return standardized_epochs
