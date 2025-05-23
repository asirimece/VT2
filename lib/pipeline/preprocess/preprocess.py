"""
FOR NEW DATASET
"""
import os
import mne
import numpy as np
from omegaconf import DictConfig
from .preprocessors import (
    bandpass_filter,
    exponential_moving_standardization
)
from .epoch import create_macro_epochs, crop_subepochs
from lib.logging import logger

logger = logger.get()


def train_test_event_split(epochs, train_frac=0.7, random_seed=42, shuffle=True):
    """Splits epochs into train and test sets, optionally shuffled for randomness."""
    n_trials = len(epochs)
    indices = np.arange(n_trials)
    if shuffle:
        rng = np.random.default_rng(random_seed)
        rng.shuffle(indices)
    n_train = int(np.floor(n_trials * train_frac))
    train_indices = indices[:n_train]
    test_indices  = indices[n_train:]
    train_epochs = epochs[train_indices]
    test_epochs  = epochs[test_indices]
    return train_epochs, test_epochs


class Preprocessor:
    def __init__(self, config: DictConfig):
        self.config = config.dataset            
        self.preproc_config = self.config.preprocessing 
        self.data_dir = self.config.home

    def run(self) -> dict:
        results = {}
        fif_files = [f for f in os.listdir(self.data_dir) if f.endswith(".fif")]

        # Get split params from config
        split_cfg = self.preproc_config.data_split.kwargs
        train_frac = split_cfg.get('train_fraction', 0.7)
        random_seed = split_cfg.get('random_seed', 42)

        for fname in sorted(fif_files):
            subj = fname.split("_")[2]
            logger.info(f"\n--- Subject: {subj} ---")
            fpath = os.path.join(self.data_dir, fname)

            # Load raw data
            raw = mne.io.read_raw_fif(fpath, preload=True, verbose=False)

            # Bandpass filter if configured (optional: skip if already filtered)
            if self.preproc_config.raw_preprocessors.get('bandpass_filter'):
                raw = bandpass_filter(
                    raw,
                    low=self.preproc_config.raw_preprocessors.bandpass_filter.kwargs.low,
                    high=self.preproc_config.raw_preprocessors.bandpass_filter.kwargs.high,
                    method=self.preproc_config.raw_preprocessors.bandpass_filter.kwargs.method
                )

            # Pick EEG channels (should be only 8, but keep for robustness)
            raw.pick_types(eeg=True, stim=False, exclude=[])

            # Epoching
            macro_epochs = create_macro_epochs(raw, self.preproc_config)
            sub_epochs = crop_subepochs(
                macro_epochs,
                self.preproc_config.epoching.kwargs.crop_window_length,
                self.preproc_config.epoching.kwargs.crop_step_size
            )

            # Split using config-driven logic
            train_epochs, test_epochs = train_test_event_split(
                sub_epochs,
                train_frac=train_frac,
                random_seed=random_seed,
                shuffle=True
            )

            # Standardize (EMS)
            standardized_train, esm_params = exponential_moving_standardization(
                train_epochs,
                smoothing_factor=self.preproc_config.exponential_moving_standardization.kwargs.smoothing_factor,
                eps=self.preproc_config.exponential_moving_standardization.kwargs.eps,
                return_params=True
            )
            standardized_test = exponential_moving_standardization(
                test_epochs,
                smoothing_factor=self.preproc_config.exponential_moving_standardization.kwargs.smoothing_factor,
                eps=self.preproc_config.exponential_moving_standardization.kwargs.eps,
                esm_params=esm_params
            )

            results[subj] = {
                "train": standardized_train,
                "test": standardized_test
            }

        return results

"""
FOR OLD DATASET
"""
"""import os
from matplotlib import pyplot as plt
import numpy as np
import mne
from moabb.datasets import BNCI2014_001
from omegaconf import DictConfig
from .preprocessors import (
    bandpass_filter,
    remove_eog_artifacts_ica,
    data_split_concatenate,
    exponential_moving_standardization
)
from .epoch import crop_subepochs, create_macro_epochs
from lib.logging import logger

logger = logger.get()


class Preprocessor:
    def __init__(self, config: DictConfig):
        self.config = config.dataset            
        self.preproc_config = self.config.preprocessing 

    def run(self) -> dict:
        # Load the dataset via MOABB.
        dataset = BNCI2014_001()
        all_data = dataset.get_data()
        results = {}

        for subj in sorted(all_data.keys()):
            logger.info(f"\n--- Subject: {subj} ---")
            subj_data = all_data[subj]
            train_session = self.preproc_config.data_split.kwargs.train_session
            test_session = self.preproc_config.data_split.kwargs.test_session

            train_raw, test_raw = data_split_concatenate(subj_data, train_session, test_session)

            train_raw = bandpass_filter(
                train_raw,
                low=self.preproc_config.raw_preprocessors.bandpass_filter.kwargs.low,
                high=self.preproc_config.raw_preprocessors.bandpass_filter.kwargs.high,
                method=self.preproc_config.raw_preprocessors.bandpass_filter.kwargs.method
            )

            if self.preproc_config.remove_eog_artifacts:
                train_raw = remove_eog_artifacts_ica(
                    train_raw,
                    eog_ch=self.config.eog_channels,
                    n_components=self.preproc_config.ica.kwargs.n_components,
                    method=self.preproc_config.ica.kwargs.method,
                    random_state=self.preproc_config.ica.kwargs.random_state,
                    show_ica_plots=self.preproc_config.show_ica_plots,
                    save_ica_plots=self.preproc_config.save_ica_plots,
                    plots_output_dir=self.preproc_config.ica_plots_dir
                )

            # Select only EEG channels.
            train_raw.pick_types(eeg=True, stim=False, exclude=[])
            train_macro_epochs = create_macro_epochs(train_raw, self.preproc_config)
            train_sub_epochs = crop_subepochs(
                train_macro_epochs,
                self.preproc_config.epoching.kwargs.crop_window_length,
                self.preproc_config.epoching.kwargs.crop_step_size
            )

            standardized_train, esm_params = exponential_moving_standardization(
                train_sub_epochs,
                smoothing_factor=self.preproc_config.exponential_moving_standardization.kwargs.smoothing_factor,
                eps=self.preproc_config.exponential_moving_standardization.kwargs.eps,
                return_params=True
            )

            test_raw = bandpass_filter(
                test_raw,
                low=self.preproc_config.raw_preprocessors.bandpass_filter.kwargs.low,
                high=self.preproc_config.raw_preprocessors.bandpass_filter.kwargs.high,
                method=self.preproc_config.raw_preprocessors.bandpass_filter.kwargs.method
            )

            if self.preproc_config.remove_eog_artifacts:
                test_raw = remove_eog_artifacts_ica(
                    test_raw,
                    eog_ch=self.config.eog_channels,
                    n_components=self.preproc_config.ica.kwargs.n_components,
                    method=self.preproc_config.ica.kwargs.method,
                    random_state=self.preproc_config.ica.kwargs.random_state,
                    show_ica_plots=self.preproc_config.show_ica_plots,
                    save_ica_plots=self.preproc_config.save_ica_plots,
                    plots_output_dir=self.preproc_config.ica_plots_dir
                )

            test_raw.pick_types(eeg=True, stim=False, exclude=[])
            test_macro_epochs = create_macro_epochs(test_raw, self.preproc_config)
            test_sub_epochs = crop_subepochs(
                test_macro_epochs,
                self.preproc_config.epoching.kwargs.crop_window_length,
                self.preproc_config.epoching.kwargs.crop_step_size
            )

            standardized_test = exponential_moving_standardization(
                test_sub_epochs,
                smoothing_factor=self.preproc_config.exponential_moving_standardization.kwargs.smoothing_factor,
                eps=self.preproc_config.exponential_moving_standardization.kwargs.eps,
                esm_params=esm_params
            )
            
            results[subj] = {
                train_session: standardized_train,
                test_session: standardized_test
            }

        return results
"""