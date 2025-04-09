"""
pipeline.py

This module defines a Preprocessor class that:
1. Loadataset BNCI2014001 data using MOABB.
2. Applies bandpass filtering.
3. Optionally removes EOG artifacts using ICA.
4. Creates macro epochs (e.g. [tmin_event, tmax_event] seconds).
5. Applies sliding-window cropping to create sub-epochs.
6. Optionally applies exponential moving standardization.
7. Saves the resulting preprocessed data to disk.

Usage:
    preprocessor = Preprocessor(config)
    results = preprocessor.run()
"""

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
    """
    Run the complete preprocessing pipeline:
    1. Load dataset using BNCI2014001.
    2. For each subject, process train/test sessions:
        - Apply bandpass filter, optionally remove EOG artifacts, and pick EEG channels.
        - Create macro epochs from the raw data.
        - Apply sliding-window cropping to get sub-epochs.
        - Optionally apply exponential moving standardization.
    3. Save and return the preprocessed data.
    """
    def __init__(self, config: DictConfig):
        self.config = config
        self.dataset_config = config.dataset.preprocessing

    def run(self) -> dict:
        # Load dataset from MOABB.
        dataset = BNCI2014_001()
        all_data = dataset.get_data()
        results = {}

        for subj in sorted(all_data.keys()):
            logger.info(f"\n--- Subject: {subj} ---")
            subj_data = all_data[subj]
            # Concatenate runs for train/test sessions
            train_raw, test_raw = data_split_concatenate(subj_data, self.dataset_config.data_split.kwargs.train_session, self.dataset_config.data_split.kwargs.test_session)
            for sess_label, raw in zip([self.dataset_config.data_split.kwargs.train_session, self.dataset_config.data_split.kwargs.test_session], [train_raw, test_raw]):
                logger.info(f"Processing session: {sess_label}")
                
                # Bandpass filter
                raw = bandpass_filter(raw, low=self.dataset_config.raw_preprocessors.bandpass_filter.kwargs.low, high=self.dataset_config.raw_preprocessors.bandpass_filter.kwargs.high, method=self.dataset_config.raw_preprocessors.bandpass_filter.kwargs.method)
                
                # Remove EOG artifacts
                if self.dataset_config.remove_eog_artifacts:
                    raw = remove_eog_artifacts_ica(
                        raw, 
                        eog_ch=self.config.dataset.eog_channels,
                        n_components=22, 
                        method='fastica',
                        random_state=42, 
                        show_ica_plots=False,
                        save_ica_plots=False
                    )
                
                # Pick EEG channels only
                raw.pick_types(eeg=True, stim=False, exclude=[])
                
                # Create macro epochs
                macro_epochs = create_macro_epochs(raw, self.dataset_config)
                logger.info(f"  Created macro epochs: shape={macro_epochs.get_data().shape}, tmin={macro_epochs.tmin}, tmax={macro_epochs.tmax}")
                
                # Apply sliding-window cropping to get sub-epochs
                sub_epochs = crop_subepochs(macro_epochs, self.dataset_config.epoching.kwargs.crop_window_length, self.dataset_config.epoching.kwargs.crop_step_size)
                logger.info(f"  After sliding, sub-epochs shape: {sub_epochs.get_data().shape}, tmin={sub_epochs.tmin}, tmax={sub_epochs.tmax}")
                
                # Apply exponential moving standardization.
                sub_epochs = exponential_moving_standardization(sub_epochs, smoothing_factor=self.dataset_config.exponential_moving_standardization.kwargs.smoothing_factor)
                logger.info(f"  Applied exponential moving standardization on session: {sess_label}")
                
                # Store subepochs
                if subj not in results:
                    results[subj] = {}
                results[subj][sess_label] = sub_epochs

        return results
