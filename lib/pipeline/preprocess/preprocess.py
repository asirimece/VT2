import os
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

            # Process training session
            logger.info(f"Processing training session: {train_session}")
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
            logger.info(f"Macro epochs: shape={train_macro_epochs.get_data().shape}, tmin={train_macro_epochs.tmin}, tmax={train_macro_epochs.tmax}")
            train_sub_epochs = crop_subepochs(
                train_macro_epochs,
                self.preproc_config.epoching.kwargs.crop_window_length,
                self.preproc_config.epoching.kwargs.crop_step_size
            )
            logger.info(f"Sub-epochs: shape={train_sub_epochs.get_data().shape}")

            standardized_train, esm_params = exponential_moving_standardization(
                train_sub_epochs,
                smoothing_factor=self.preproc_config.exponential_moving_standardization.kwargs.smoothing_factor,
                eps=self.preproc_config.exponential_moving_standardization.kwargs.eps,
                return_params=True
            )
            logger.info("Applied ESM on training data and extracted parameters.")

            # Process test session
            logger.info(f"Processing test session: {test_session}")
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
            logger.info(f"Test macro epochs: shape={test_macro_epochs.get_data().shape}, tmin={test_macro_epochs.tmin}, tmax={test_macro_epochs.tmax}")
            test_sub_epochs = crop_subepochs(
                test_macro_epochs,
                self.preproc_config.epoching.kwargs.crop_window_length,
                self.preproc_config.epoching.kwargs.crop_step_size
            )
            logger.info(f"Test sub-epochs: shape={test_sub_epochs.get_data().shape}")

            standardized_test = exponential_moving_standardization(
                test_sub_epochs,
                smoothing_factor=self.preproc_config.exponential_moving_standardization.kwargs.smoothing_factor,
                eps=self.preproc_config.exponential_moving_standardization.kwargs.eps,
                esm_params=esm_params
            )
            logger.info("Applied ESM on test data using training parameters.")
            
            results[subj] = {
                train_session: standardized_train,
                test_session: standardized_test
            }

        return results
