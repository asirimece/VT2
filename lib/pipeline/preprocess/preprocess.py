"""
FOR NEW DATASET
"""
import os
import mne
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
from .preprocessors import bandpass_filter, exponential_moving_standardization
from .epoch import create_macro_epochs, crop_subepochs
from lib.logging import logger

logger = logger.get()

def train_test_event_split(epochs, train_frac=0.7, random_seed=42):
    """
    Stratified split of MNE Epochs (whole trials) so that
    no trial appears in both train and test.
    """
    idx = np.arange(len(epochs))
    y   = epochs.events[:, -1]
    train_idx, test_idx = train_test_split(
        idx,
        train_size=train_frac,
        random_state=random_seed,
        stratify=y
    )
    return epochs[train_idx], epochs[test_idx]

class Preprocessor:
    def __init__(self, config: DictConfig):
        self.config         = config.dataset
        self.preproc_config = self.config.preprocessing
        self.raw_preproc    = self.preproc_config.raw_preprocessors or {}
        self.data_dir       = self.config.home

    def run(self) -> dict:
        results    = {}
        fif_files  = sorted(f for f in os.listdir(self.data_dir) if f.endswith(".fif"))
        split_cfg  = self.preproc_config.data_split.kwargs
        train_frac = split_cfg.get('train_fraction', 0.7)
        random_seed= split_cfg.get('random_seed', 42)

        pbar = tqdm(fif_files, desc="Preprocessing subjects", unit="subj")
        for fname in pbar:
            subj = fname.split("_")[2]
            pbar.set_postfix_str(f"subject={subj}")
            raw = mne.io.read_raw_fif(os.path.join(self.data_dir, fname),
                                      preload=True, verbose=False)

            # 1) optional resample / filter
            res_cfg = self.raw_preproc.get("resample")
            if res_cfg:
                raw.resample(res_cfg.sfreq, npad=res_cfg.get("npad","auto"),
                             verbose=False)
            if self.preproc_config.raw_preprocessors.get('bandpass_filter'):
                raw = bandpass_filter(raw,
                                      low=self.preproc_config.raw_preprocessors
                                                           .bandpass_filter.kwargs.low,
                                      high=self.preproc_config.raw_preprocessors
                                                           .bandpass_filter.kwargs.high,
                                      method=self.preproc_config.raw_preprocessors
                                                           .bandpass_filter.kwargs.method)

            raw.pick("eeg")

            # 2) epoch whole trials (macro_epochs)
            macro_epochs = create_macro_epochs(raw, self.config)
            print("events shape:", macro_epochs.events.shape)
            print("first 10 events:\n", macro_epochs.events[:10])
            logger.info(f"Subject {subj} ▶ macro labels: "
                        f"{np.unique(macro_epochs.events[:, -1])}")

            # 3) **split here** on whole trials
            train_macro, test_macro = train_test_event_split(
                macro_epochs,
                train_frac=train_frac,
                random_seed=random_seed
            )

            # 4) now crop each side into fixed‐length windows
            train_epochs = crop_subepochs(
                train_macro,
                self.config,
                self.preproc_config.epoching.kwargs.crop_window_length,
                self.preproc_config.epoching.kwargs.crop_step_size
            )
            test_epochs = crop_subepochs(
                test_macro,
                self.config,
                self.preproc_config.epoching.kwargs.crop_window_length,
                self.preproc_config.epoching.kwargs.crop_step_size
            )
            logger.info(f"Subject {subj} ▶ train sub‐labels: "
                        f"{np.unique(train_epochs.events[:, -1])}")
            logger.info(f"Subject {subj} ▶ test  sub‐labels: "
                        f"{np.unique(test_epochs.events[:, -1])}")

            # 5) standardize
            standardized_train, esm_params = exponential_moving_standardization(
                train_epochs,
                smoothing_factor=self.preproc_config
                                   .exponential_moving_standardization.kwargs
                                   .smoothing_factor,
                eps=self.preproc_config
                        .exponential_moving_standardization.kwargs.eps,
                return_params=True
            )
            standardized_test = exponential_moving_standardization(
                test_epochs,
                smoothing_factor=self.preproc_config
                                   .exponential_moving_standardization.kwargs
                                   .smoothing_factor,
                eps=self.preproc_config
                        .exponential_moving_standardization.kwargs.eps,
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