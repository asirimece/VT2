"""
Pipeline steps:
1. Preprocessing:
   - Run the preprocessing pipeline (filtering, ICA, epoching, sliding-window cropping, etc.)
   - Save the resulting preprocessed data to disk (path defined in config.data.preprocessed_data_file)
2. Baseline Model Training:
   - Instantiate the Trainer (which internally handles both single and pooled modes via config.experiment.mode)
   - Train the Deep4Net model on the preprocessed data

This module provides a run(config: DictConfig) function to be invoked from your main entry point.
"""

from lib.logging import logger
from omegaconf import DictConfig, OmegaConf
from lib.pipeline.preprocess.preprocess import Preprocessor
from lib.dataset.utils import save_preprocessed_data
from lib.base.train import BaselineTrainer

logger = logger.get()


def run(config: DictConfig) -> None:
    print("[DEBUG] Running baseline experiment with the following configuration:")
    print(OmegaConf.to_yaml(config))
    
    logger.info("Starting baseline model training.")
    # STEP 1: Preprocessing
    preprocessor = Preprocessor(config)
    preprocessed_data = preprocessor.run()
    
    preprocessed_data_file = (
        config.dataset.preprocessing.output_file
        if "dataset" in config and "preprocessing" in config.dataset and "output_file" in config.dataset.preprocessing
        else "./outputs/preprocessed_data.pkl"
    )
    save_preprocessed_data(preprocessed_data, preprocessed_data_file)
    print(f"Preprocessed data saved to: {preprocessed_data_file}")
    
    # STEP 2: Baseline Model Training
    trainer = BaselineTrainer()  # Trainer internally loads its model and experiment configurations
    training_results = trainer.run()  # Handles both "single" and "pooled" modes based on config.experiment.mode
    print("Baseline model training complete.")
    
