# run_base.py

from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from src.features.feature_extractor import FeatureExtractionPipeline
from src.base.baseline_trainer import BaselineTrainer

def run(config):
    """
    High-level function that orchestrates the entire baseline training pipeline
    (for both single-subject and pooled modes). The steps are:
    
      1) Preprocessing
      2) Feature Extraction
      3) Baseline (Deep4Net) Training
    
    The specific mode (single vs. pooled) is handled internally by the BaselineTrainer
    based on cfg.experiment.mode.
    """

    # 1. Preprocessing
    preprocessing = PreprocessingPipeline()
    preprocessed_data = preprocessing.run(config)

    # 2. Feature Extraction
    feature_pipeline = FeatureExtractionPipeline()
    features = feature_pipeline.run(config, preprocessed_data)

    # 3. Baseline Training
    trainer = BaselineTrainer()
    trainer.run(config, features)
    
    print("Baseline training pipeline completed successfully.")
