import pickle
from lib.logging import logger
from omegaconf import DictConfig, OmegaConf
from lib.pipeline.preprocess.preprocess import Preprocessor
from lib.dataset.utils import save_preprocessed_data
from lib.pipeline.features.extract import FeatureExtractor, save_features

logger = logger.get()


def run(config: DictConfig) -> None:
   logger.info("==== Starting transfer learning pipeline. ====")
   # STEP 1: Preprocessing
   #preprocessor = Preprocessor(config)
   #preprocessed_data = preprocessor.run()
   #save_preprocessed_data(preprocessed_data, config.dataset.preprocessing.output_file)
   #logger.info(f"Preprocessed data saved to: {config.dataset.preprocessing.output_file}")

   with open("./dump/preprocessed_data.pkl", "rb") as f:
        preprocessed_data = pickle.load(f)
        
   # STEP 2: Feature Extraction & Selection
   features = FeatureExtractor.run(config, preprocessed_data)
   save_features(features, config.transform.output_file)
   logger.info(f"Features saved to: {config.transform.output_file}")
