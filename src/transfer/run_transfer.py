import pickle
from omegaconf import DictConfig, OmegaConf
from lib.mtl.evaluate import MTLEvaluator
from lib.mtl.train import MTLTrainer
from lib.tl.evaluate import TLEvaluator
from lib.tl.train import TLTrainer
from lib.pipeline.preprocess.preprocess import Preprocessor
from lib.dataset.utils import save_preprocessed_data
from lib.pipeline.features.extract import FeatureExtractor, save_features
from lib.logging import logger

logger = logger.get()


def run(config: DictConfig) -> None:
   logger.info("==== Starting transfer learning pipeline ====")

   #preprocessor = Preprocessor(config)
   #preprocessed_data = preprocessor.run()
   #save_preprocessed_data(preprocessed_data, config.dataset.preprocessing.output_file)
        
   #features = FeatureExtractor.run(config, preprocessed_data)
   #save_features(features, config.transform.output_file)
   
   with open("./dump/features.pkl", "rb") as f:
        features = pickle.load(f)
        
   trainer = MTLTrainer(config.experiment, config.model)
   mtl_wrapper = trainer.run()
   
   """if getattr(config.experiment, "prepare_recorder", False):
       # (optionally log a message)
       print("[INFO] prepare_recorder set → skipping evaluation step.")
       return mtl_wrapper
"""

   evaluator = MTLEvaluator(mtl_wrapper, config)
   evaluator.evaluate()
   
   tl_wrapper = TLTrainer(config).run()
   
   evaluator = TLEvaluator(tl_wrapper, config)
   evaluator.evaluate()
   