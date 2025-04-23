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

   preprocessor = Preprocessor(config)
   preprocessed_data = preprocessor.run()
   save_preprocessed_data(preprocessed_data, config.dataset.preprocessing.output_file)

   with open("./dump/preprocessed_data.pkl", "rb") as f:
        preprocessed_data = pickle.load(f)
        
   features = FeatureExtractor.run(config, preprocessed_data)
   save_features(features, config.transform.output_file)
   
   trainer = MTLTrainer(config.experiment, config.model)
   mtl_wrapper = trainer.run()
   
   evaluator = MTLEvaluator(mtl_wrapper, config)
   evaluator.evaluate()
   
   tl_wrapper = TLTrainer(config).run()
   
   evaluator = TLEvaluator(tl_wrapper, config)
   evaluator.evaluate()
