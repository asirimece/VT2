import pickle
from omegaconf import DictConfig, OmegaConf
from lib.mtl.evaluate import MTLEvaluator
from lib.mtl.train import MTLTrainer
from lib.tl.evaluate import TLEvaluator
from lib.tl.train import TLTrainer
from lib.pipeline.preprocess.preprocess import Preprocessor
from lib.dataset.utils import save_preprocessed_data
from lib.pipeline.features.deep import DeepFeatureExtractor
from lib.logging import logger

logger = logger.get()


def run(config: DictConfig) -> None:
     logger.info("==== Starting transfer learning pipeline ====")

     #preprocessor = Preprocessor(config)
     #preprocessed_data = preprocessor.run()
     #save_preprocessed_data(preprocessed_data, config.dataset.preprocessing.output_file)

     with open("./dump/preprocessed_data_custom.pkl", "rb") as f:
          preprocessed_data = pickle.load(f)


     DeepFeatureExtractor().extract_and_save(
          preprocessed_data,
          output_path="./dump/deep_features.pkl",
          subset='train'
     )
     
     #with open("./dump/deep_features.pkl", "rb") as f:
          #features = pickle.load(f)
          
     trainer = MTLTrainer(config.experiment, config.model)
     mtl_wrapper = trainer.run()

     evaluator = MTLEvaluator(mtl_wrapper, config)
     evaluator.evaluate()

     tl_wrapper = TLTrainer(config).run()

     evaluator = TLEvaluator(tl_wrapper, config)
     evaluator.evaluate()