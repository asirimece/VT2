import pickle
from lib.base.evaluate import BaselineEvaluator
from lib.logging import logger
from omegaconf import DictConfig, OmegaConf
from lib.pipeline.preprocess.preprocess import Preprocessor
from lib.dataset.utils import save_preprocessed_data
from lib.base.train import BaselineTrainer

logger = logger.get()


def run(config: DictConfig) -> None: 
   logger.info("==== Starting baseline model training ====")
   #preprocessor = Preprocessor(config)
   #preprocessed_data = preprocessor.run()
   
   #save_preprocessed_data(preprocessed_data, config.dataset.preprocessing.output_file)
   
   with open("./dump/preprocessed_data.pkl", "rb") as f:
          preprocessed_data = pickle.load(f)
          
   trainer = BaselineTrainer()  
   training_results = trainer.run()  

   evaluator = BaselineEvaluator(config.experiment.evaluators)
   evaluator.evaluate_all(training_results)
   