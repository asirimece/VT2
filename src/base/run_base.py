from lib.base.evaluate import BaselineEvaluator
from lib.logging import logger
from omegaconf import DictConfig, OmegaConf
from lib.pipeline.preprocess.preprocess import Preprocessor
from lib.dataset.utils import save_preprocessed_data
from lib.base.train import BaselineTrainer

logger = logger.get()


def run(config: DictConfig) -> None: 
   logger.info("==== Starting baseline model training. ====")
   # STEP 1: Preprocessing
   #preprocessor = Preprocessor(config)
   #preprocessed_data = preprocessor.run()
   
   #save_preprocessed_data(preprocessed_data, config.dataset.preprocessing.output_file)
   #logger.info(f"Preprocessed data saved to: {config.dataset.preprocessing.output_file}")
   
   # STEP 2: Baseline Model Training
   trainer = BaselineTrainer()  # Trainer internally loads its model and experiment configurations
   training_results = trainer.run()  # Handles both "single" and "pooled" modes based on config.experiment.mode
   logger.info("Baseline model training complete.")

   # STEP 3: Evaluation
   evaluator = BaselineEvaluator(config.experiment.evaluators)
   evaluator.evaluate_all(training_results)