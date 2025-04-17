from lib.base.evaluate import Evaluator
from lib.logging import logger
from omegaconf import DictConfig, OmegaConf
from lib.pipeline.preprocess.preprocess import Preprocessor
from lib.dataset.utils import save_preprocessed_data
from lib.base.train import BaselineTrainer

logger = logger.get()


def run(config: DictConfig) -> None: 
   logger.info("==== Starting baseline model training. ====")
   # STEP 1: Preprocessing
   preprocessor = Preprocessor(config)
   preprocessed_data = preprocessor.run()
   
   save_preprocessed_data(preprocessed_data, config.dataset.preprocessing.output_file)
   logger.info(f"Preprocessed data saved to: {config.dataset.preprocessing.output_file}")
   
   # STEP 2: Baseline Model Training
   trainer = BaselineTrainer()  # Trainer internally loads its model and experiment configurations
   training_results = trainer.run()  # Handles both "single" and "pooled" modes based on config.experiment.mode
   logger.info("Baseline model training complete.")

   # STEP 3: Evaluation
   # Retrieve the evaluators configuration.
   # Confirm that the evaluators block is in the config
   if "evaluators" in config:
      raw_eval_config = config.experiment.evaluators
      logger.debug("Raw evaluator config: %s", raw_eval_config)
   else:
      logger.error("No 'evaluators' key found in config; evaluation will use empty configuration.")
      raw_eval_config = {}

   # Use evaluator config as is if it’s already a plain dict; if not, convert it.
   if OmegaConf.is_config(raw_eval_config):
      evaluator_config = OmegaConf.to_container(raw_eval_config, resolve=True)
   else:
      evaluator_config = raw_eval_config
   evaluator = Evaluator(evaluator_config)
   eval_results = evaluator.evaluate_all(training_results.results_by_experiment)
   logger.debug("Evaluation results:")
   logger.debug(eval_results)
   print("[DEBUG] Evaluation results:")
   print(eval_results)
   
   