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
from lib.augment.augment import apply_raw_augmentations

logger = logger.get()


def run(config: DictConfig) -> None:
     logger.info("==== Starting transfer learning pipeline ====")

     #preprocessor = Preprocessor(config)
     #preprocessed_data = preprocessor.run()
     #save_preprocessed_data(preprocessed_data, config.dataset.preprocessing.output_file)

     with open("./dump/preprocessed_data_custom.pkl", "rb") as f:
          preprocessed_data = pickle.load(f)

     # 1) Load preprocessed epochs
     with open(config.experiment.preprocessed_file, "rb") as f:
          preprocessed_data = pickle.load(f)

     # 2) PHASE 1 AUGMENTATION (raw-signal)
     if config.experiment.transfer.phase1_aug:
          for subj, splits in preprocessed_data.items():
               # get NumPy array of shape (n_trials, n_ch, n_t)
               X = splits["train"].get_data()
               # apply noise/warp/shift
               X_aug = apply_raw_augmentations(X, config.augment.augmentations)
               # re-wrap as an EpochsArray or your expected format:
               splits["train"]._data = X_aug
               # leave splits["test"] untouched
          
     features = FeatureExtractor.run(config, preprocessed_data)
     save_features(features, config.transform.output_file)

     #with open("./dump/features.pkl", "rb") as f:
          #features = pickle.load(f)
     
     # 4) PHASE 1: train your MTL backbone
     trainer = MTLTrainer(config.experiment, config.model)
     mtl_wrapper = trainer.run()

     evaluator = MTLEvaluator(mtl_wrapper, config)
     evaluator.evaluate()

     tl_wrapper = TLTrainer(config).run()

     evaluator = TLEvaluator(tl_wrapper, config)
     evaluator.evaluate()