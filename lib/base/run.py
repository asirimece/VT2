#!/usr/bin/env python
"""
evaluation_runner.py

Loads training results and performs systematic evaluation using the Evaluator.
This script is specialized solely for evaluation.
"""

import pickle
from lib.logging import logger
from omegaconf import OmegaConf
from lib.base.evaluate import BaselineEvaluator
from lib.base.train import BaselineTrainer, BaseWrapper

def main():
    # Instantiate the Trainer and run training.
    trainer = BaselineTrainer()
    training_results = trainer.run()  # This returns a BaseWrapper object.
    
    # Load evaluation configuration.
    eval_config = OmegaConf.load("config/experiment/base.yaml")
    evaluator = Evaluator(eval_config.evaluators)
    
    #with open("trained_models/training_results.pkl", "rb") as f:
        #results = pickle.load(f)
    
    # Directly flow the training results to the evaluator.
    for subj, subj_results in training_results.results_by_subject.items():
        ground_truth = subj_results["ground_truth"]
        predictions = subj_results["predictions"]
        logger.info(f"[DEBUG] Evaluating Subject {subj}...")
        eval_metrics = evaluator.evaluate(ground_truth, predictions)
        logger.info(f"Evaluation metrics for Subject {subj}:")
        for metric, value in eval_metrics.items():
            logger.info(f"  {metric}: {value}")

if __name__ == "__main__":
    main()
