#!/usr/bin/env python
"""
evaluation_runner.py

Loads training results and performs systematic evaluation using the Evaluator.
This script is specialized solely for evaluation.
"""

import pickle
from omegaconf import OmegaConf
from lib.evaluate.evaluate import Evaluator
from lib.base.trainer import Trainer, TrainingResults

def main():
    # Instantiate the Trainer and run training.
    trainer = Trainer()
    training_results = trainer.run()  # This returns a TrainingResults object.
    
    # Load evaluation configuration.
    eval_cfg = OmegaConf.load("vt2/config/experiment/base.yaml")
    evaluator = Evaluator(eval_cfg.evaluators)
    
    #with open("trained_models/training_results.pkl", "rb") as f:
        #results = pickle.load(f)
    
    # Directly flow the training results to the evaluator.
    for subj, subj_results in training_results.results_by_subject.items():
        ground_truth = subj_results["ground_truth"]
        predictions = subj_results["predictions"]
        print(f"[DEBUG] Evaluating Subject {subj}...")
        eval_metrics = evaluator.evaluate(ground_truth, predictions)
        print(f"Evaluation metrics for Subject {subj}:")
        for metric, value in eval_metrics.items():
            print(f"  {metric}: {value}")

if __name__ == "__main__":
    main()
