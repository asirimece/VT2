#!/usr/bin/env python
"""
trainer.py

Defines a Trainer class that loads configuration and preprocessed data,
trains Deep4Net for both single (subject-by-subject) and pooled across subjects,
aggregates trial-level predictions, wraps the results in a BaseWrapper object,
and saves these results to files.
Usage:
    python trainer.py
"""

import os
import pickle
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from omegaconf import OmegaConf
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.dataset.dataset import EEGDataset
from lib.model.deep4net import Deep4NetModel
from lib.base.evaluate import Evaluator
from lib.logging import logger

logger = logger.get()


class BaseWrapper:
    """
    Container for wrapping training results.
    """
    def __init__(self, results_by_experiment):
        """
        results_by_experiment: dict with keys for each experiment type ("single" and "pooled")
         mapping to their respective results.
        """
        self.results_by_experiment = results_by_experiment

    def get_experiment_results(self, key):
        return self.results_by_experiment.get(key)


class BaselineTrainer:
    def __init__(self, base_config_path="config/experiment/base.yaml",
                       model_config_path="config/model/deep4net.yaml"):
        # Load configuration files using OmegaConf.
        self.base_config = OmegaConf.load(base_config_path)
        self.model_config = OmegaConf.load(model_config_path)
        print("[DEBUG] Loaded base configuration:")
        print(OmegaConf.to_yaml(self.base_config))
        print("[DEBUG] Loaded model configuration:")
        print(OmegaConf.to_yaml(self.model_config))
        
        # Read the experiment configuration.
        exp_config = self.base_config.experiment
        self.device = exp_config.device
        
        # Store both the single and pooled training configurations.
        self.single_train_config = exp_config.single
        self.pooled_train_config = exp_config.pooled

        # Read saving paths from the configuration.
        self.single_results_path = self.base_config.logging.single_results_path
        self.pooled_results_path = self.base_config.logging.pooled_results_path

        # Load preprocessed data as specified in the configuration.
        preprocessed_data = self.base_config.data.preprocessed_data
        with open(preprocessed_data, "rb") as f:
            self.preprocessed_data = pickle.load(f)
        print(f"[DEBUG] Loaded preprocessed data for {len(self.preprocessed_data)} subject(s).")
        print(f"[DEBUG] Subject keys: {list(self.preprocessed_data.keys())}")

    def save_epochs_preview_plot(self, epochs, subj_id, label="train", out_dir="plots"):
        """
        Saves a preview plot of epochs for visual inspection.
        """
        os.makedirs(out_dir, exist_ok=True)
        fig = epochs[:5].plot(
            n_epochs=5,
            n_channels=10,
            scalings=dict(eeg=20e-6),
            title=f"Subject {subj_id} ({label}) - Sample Epochs",
            block=False
        )
        fig.canvas.draw()
        out_name = os.path.join(out_dir, f"epoch_preview_subj_{subj_id}_{label}.png")
        fig.savefig(out_name, dpi=150)
        plt.close(fig)
        print(f"[DEBUG] Saved epochs preview to: {out_name}")

    def train_deep4net_model(self, X_train, y_train, trial_ids_train,
                              X_test, y_test, trial_ids_test,
                              model_config, train_config, device="cpu"):
        """
        Trains Deep4Net on sub-epochs and aggregates trial-level predictions.
        Returns the trained model and a dictionary with ground truth and predictions.
        """
        print("[DEBUG] In train_deep4net_model:")
        print("  - y_train distribution:", np.bincount(y_train.astype(int)))
        print("  - y_test distribution:", np.bincount(y_test.astype(int)))
        
        # Instantiate the model using the merged configuration.
        model_instance = Deep4NetModel(model_config)
        model = model_instance.get_model().to(device)
        print(f"[DEBUG]  - Built Deep4Net on device: {device}")
        print(f"[DEBUG]  - Model parameters: {model_config}")
        
        train_dataset = EEGDataset(X_train, y_train, trial_ids_train)
        test_dataset  = EEGDataset(X_test, y_test, trial_ids_test)
        train_loader  = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True)
        test_loader   = DataLoader(test_dataset, batch_size=train_config["batch_size"], shuffle=False)
        print("[DEBUG]  - Number of train sub-epochs:", len(train_dataset))
        print("[DEBUG]  - Number of test sub-epochs:", len(test_dataset))
        
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config["learning_rate"])
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(train_config["epochs"]):
            losses = []
            for batch_idx, (batch_X, batch_y, _) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                if (batch_idx + 1) % 10 == 0:
                    print(f"  - Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            logger.info(f"  - Epoch {epoch+1}/{train_config['epochs']} complete, Avg Loss = {np.mean(losses):.4f}")
        
        # Evaluation: aggregate sub-epoch predictions to trial-level.
        model.eval()
        all_logits = []
        all_trial_ids = []
        all_targets = []
        with torch.no_grad():
            for batch_X, batch_y, batch_tid in test_loader:
                batch_X = batch_X.to(device)
                out = model(batch_X).cpu().numpy()
                all_logits.append(out)
                all_trial_ids.append(batch_tid.numpy())
                all_targets.extend(batch_y.numpy())
        
        all_logits    = np.concatenate(all_logits, axis=0)
        all_trial_ids = np.concatenate(all_trial_ids, axis=0)
        all_targets   = np.array(all_targets)
        print(f"  - Aggregating predictions for {len(np.unique(all_trial_ids))} unique trials.")
        
        unique_trials = np.unique(all_trial_ids)
        trial_logits = []
        trial_labels = []
        for t in unique_trials:
            idx = np.where(all_trial_ids == t)[0]
            avg_logits = np.mean(all_logits[idx], axis=0)
            trial_logits.append(avg_logits)
            trial_labels.append(all_targets[idx[0]])
        trial_logits = np.array(trial_logits)
        trial_preds = trial_logits.argmax(axis=1)
        
        acc = accuracy_score(trial_labels, trial_preds)
        kappa = cohen_kappa_score(trial_labels, trial_preds)
        cm = confusion_matrix(trial_labels, trial_preds)
        logger.info(f"  - Trial-level Test Accuracy: {acc:.4f}, Kappa: {kappa:.4f}")
        logger.info("  - Confusion Matrix:\n%s", cm)
        
        trial_results = {
            "ground_truth": trial_labels,
            "predictions": trial_preds,
        }
        return model, trial_results

    def train_subject(self, subj, subject_data):
        """
        Trains Deep4Net on a single subject’s data (multiple runs) and returns the results.
        """
        print(f"\n=== Training Subject {subj} ===")
        train_ep = subject_data["0train"]
        print(f"[DEBUG] TRAIN_EP: {train_ep}")
        
        test_ep  = subject_data["1test"]
        print(f"[DEBUG] TEST_EP: {test_ep}")
        
        print("[DEBUG] train_ep.events[:10] =\n", train_ep.events[:10])
        print("[DEBUG] test_ep.events[:10]  =\n", test_ep.events[:10])
        self.save_epochs_preview_plot(train_ep, subj_id=subj, label="train")
        self.save_epochs_preview_plot(test_ep, subj_id=subj, label="test")
        
        X_train = train_ep.get_data()
        y_train = train_ep.events[:, -1]
        tid_tr  = train_ep.events[:, 1]
        
        X_test  = test_ep.get_data()
        y_test  = test_ep.events[:, -1]
        tid_te  = test_ep.events[:, 1]
        
        print("[DEBUG] X_train shape:", X_train.shape)
        print("[DEBUG] y_train[:10]:", y_train[:10])
        print("[DEBUG] tid_tr[:10]:", tid_tr[:10])
        print("[DEBUG] Unique labels in train_ep:", np.unique(y_train))
        print("[DEBUG] Unique trial IDs in train_ep:", np.unique(tid_tr))
        
        # Inline merge of model parameters for the single experiment.
        common_config = {key: self.model_config[key] for key in ["name", "in_chans", "n_classes", "n_times", "final_conv_length"] if key in self.model_config}
        exp_specific = self.model_config.get("single", {})  # Get experiment-specific parameters.
        merged_model_config = {**common_config, **exp_specific}
        print(f"[DEBUG] Model configuration for 'single': {merged_model_config}")
        
        run_results = []
        for run_i in range(self.single_train_config.n_runs):
            print(f"\n[DEBUG] [Run {run_i+1}/{self.single_train_config.n_runs}] for Subject {subj}")
            _, trial_results = self.train_deep4net_model(
                X_train, y_train, tid_tr,
                X_test, y_test, tid_te,
                merged_model_config, self.single_train_config, device=self.device
            )
            run_results.append(trial_results)
        return run_results[-1]

    def run(self):
        """
        Trains the model for both single-subject and pooled experiments.
        Returns a BaseWrapper object containing results for both experiments.
        """
        all_results = {}
        
        # --- Single-Subject Training ---
        print("[DEBUG] Running single-subject experiments...")
        single_results = {}
        for subj in sorted(self.preprocessed_data.keys()):
            subject_data = self.preprocessed_data[subj]
            trial_results = self.train_subject(subj, subject_data)
            single_results[subj] = trial_results
            print(f"[DEBUG] Subject {subj} training complete.")
        
        # Save single-subject results using the config-defined path.
        os.makedirs(os.path.dirname(self.single_results_path), exist_ok=True)
        with open(self.single_results_path, "wb") as f:
            pickle.dump(single_results, f)
        print(f"[DEBUG] Single-subject training results saved to {self.single_results_path}")
        all_results["single"] = single_results
        
        # --- Pooled Training ---
        print("[DEBUG] Running pooled experiment...")
        # Inline merge of model parameters for the pooled experiment.
        common_config = {key: self.model_config[key] for key in ["name", "in_chans", "n_classes", "n_times", "final_conv_length"] if key in self.model_config}
        exp_specific = self.model_config.get("pooled", {})
        merged_model_config = {**common_config, **exp_specific}
        print(f"[DEBUG] Model configuration for 'pooled': {merged_model_config}")
        
        # Merge data from all subjects.
        X_train_pool, y_train_pool, tid_train_pool = [], [], []
        X_test_pool, y_test_pool, tid_test_pool = [], [], []
        for subj_index, subj in enumerate(sorted(self.preprocessed_data.keys())):
            subject_data = self.preprocessed_data[subj]
            train_ep = subject_data["0train"]
            test_ep  = subject_data["1test"]
            offset = subj_index * 2000  # Create a unique offset for trial IDs.
            X_train_pool.append(train_ep.get_data())
            y_train_pool.append(train_ep.events[:, -1])
            tid_train_pool.append(train_ep.events[:, 1] + offset)
            X_test_pool.append(test_ep.get_data())
            y_test_pool.append(test_ep.events[:, -1])
            tid_test_pool.append(test_ep.events[:, 1] + offset)
        X_train_pool = np.concatenate(X_train_pool, axis=0)
        y_train_pool = np.concatenate(y_train_pool, axis=0)
        tid_train_pool = np.concatenate(tid_train_pool, axis=0)
        X_test_pool = np.concatenate(X_test_pool, axis=0)
        y_test_pool = np.concatenate(y_test_pool, axis=0)
        tid_test_pool = np.concatenate(tid_test_pool, axis=0)
        
        pool_run_results = []
        for run_i in range(self.pooled_train_config.n_runs):
            print(f"\n[DEBUG] [Run {run_i+1}/{self.pooled_train_config.n_runs}] for Pooled Training")
            _, trial_results = self.train_deep4net_model(
                X_train_pool, y_train_pool, tid_train_pool,
                X_test_pool, y_test_pool, tid_test_pool,
                merged_model_config, self.pooled_train_config, device=self.device
            )
            pool_run_results.append(trial_results)
        
        pooled_results = pool_run_results[-1]
        all_results["pooled"] = pooled_results
        
        # Save pooled results using the config-defined path.
        os.makedirs(os.path.dirname(self.pooled_results_path), exist_ok=True)
        with open(self.pooled_results_path, "wb") as f:
            pickle.dump(pooled_results, f)
        print(f"[DEBUG] Pooled training results saved to {self.pooled_results_path}")
        
        # Wrap overall results in a BaseWrapper.
        training_results = BaseWrapper(all_results)
        return training_results

if __name__ == "__main__":
    trainer = BaselineTrainer()
    trainer.run()
