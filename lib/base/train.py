#!/usr/bin/env python
"""
trainer.py

Defines a Trainer class that loads configuration and preprocessed data,
trains Deep4Net subject-by-subject (in "single" mode) or pooled across subjects (in "pooled" mode),
aggregates trial-level predictions, wraps the results in a BaseWrapper object,
and saves these results to a file.
These results are then used by the evaluator.
Usage:
    python trainer.py
"""

import os
import pickle
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
    def __init__(self, results_by_subject):
        """
        results_by_subject: dict mapping subject id (or 'pooled') to a dict containing at least:
            - "ground_truth": array of true labels per trial,
            - "predictions": array of predicted labels per trial.
        """
        self.results_by_subject = results_by_subject

    def get_subject_results(self, subject):
        return self.results_by_subject.get(subject)


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
        self.mode = exp_config.mode.lower()  # "pooled" or "single"
        self.device = exp_config.device
        
        # Choose training configuration based on mode.
        if self.mode == "pooled":
            self.train_config = exp_config.pooled
        else:
            self.train_config = exp_config.single

        self.results_save_path = self.base_config.logging.results_save_path

        # Load preprocessed data file as specified in the configuration.
        preprocessed_data_file = self.base_config.data.preprocessed_data_file
        with open(preprocessed_data_file, "rb") as f:
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
        
        # Instantiate the model via the OO class.
        model_instance = Deep4NetModel(model_config)
        model = model_instance.get_model().to(device)
        print(f"[DEBUG]  - Built Deep4Net on device: {device}")
        
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
        
        from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
        acc = accuracy_score(trial_labels, trial_preds)
        kappa = cohen_kappa_score(trial_labels, trial_preds)
        cm = confusion_matrix(trial_labels, trial_preds)
        logger.info(f"  - Trial-level Test Accuracy: {acc:.4f}, Kappa: {kappa:.4f}")
        logger.info("  - Confusion Matrix:\n", cm)
        
        trial_results = {
            "ground_truth": trial_labels,
            "predictions": trial_preds,
            # "probabilities": probabilities,  # Add if computed.
        }
        return model, trial_results

    def train_subject(self, subj, subject_data):
        """
        Trains Deep4Net on a single subject’s data (multiple runs) and returns the results.
        """
        print(f"\n=== Training Subject {subj} ===")
        train_ep = subject_data["0train"]
        print(f"DEBUG--- TRAIN_EP: {train_ep}")
        
        test_ep  = subject_data["1test"]
        print(f"DEBUG--- TEST_EP: {test_ep}")
        
        print("[DEBUG] train_ep.events[:10] =\n", train_ep.events[:10])
        print("[DEBUG] test_ep.events[:10]  =\n", test_ep.events[:10])
        self.save_epochs_preview_plot(train_ep, subj_id=subj, label="train")
        self.save_epochs_preview_plot(test_ep, subj_id=subj, label="test")
        
        X_train = train_ep.get_data()
        y_train = train_ep.events[:, -1]   # labels from column 3
        tid_tr  = train_ep.events[:, 1]     # trial IDs from column 2
        
        X_test  = test_ep.get_data()
        y_test  = test_ep.events[:, -1]
        tid_te  = test_ep.events[:, 1]
        
        print("[DEBUG] X_train shape:", X_train.shape)
        print("[DEBUG] y_train[:10]:", y_train[:10])
        print("[DEBUG] tid_tr[:10]:", tid_tr[:10])
        print("[DEBUG] Unique labels in train_ep:", np.unique(y_train))
        print("[DEBUG] Unique trial IDs in train_ep:", np.unique(tid_tr))
        
        run_results = []
        for run_i in range(self.train_config.n_runs):
            print(f"\n[DEBUG] [Run {run_i+1}/{self.train_config.n_runs}] for Subject {subj}")
            _, trial_results = self.train_deep4net_model(
                X_train, y_train, tid_tr,
                X_test, y_test, tid_te,
                self.model_config, self.train_config, device=self.device
            )
            run_results.append(trial_results)
        # For simplicity, use the last run's results.
        return run_results[-1]

    def run(self):
        """
        Trains the model either in "single" (subject-level) mode or "pooled" mode.
        In "single" mode, each subject is trained separately.
        In "pooled" mode, data from all subjects are concatenated and a single model is trained.
        Returns a BaseWrapper object.
        """
        results_all_subjects = {}
        mode = self.mode
        print(f"[DEBUG] Training mode: {mode}")
        
        if mode == "pooled":
            # Pool training and test data across subjects.
            X_train_pool, y_train_pool, tid_train_pool = [], [], []
            X_test_pool, y_test_pool, tid_test_pool = [], [], []
            # Use enumeration to generate a unique offset for each subject.
            for subj_index, subj in enumerate(sorted(self.preprocessed_data.keys())):
                subject_data = self.preprocessed_data[subj]
                train_ep = subject_data["0train"]
                test_ep  = subject_data["1test"]
                # Add an offset to trial IDs to make them globally unique.
                offset = subj_index * 2000  # Assumes each subject has fewer than 2000 trials.
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
            for run_i in range(self.train_config.n_runs):
                print(f"\n[DEBUG] [Run {run_i+1}/{self.train_config.n_runs}] for Pooled Training")
                _, trial_results = self.train_deep4net_model(
                    X_train_pool, y_train_pool, tid_train_pool,
                    X_test_pool, y_test_pool, tid_test_pool,
                    self.model_config, self.train_config, device=self.device
                )
                pool_run_results.append(trial_results)
            # For simplicity, use the last run's pooled results.
            results_all_subjects["pooled"] = pool_run_results[-1]
        else:  # "single" mode
            for subj in sorted(self.preprocessed_data.keys()):
                subject_data = self.preprocessed_data[subj]
                trial_results = self.train_subject(subj, subject_data)
                results_all_subjects[subj] = trial_results
                print(f"[DEBUG] Subject {subj} training complete.")
        
        training_results = BaseWrapper(results_all_subjects)
        
        # Save the training results using the configured path.
        os.makedirs(os.path.dirname(self.results_save_path), exist_ok=True)
        with open(self.results_save_path, "wb") as f:
            pickle.dump(training_results, f)
        print(f"[DEBUG] Training results saved to {self.results_save_path}")
        
        return training_results
