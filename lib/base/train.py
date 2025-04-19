#!/usr/bin/env python
"""
trainer.py

Trains Deep4Net for both single‑subject and pooled baselines,
with multiple runs, seed control, and weight decay.
"""

import os
import pickle
import random
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
from lib.base.evaluate import BaselineEvaluator
from lib.logging import logger

logger = logger.get()


class BaseWrapper:
    """Container for wrapping training results."""
    def __init__(self, results_by_experiment):
        self.results_by_experiment = results_by_experiment

    def get_experiment_results(self, key):
        return self.results_by_experiment.get(key)


class BaselineTrainer:
    def __init__(self,
                 base_config_path="config/experiment/base.yaml",
                 model_config_path="config/model/deep4net.yaml"):
        # Load configs
        self.base_config  = OmegaConf.load(base_config_path)
        self.model_config = OmegaConf.load(model_config_path)

        print("[DEBUG] Loaded base configuration:")
        print(OmegaConf.to_yaml(self.base_config))
        print("[DEBUG] Loaded model configuration:")
        print(OmegaConf.to_yaml(self.model_config))

        exp_cfg = self.base_config.experiment
        self.device     = exp_cfg.device
        self.single_cfg = exp_cfg.single
        self.pooled_cfg = exp_cfg.pooled

        self.single_results_path = self.base_config.logging.single_results_path
        self.pooled_results_path = self.base_config.logging.pooled_results_path

        # Load preprocessed data
        with open(self.base_config.data.preprocessed_data, "rb") as f:
            self.preprocessed_data = pickle.load(f)
        print(f"[DEBUG] Loaded preprocessed data for {len(self.preprocessed_data)} subjects.")

    def train_deep4net_model(self,
                             X_train, y_train, train_ids,
                             X_test,  y_test,  test_ids,
                             model_cfg, train_cfg, device="cpu"):
        """
        Trains Deep4Net once and returns (model, trial_results).
        Applies weight_decay from train_cfg.weight_decay.
        """
        # Build model
        model_inst = Deep4NetModel(model_cfg)
        model      = model_inst.get_model().to(device)

        # DataLoaders
        train_ds = EEGDataset(X_train, y_train, train_ids)
        test_ds  = EEGDataset(X_test,  y_test,  test_ids)
        train_loader = DataLoader(train_ds,
                                  batch_size=train_cfg.batch_size,
                                  shuffle=True)
        test_loader  = DataLoader(test_ds,
                                  batch_size=train_cfg.batch_size,
                                  shuffle=False)

        # Optimizer + weight decay
        Optim     = getattr(torch.optim, train_cfg.optimizer)
        optimizer = Optim(model.parameters(),
                          lr=train_cfg.learning_rate,
                          weight_decay=train_cfg.weight_decay)

        criterion = nn.CrossEntropyLoss()

        # Training loop
        model.train()
        for epoch in range(train_cfg.epochs):
            losses, correct, total = [], 0, 0
            for Xb, yb, _ in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(Xb)
                loss   = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total   += Xb.size(0)
                losses.append(loss.item())

            print(f"[DEBUG] Epoch {epoch+1}/{train_cfg.epochs} "
                  f"Loss={np.mean(losses):.4f} Acc={correct/total:.4f} "
                  f"WD={train_cfg.weight_decay}")

        # Evaluation: aggregate to trial-level
        model.eval()
        all_logits, all_tids, all_y = [], [], []
        with torch.no_grad():
            for Xb, yb, tid in test_loader:
                Xb = Xb.to(device)
                out = model(Xb).cpu().numpy()
                all_logits.append(out)
                all_tids.append(tid.numpy())
                all_y.extend(yb.numpy())

        all_logits = np.concatenate(all_logits, axis=0)
        all_tids   = np.concatenate(all_tids,   axis=0)
        unique_t   = np.unique(all_tids)

        preds, labels = [], []
        for t in unique_t:
            idx = np.where(all_tids == t)[0]
            avg_logit = all_logits[idx].mean(axis=0)
            preds.append(int(avg_logit.argmax()))
            labels.append(int(all_y[idx[0]]))

        acc   = accuracy_score(labels, preds)
        kappa = cohen_kappa_score(labels, preds)
        cm    = confusion_matrix(labels, preds)
        logger.info(f"Trial‑level Test → Acc: {acc:.4f}, Kappa: {kappa:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")

        return model, {"ground_truth": labels, "predictions": preds}


    def train_subject(self, subj, subject_data):
        """
        Runs multiple runs for a single subject, reseeding each time.
        Returns a list of trial_results dicts.
        """
        tr = subject_data["0train"]
        te = subject_data["1test"]
        Xtr, ytr = tr.get_data(), tr.events[:, -1]
        Xte, yte = te.get_data(),  te.events[:, -1]
        tid_tr   = tr.events[:, 1]
        tid_te   = te.events[:, 1]

        # Merge model config for 'single'
        common = {k: self.model_config[k]
                  for k in ["name","in_chans","n_classes","n_times","final_conv_length"]
                  if k in self.model_config}
        merged_cfg = {**common, **self.model_config.get("single", {})}

        results_runs = []
        for run_i in range(self.single_cfg.n_runs):
            seed = self.single_cfg.seed_start + run_i
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            print(f"[DEBUG] Single run {run_i+1}/{self.single_cfg.n_runs} "
                  f"for subj {subj} (seed={seed})")

            _, trial_res = self.train_deep4net_model(
                Xtr, ytr, tid_tr,
                Xte, yte, tid_te,
                merged_cfg, self.single_cfg,
                device=self.device
            )
            results_runs.append(trial_res)

        return results_runs


    def train_pooled(self):
        """
        Runs multiple pooled runs, reseeding each time.
        Returns a list of trial_results dicts.
        """
        # Prepare pooled data once
        Xtr_list, ytr_list, tid_tr_list = [], [], []
        Xte_list, yte_list, tid_te_list = [], [], []

        for idx, (_, data) in enumerate(self.preprocessed_data.items()):
            tr = data["0train"]; te = data["1test"]
            offset = idx * 1_000_000
            Xtr_list.append(tr.get_data());     ytr_list.append(tr.events[:,-1])
            tid_tr_list.append(tr.events[:,1] + offset)
            Xte_list.append(te.get_data());     yte_list.append(te.events[:,-1])
            tid_te_list.append(te.events[:,1] + offset)

        Xtr   = np.concatenate(Xtr_list, axis=0)
        ytr   = np.concatenate(ytr_list, axis=0)
        tid_tr= np.concatenate(tid_tr_list, axis=0)
        Xte   = np.concatenate(Xte_list, axis=0)
        yte   = np.concatenate(yte_list, axis=0)
        tid_te= np.concatenate(tid_te_list, axis=0)

        # Merge model config for 'pooled'
        common = {k: self.model_config[k]
                  for k in ["name","in_chans","n_classes","n_times","final_conv_length"]
                  if k in self.model_config}
        merged_cfg = {**common, **self.model_config.get("pooled", {})}

        pooled_runs = []
        for run_i in range(self.pooled_cfg.n_runs):
            seed = self.pooled_cfg.seed_start + run_i
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            print(f"[DEBUG] Pooled run {run_i+1}/{self.pooled_cfg.n_runs} (seed={seed})")

            _, trial_res = self.train_deep4net_model(
                Xtr, ytr, tid_tr,
                Xte, yte, tid_te,
                merged_cfg, self.pooled_cfg,
                device=self.device
            )
            pooled_runs.append(trial_res)

        return pooled_runs


    def run(self):
        """
        Orchestrates single & pooled training and saves list-of-runs results.
        """
        # --- single-subject ---
        single_res = {}
        for subj, data in self.preprocessed_data.items():
            single_res[subj] = self.train_subject(subj, data)
        os.makedirs(os.path.dirname(self.single_results_path), exist_ok=True)
        with open(self.single_results_path, "wb") as f:
            pickle.dump(single_res, f)
        print(f"[INFO] Single-subject training results are saved.")

        # --- pooled ---
        pooled_res = self.train_pooled()
        os.makedirs(os.path.dirname(self.pooled_results_path), exist_ok=True)
        with open(self.pooled_results_path, "wb") as f:
            pickle.dump(pooled_res, f)
        print(f"[INFO] Pooled training results are saved.")

        return BaseWrapper({"single": single_res,
                            "pooled": pooled_res})

