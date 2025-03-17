#!/usr/bin/env python
"""
train.py

Baseline training script for Deep4Net using Braindecode.
It supports two modes:
  - "single": Train one model per subject (subject-by-subject training).
  - "pooled": Combine all subjects’ training data to train one model.
The training is repeated for n_runs to assess variability.
The preprocessed data is assumed to be a pickle file containing a dictionary:
    { subject: { "0train": epochs, "1test": epochs } }
where epochs are MNE Epochs objects.

Deep4Net is imported from lib/models/deep4net.py.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

# Import the Deep4Net builder
from lib.models.deep4net import build_deep4net_model

# For converting MNE Epochs into a Braindecode-compatible dataset
from braindecode.datasets import BaseConcatDataset
from braindecode.datasets import create_from_mne_epochs  # function to wrap epochs in a dataset

# For evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score

def load_preprocessed_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def create_dataset_from_epochs(epochs):
    """
    Convert an MNE Epochs object to a Braindecode dataset.
    """
    # Braindecode provides a utility to wrap MNE Epochs into a dataset.
    # This creates a BaseConcatDataset where each element is a trial.
    dataset = create_from_mne_epochs(epochs)
    return dataset

def get_dataloaders(preprocessed_data, mode, batch_size):
    """
    Depending on the mode, prepare training and testing DataLoaders.
    mode:
      - "single": returns a dict mapping subject -> (train_loader, test_loader)
      - "pooled": returns one pair (train_loader, test_loader) combining all subjects.
    """
    if mode == "single":
        loaders = {}
        for subj, sessions in preprocessed_data.items():
            if "0train" not in sessions or "1test" not in sessions:
                continue
            train_epochs = sessions["0train"]
            test_epochs = sessions["1test"]
            train_dataset = create_dataset_from_epochs(train_epochs)
            test_dataset = create_dataset_from_epochs(test_epochs)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            loaders[subj] = (train_loader, test_loader)
        return loaders
    elif mode == "pooled":
        all_train = []
        all_test = []
        for subj, sessions in preprocessed_data.items():
            if "0train" not in sessions or "1test" not in sessions:
                continue
            all_train.append(create_dataset_from_epochs(sessions["0train"]))
            all_test.append(create_dataset_from_epochs(sessions["1test"]))
        # Concatenate datasets using Braindecode's BaseConcatDataset
        train_dataset = BaseConcatDataset(all_train)
        test_dataset = BaseConcatDataset(all_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return (train_loader, test_loader)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

def train_model(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in train_loader:
            # Braindecode dataset returns a dict with keys 'X' and 'y'
            # X: shape (batch_size, channels, samples)
            # y: labels
            X = batch["X"].to(device).float()  # ensure float tensor
            y = batch["y"].to(device).long()   # ensure long tensor for classification
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch in test_loader:
            X = batch["X"].to(device).float()
            y = batch["y"].to(device).long()
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_true.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    acc = accuracy_score(all_true, all_preds)
    kappa = cohen_kappa_score(all_true, all_preds)
    conf_mat = confusion_matrix(all_true, all_preds)
    return acc, kappa, conf_mat

def run_training(cfg):
    # Load preprocessed data (MNE Epochs objects)
    preproc_file = cfg.data.preprocessed_data_file
    preprocessed_data = load_preprocessed_data(preproc_file)
    
    mode = cfg.training.mode
    batch_size = cfg.training.batch_size
    n_runs = cfg.training.n_runs
    epochs = cfg.training.epochs
    lr = cfg.training.learning_rate
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    
    # Prepare data loaders depending on mode
    if mode == "single":
        loaders = get_dataloaders(preprocessed_data, mode="single", batch_size=batch_size)
    elif mode == "pooled":
        loaders = get_dataloaders(preprocessed_data, mode="pooled", batch_size=batch_size)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    # Dictionary to store results for each run
    all_run_results = {}
    
    # For each run, reinitialize the model and train
    for run in range(n_runs):
        print(f"\n=== Run {run+1}/{n_runs} ===")
        run_results = {}
        if mode == "single":
            for subj, (train_loader, test_loader) in loaders.items():
                print(f"\nTraining for subject {subj}")
                # Load model config (for Deep4Net) from the model YAML
                model_cfg = OmegaConf.load("vt2/config/model/deep4net.yaml")
                model = build_deep4net_model(model_cfg).to(device)
                # Define loss and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                
                model = train_model(model, train_loader, criterion, optimizer, device, epochs)
                acc, kappa, conf_mat = evaluate_model(model, test_loader, device)
                run_results[subj] = {"accuracy": acc, "kappa": kappa, "confusion_matrix": conf_mat}
                print(f"Subject {subj}: Accuracy = {acc:.3f}, Kappa = {kappa:.3f}")
            all_run_results[f"run_{run+1}"] = run_results
        else:  # pooled mode
            train_loader, test_loader = loaders
            model_cfg = OmegaConf.load("vt2/config/model/deep4net.yaml")
            model = build_deep4net_model(model_cfg).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            model = train_model(model, train_loader, criterion, optimizer, device, epochs)
            acc, kappa, conf_mat = evaluate_model(model, test_loader, device)
            run_results = {"accuracy": acc, "kappa": kappa, "confusion_matrix": conf_mat}
            print(f"Pooled training: Accuracy = {acc:.3f}, Kappa = {kappa:.3f}")
            all_run_results[f"run_{run+1}"] = run_results

    return all_run_results

def main():
    # Load training configuration
    cfg = OmegaConf.load("vt2/config/training/base.yaml")
    results = run_training(cfg)
    
    # Save results
    results_save_path = cfg.logging.results_save_path
    os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
    with open(results_save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Training complete. Results saved to {results_save_path}")

if __name__ == "__main__":
    main()
