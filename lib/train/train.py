#!/usr/bin/env python
"""
trainer.py

Defines a Trainer class that loads configuration and preprocessed data,
trains Deep4Net subject-by-subject (in "single" mode) or pooled across subjects (in "pooled" mode),
aggregates trial-level predictions, wraps the results in a TrainingResults object,
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

import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.dataset.dataset import EEGDataset
from lib.model.deep4net import Deep4NetModel
from lib.evaluate.evaluate import Evaluator
from omegaconf import OmegaConf


class TrainingResults:
    """
    A simple container for wrapping training results.
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


class Trainer:
    def __init__(self, base_cfg_path="vt2/config/experiment/base.yaml",
                       model_cfg_path="vt2/config/model/deep4net.yaml"):
        # Load configuration files using Hydra (OmegaConf)
        self.base_cfg = OmegaConf.load(base_cfg_path)
        self.model_cfg = OmegaConf.load(model_cfg_path)
        print("[DEBUG] Loaded base configuration:")
        print(OmegaConf.to_yaml(self.base_cfg))
        print("[DEBUG] Loaded model configuration:")
        print(OmegaConf.to_yaml(self.model_cfg))
        
        self.train_cfg = self.base_cfg.experiment
        self.mode = self.train_cfg.get("mode", "single").lower()  # "single" or "pooled"
        self.device = self.train_cfg.device
        self.results_save_path = self.base_cfg.logging.results_save_path

        # Load preprocessed data file as specified in the config.
        preprocessed_data_file = self.base_cfg.data.preprocessed_data_file
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
                              model_cfg, train_cfg, device="cpu"):
        """
        Trains Deep4Net on sub-epochs and aggregates trial-level predictions.
        Returns the trained model and a dictionary with ground truth and predictions.
        """
        print("[DEBUG] In train_deep4net_model:")
        print("  - y_train distribution:", np.bincount(y_train.astype(int)))
        print("  - y_test distribution:", np.bincount(y_test.astype(int)))
        
        # Instantiate the model via the OO class.
        model_instance = Deep4NetModel(model_cfg)
        model = model_instance.get_model().to(device)
        print(f"  - Built Deep4Net on device: {device}")
        
        train_dataset = EEGDataset(X_train, y_train, trial_ids_train)
        test_dataset  = EEGDataset(X_test, y_test, trial_ids_test)
        train_loader  = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True)
        test_loader   = DataLoader(test_dataset, batch_size=train_cfg["batch_size"], shuffle=False)
        print("  - Number of train sub-epochs:", len(train_dataset))
        print("  - Number of test sub-epochs:", len(test_dataset))
        
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(train_cfg["epochs"]):
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
            print(f"  - Epoch {epoch+1}/{train_cfg['epochs']} complete, Avg Loss = {np.mean(losses):.4f}")
        
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
        print(f"  - Trial-level Test Accuracy: {acc:.4f}, Kappa: {kappa:.4f}")
        print("  - Confusion Matrix:\n", cm)
        
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
        print(f"DEBUG--- TRAIN_EP:{train_ep}")  # or id(train_ep)
        
        test_ep  = subject_data["1test"]
        print(f"DEBUG--- TEST_EP:{test_ep}")
        
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
        for run_i in range(self.train_cfg.n_runs):
            print(f"\n[DEBUG] [Run {run_i+1}/{self.train_cfg.n_runs}] for Subject {subj}")
            _, trial_results = self.train_deep4net_model(
                X_train, y_train, tid_tr,
                X_test, y_test, tid_te,
                self.model_cfg, self.train_cfg, device=self.device
            )
            run_results.append(trial_results)
        # For simplicity, use the last run's results.
        return run_results[-1]

    def run(self):
        """
        Trains the model either in "single" (subject-level) mode or "pooled" mode.
        In "single" mode, each subject is trained separately.
        In "pooled" mode, data from all subjects are concatenated and a single model is trained.
        Returns a TrainingResults object.
        """
        results_all_subjects = {}
        mode = self.mode
        print(f"[DEBUG] Training mode: {mode}")
        
        if mode == "pooled":
            # Pool training and test data across subjects.
            X_train_pool, y_train_pool, tid_train_pool = [], [], []
            X_test_pool, y_test_pool, tid_test_pool = [], [], []
            for subj in sorted(self.preprocessed_data.keys()):
                subject_data = self.preprocessed_data[subj]
                train_ep = subject_data["0train"]
                test_ep  = subject_data["1test"]
                X_train_pool.append(train_ep.get_data())
                y_train_pool.append(train_ep.events[:, -1])
                tid_train_pool.append(train_ep.events[:, 1])
                X_test_pool.append(test_ep.get_data())
                y_test_pool.append(test_ep.events[:, -1])
                tid_test_pool.append(test_ep.events[:, 1])
            X_train_pool = np.concatenate(X_train_pool, axis=0)
            y_train_pool = np.concatenate(y_train_pool, axis=0)
            tid_train_pool = np.concatenate(tid_train_pool, axis=0)
            X_test_pool = np.concatenate(X_test_pool, axis=0)
            y_test_pool = np.concatenate(y_test_pool, axis=0)
            tid_test_pool = np.concatenate(tid_test_pool, axis=0)
            
            pool_run_results = []
            for run_i in range(self.train_cfg.n_runs):
                print(f"\n[DEBUG] [Run {run_i+1}/{self.train_cfg.n_runs}] for Pooled Training")
                _, trial_results = self.train_deep4net_model(
                    X_train_pool, y_train_pool, tid_train_pool,
                    X_test_pool, y_test_pool, tid_test_pool,
                    self.model_cfg, self.train_cfg, device=self.device
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
        
        training_results = TrainingResults(results_all_subjects)
        
        # Save the training results using the configured path.
        os.makedirs(os.path.dirname(self.results_save_path), exist_ok=True)
        with open(self.results_save_path, "wb") as f:
            pickle.dump(training_results, f)
        print(f"[DEBUG] Training results saved to {self.results_save_path}")
        
        """# --- Immediately evaluate the training results ---
        # Load evaluation configuration from the evaluation YAML.
        eval_cfg = OmegaConf.load("vt2/config/experiment/base.yaml")
        evaluator = Evaluator(eval_cfg.evaluators)
        print("[DEBUG] Running evaluation on training results...")
        for subj, subj_results in training_results.results_by_subject.items():
            ground_truth = subj_results["ground_truth"]
            predictions = subj_results["predictions"]
            print(f"[DEBUG] Evaluating Subject {subj}...")
            eval_metrics = evaluator.evaluate(ground_truth, predictions)
            print(f"Evaluation metrics for Subject {subj}:")
            for metric, value in eval_metrics.items():
                print(f"  {metric}: {value}")"""
                
        return training_results


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()

    
    
    
"""#!/usr/bin/env python

train.py

Loads raw data (or sub-epoched data) and:
1) Visually confirms event alignment by plotting raw + events and saving to file.
2) Checks event distribution / class labels.
3) Optionally sets tmin/tmax if you are re-epoching.
4) Plots a few epochs to confirm the data truly captures MI.
5) Trains Deep4Net on the final sub-epochs, with debug prints.

Usage:
    python train.py


import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

import mne
import matplotlib
matplotlib.use("Agg")  # so we can save figures without a GUI
import matplotlib.pyplot as plt

from lib.dataset.dataset import EEGDataset
from braindecode.models import Deep4Net
from lib.model.deep4net import build_deep4net_model

from omegaconf import OmegaConf

def save_raw_plot_with_events(raw, subj_id, session_label, event_id=None, out_dir="plots"):

    Plot the raw data with event lines overlaid, then save to disk.
    Helps confirm if the event lines align with actual MI or just beep onset.

    os.makedirs(out_dir, exist_ok=True)
    events, _ = mne.events_from_annotations(raw)
    if event_id is None:
        # If you have a known mapping, e.g. {'feet':1, 'left_hand':2, ...}, pass it in
        event_id = {}
    print(f"[DEBUG] Subject {subj_id} / {session_label} => raw plot with {len(events)} events.")

    fig = raw.plot(
        events=events,
        event_id=event_id,
        duration=10.0,
        scalings='auto',
        title=f"Subject {subj_id}, {session_label} - Raw + Events",
        block=False
    )
    # Force a draw before saving
    fig.canvas.draw()
    out_path = os.path.join(out_dir, f"raw_plot_subj{subj_id}_{session_label}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[DEBUG] Saved raw+events plot to: {out_path}")


def save_epochs_preview_plot(epochs, subj_id, label="train", out_dir="plots"):

    Saves a PNG image showing a few epochs (EEG signals in time).
    Helps confirm that tmin/tmax are correct (e.g. 2..6 s) and
    that the data is in the MI window rather than baseline.

    os.makedirs(out_dir, exist_ok=True)
    fig = epochs[:5].plot(
        n_epochs=5,
        n_channels=10,
        scalings=dict(eeg=20e-6),  # or adjust to suit your data amplitude
        title=f"Subject {subj_id} ({label}) - sample epochs",
        block=False
    )
    fig.canvas.draw()
    out_name = os.path.join(out_dir, f"epoch_preview_subj_{subj_id}_{label}.png")
    fig.savefig(out_name, dpi=150)
    plt.close(fig)
    print(f"[DEBUG] Saved epochs preview to {out_name}")


def train_deep4net_model(X_train, y_train, trial_ids_train,
                         X_test,  y_test,  trial_ids_test,
                         model_cfg, train_cfg, device="cpu"):

    Train a Deep4Net on sub-epochs and evaluate at trial-level by
    averaging the sub-epoch logits for each original trial.

    # Debug: label distribution
    print("[DEBUG] y_train distribution:", np.bincount(y_train.astype(int)))
    print("[DEBUG] y_test distribution:",  np.bincount(y_test.astype(int)))

    # Build model
    model = build_deep4net_model(model_cfg).to(device)
    print(f"[DEBUG] Built Deep4Net on device: {device}")

    # Datasets / DataLoaders
    train_dataset = EEGDataset(X_train, y_train, trial_ids_train)
    test_dataset  = EEGDataset(X_test,  y_test,  trial_ids_test)
    train_loader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=train_cfg["batch_size"], shuffle=False)

    print(f"[DEBUG] #Train sub-epochs: {len(train_dataset)}, #Test sub-epochs: {len(test_dataset)}")

    # Setup optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(train_cfg["epochs"]):
        losses = []
        for batch_idx, (batch_X, batch_y, _) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # Debug every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"[DEBUG] Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1}/{train_cfg['epochs']}, Average Loss = {np.mean(losses):.4f}")

    # Evaluate trial-level
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

    print(f"[DEBUG] Aggregating predictions for {len(np.unique(all_trial_ids))} unique trials.")
    
    # Aggregate sub-epoch predictions per trial
    unique_trials = np.unique(all_trial_ids)
    trial_logits  = []
    trial_labels  = []
    for t in unique_trials:
        idx = np.where(all_trial_ids == t)[0]
        avg_logits = np.mean(all_logits[idx], axis=0)
        trial_logits.append(avg_logits)
        trial_labels.append(all_targets[idx[0]])

    trial_logits = np.array(trial_logits)
    trial_preds = trial_logits.argmax(axis=1)

    acc   = accuracy_score(trial_labels, trial_preds)
    kappa = cohen_kappa_score(trial_labels, trial_preds)
    cm    = confusion_matrix(trial_labels, trial_preds)
    print(f"Trial-level Test Accuracy: {acc:.4f}, Kappa={kappa:.4f}")
    print("Confusion matrix:\n", cm)
    return model, acc


def main():
    # Load configurations from YAML files
    base_cfg = OmegaConf.load("vt2/config/experiment/base.yaml")
    model_cfg = OmegaConf.load("vt2/config/model/deep4net.yaml")
    
    print("[DEBUG] Loaded base configuration:")
    print(OmegaConf.to_yaml(base_cfg))
    print("[DEBUG] Loaded model configuration:")
    print(OmegaConf.to_yaml(model_cfg))
    
    train_cfg = base_cfg.experiment
    
    # ----------------------------------
    # 1) Load data
    # ----------------------------------
    # Example: either load a raw file or sub-epochs from your pipeline
    # For demonstration, let's assume you load a raw GDF for subject 1, session "train"
    # Or you can load preprocessed_data with sub-epochs. 
    # We'll show the raw approach for alignment check:
    #raw_fname = "./vt2/data/bci_iv2a/A01T.gdf"  # example
    #raw = mne.io.read_raw_gdf(raw_fname, preload=True)
    
    # 2) Save raw+events plot for alignment
    #events, event_id = mne.events_from_annotations(raw)
    # Suppose we know from docs that: {1:'feet',2:'left_hand',3:'right_hand',4:'tongue'}
    # We'll pass that to the raw plot function:
    #my_event_dict = {'feet':1, 'left_hand':2, 'right_hand':3, 'tongue':4}
    #save_raw_plot_with_events(raw, subj_id=1, session_label="train", event_id=my_event_dict)

    # 3) (If you confirm beep onset => tmin=2..6, do it here)
    # epochs = mne.Epochs(raw, events, event_id=my_event_dict, tmin=2.0, tmax=6.0, preload=True)
    # Then sub-epoch further if needed. 
    # Or if your pipeline is already done, just skip to load sub-epoched data:
    # with open("./outputs/preprocessed_data.pkl", "rb") as f:
    #     preprocessed_data = pickle.load(f)

    # For demonstration, let's say we skip the rest of pipeline and load sub-epoched data:
    data_file = "./outputs/preprocessed_data.pkl"
    with open(data_file, "rb") as f:
        preprocessed_data = pickle.load(f)
    
    print(f"[DEBUG] Loaded preprocessed data for {len(preprocessed_data)} subject(s).")
    
    # 2) (Optional) For debugging, print available subject keys
    print(f"[DEBUG] Subject keys in preprocessed data: {list(preprocessed_data.keys())}")


    device = train_cfg["device"]

    # ----------------------------------
    # 5) Train subject-by-subject
    # ----------------------------------
    results = {}
    for subj in sorted(preprocessed_data.keys()):
        print(f"\n=== Subject {subj} ===")
        subject_data = preprocessed_data[subj]
        
        # Assume subject_data has keys "0train" and "1test"
        train_ep = subject_data["0train"]
        test_ep  = subject_data["1test"]
        
        print("[DEBUG] train_ep.events[:10] =\n", train_ep.events[:10])
        print("[DEBUG] test_ep.events[:10]  =\n", test_ep.events[:10])

        # Debug: print shape and time range for macro epochs
        print(f"[DEBUG] Subject {subj} Train Epochs shape: {train_ep.get_data().shape}, tmin/tmax: {train_ep.tmin}, {train_ep.tmax}")
        print(f"[DEBUG] Subject {subj} Test Epochs shape: {test_ep.get_data().shape}, tmin/tmax: {test_ep.tmin}, {test_ep.tmax}")

        # Debug: shape, time range
        print("Train sub-epochs shape:", train_ep.get_data().shape, 
              "tmin/tmax:", (train_ep.tmin, train_ep.tmax))
        print("Test sub-epochs shape:", test_ep.get_data().shape, 
              "tmin/tmax:", (test_ep.tmin, test_ep.tmax))

        # Save a small preview of these sub-epochs
        save_epochs_preview_plot(train_ep, subj_id=subj, label="train")
        save_epochs_preview_plot(test_ep,  subj_id=subj, label="test")

        X_train = train_ep.get_data()
        y_train = train_ep.events[:, -1]
        tid_tr  = train_ep.events[:, 1]

        print("[DEBUG] X_train shape:", X_train.shape)
        print("[DEBUG] y_train[:10]:", y_train[:10])
        print("[DEBUG] tid_tr[:10]: ", tid_tr[:10])
        print("[DEBUG] Unique labels in train_ep:", np.unique(y_train))
        print("[DEBUG] Unique trial IDs in train_ep:", np.unique(tid_tr))

        X_test  = test_ep.get_data()
        y_test  = test_ep.events[:, -1]
        tid_te  = test_ep.events[:, 1]

        # Debug: Print unique trial IDs for train and test
        print(f"[DEBUG] Subject {subj} Unique Train Trial IDs: {np.unique(tid_tr)}")
        print(f"[DEBUG] Subject {subj} Unique Test Trial IDs: {np.unique(tid_te)}")
        
        # Possibly multiple runs
        run_accs = []
        for run_i in range(train_cfg["n_runs"]):
            print(f"\n[Run {run_i+1}/{train_cfg['n_runs']}] for Subject {subj}")
            model, acc = train_deep4net_model(
                X_train, y_train, tid_tr,
                X_test,  y_test,  tid_te,
                model_cfg, train_cfg, device=device
            )
            run_accs.append(acc)

        mean_acc = np.mean(run_accs)
        results[subj] = mean_acc
        print(f"Subject {subj} => mean acc over {train_cfg['n_runs']} runs: {mean_acc:.4f}")

    # 6) Print final results
    print("\nFinal results (Trial-level Accuracy per subject):")
    for subj, acc in results.items():
        print(f"  Subject {subj}: {acc:.4f}")


if __name__ == "__main__":
    main()
"""


"""#!/usr/bin/env python

train.py

Loads preprocessed sub-epoched data from ./outputs/preprocessed_data.pkl (or config path),
builds Deep4Net, trains on sub-epochs, and aggregates predictions
to get trial-level performance.

We've added debug statements to:
1) Show the distribution of labels in y_train / y_test.
2) Save an example epochs plot to confirm the time window is correct.
3) Print batch-level debug info during training.

Usage:
    python train.py


import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import mne
from lib.dataset.dataset import EEGDataset
from braindecode.models import Deep4Net
from lib.model.deep4net import build_deep4net_model

# -------------------------------------------------------------------
# Training function
# -------------------------------------------------------------------
def train_deep4net_model(X_train, y_train, trial_ids_train,
                         X_test,  y_test,  trial_ids_test,
                         model_cfg, train_cfg, device="cpu"):
    # 1) Debug: print distribution of labels in training/test
    #           so we can confirm that all 4 classes appear
    print("[DEBUG] y_train distribution:", np.bincount(y_train.astype(int)))
    print("[DEBUG] y_test distribution:",  np.bincount(y_test.astype(int)))

    # 2) Build model
    model = build_deep4net_model(model_cfg).to(device)
    print(f"[DEBUG] Built Deep4Net on device: {device}")

    # 3) Create DataLoaders
    train_dataset = EEGDataset(X_train, y_train, trial_ids_train)
    test_dataset  = EEGDataset(X_test,  y_test,  trial_ids_test)

    batch_size = train_cfg.get("batch_size", 64)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # 4) Debug: print total number of training/test sub-epochs
    print(f"[DEBUG] #Train sub-epochs: {len(train_dataset)}, #Test sub-epochs: {len(test_dataset)}")

    # 5) Setup optimizer and loss
    lr = train_cfg.get("learning_rate", 1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    n_epochs = train_cfg.get("epochs", 50)

    # 6) Training loop
    model.train()
    for epoch in range(n_epochs):
        losses = []
        for batch_idx, (batch_X, batch_y, _) in enumerate(train_loader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # Debug: show intermediate losses
            if (batch_idx + 1) % 10 == 0:
                print(f"[DEBUG] Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1}/{n_epochs}, Average Loss = {np.mean(losses):.4f}")

    # 7) Evaluation: aggregate predictions per original trial
    model.eval()
    all_logits = []
    all_trial_ids = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y, batch_tid in test_loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X).cpu().numpy()
            all_logits.append(logits)
            all_trial_ids.append(batch_tid.numpy())
            all_targets.extend(batch_y.numpy())

    all_logits    = np.concatenate(all_logits, axis=0)
    all_trial_ids = np.concatenate(all_trial_ids, axis=0)
    all_targets   = np.array(all_targets)

    unique_trials = np.unique(all_trial_ids)
    trial_logits  = []
    trial_labels  = []
    for t in unique_trials:
        idx = np.where(all_trial_ids == t)[0]
        avg_logits = np.mean(all_logits[idx], axis=0)  # average sub-epoch logits
        trial_logits.append(avg_logits)
        # all sub-epochs for t share the same label => just take the first sub-epoch's label
        trial_labels.append(all_targets[idx[0]])

    trial_logits = np.array(trial_logits)
    trial_preds = trial_logits.argmax(axis=1)

    acc   = accuracy_score(trial_labels, trial_preds)
    kappa = cohen_kappa_score(trial_labels, trial_preds)
    cm    = confusion_matrix(trial_labels, trial_preds)
    print(f"Trial-level Test Accuracy: {acc:.4f}, Kappa: {kappa:.4f}")
    print("Confusion matrix:\n", cm)
    return model, acc


# -------------------------------------------------------------------
# Main training loop
# -------------------------------------------------------------------
def main():
    # 1) Load sub-epoch preprocessed data
    data_file = "./outputs/preprocessed_data.pkl"
    with open(data_file, "rb") as f:
        preprocessed_data = pickle.load(f)

    # 2) Example config
    model_cfg = {
        "in_chans": 22,
        "n_classes": 4,
        "input_window_samples": 500,  # 2 s at 250 Hz
        "final_conv_length": "auto"
    }
    train_cfg = {
        "batch_size": 64,
        "learning_rate": 0.005,
        "epochs": 100,
        "n_runs": 5,  # repeat training multiple times
        "device": "cpu"
    }
    device = train_cfg["device"]

    overall_results = {}
    # 3) Loop over subjects
    for subj in sorted(preprocessed_data.keys()):
        print(f"\n=== Subject {subj} ===")
        train_ep = preprocessed_data[subj]["0train"]
        test_ep  = preprocessed_data[subj]["1test"]

        # Debug: Check shape, time range
        print("Train sub-epochs shape:", train_ep.get_data().shape, 
              "tmin/tmax:", (train_ep.tmin, train_ep.tmax))
        print("Test sub-epochs shape:", test_ep.get_data().shape, 
              "tmin/tmax:", (test_ep.tmin, test_ep.tmax))

        X_train = train_ep.get_data()
        y_train = train_ep.events[:, -1]
        tid_tr  = train_ep.events[:, 1]

        X_test = test_ep.get_data()
        y_test = test_ep.events[:, -1]
        tid_te = test_ep.events[:, 1]

        run_accuracies = []
        # 4) Perform multiple training runs
        for run_i in range(train_cfg["n_runs"]):
            print(f"\n[Run {run_i+1}/{train_cfg['n_runs']}] for Subject {subj}")
            model, acc = train_deep4net_model(
                X_train, y_train, tid_tr,
                X_test, y_test, tid_te,
                model_cfg, train_cfg, device=device
            )
            run_accuracies.append(acc)

        mean_acc = np.mean(run_accuracies)
        overall_results[subj] = mean_acc
        print(f"Subject {subj} => Mean accuracy over {train_cfg['n_runs']} runs: {mean_acc:.4f}")

    # 5) Print final results
    print("\nFinal results (average accuracy per subject):")
    for subj, acc_val in overall_results.items():
        print(f"  Subject {subj} => {acc_val:.4f}")


if __name__ == "__main__":
    main()
"""