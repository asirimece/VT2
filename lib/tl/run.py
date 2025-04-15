#!/usr/bin/env python
import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

# Set up reproducibility
seed = 42  # or any fixed integer value
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Optional: Define a worker init function for DataLoader if using multiple workers
def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
    # torch.manual_seed(seed + worker_id)  # not typically needed if using torch.cuda.manual_seed_all

from lib.tl.model import TLModel
from lib.tl.dataset import TLSubjectDataset
from lib.tl.trainer import TLTrainer
from lib.tl.evaluator import TLEvaluator

@hydra.main(config_path="../../config/experiment", config_name="tl.yaml", version_base=None)
def main(cfg: DictConfig):
    # Print device and output information
    device = cfg.device
    os.makedirs(cfg.out_dir, exist_ok=True)
    print(f"[DEBUG] Device: {device}")
    print(f"[DEBUG] Output directory: {cfg.out_dir}")

    # 1) Load new subject data from preprocessed_data file.
    with open(cfg.preprocessed_data, "rb") as f:
        preprocessed_data = pickle.load(f)
    print(f"[DEBUG] Preprocessed data loaded from {cfg.preprocessed_data}")

    if cfg.subject not in preprocessed_data:
        raise ValueError(f"Subject '{cfg.subject}' not found in {cfg.preprocessed_data}!")
    
    subject_data = preprocessed_data[cfg.subject]
    train_ep = subject_data["0train"]
    test_ep  = subject_data["1test"]

    X_train = train_ep.get_data()  # Expected shape: (n_trials, channels, times)
    y_train = train_ep.events[:, -1]
    X_test  = test_ep.get_data()
    y_test  = test_ep.events[:, -1]

    print(f"[DEBUG] X_train shape: {X_train.shape}")
    print(f"[DEBUG] y_train shape: {y_train.shape}")
    print(f"[DEBUG] X_test shape: {X_test.shape}")
    print(f"[DEBUG] y_test shape: {y_test.shape}")

    # Determine shape parameters from training data.
    n_chans = X_train.shape[1]
    window_samples = X_train.shape[2]
    print(f"[DEBUG] Number of channels: {n_chans}")
    print(f"[DEBUG] Window samples: {window_samples}")

    # 2) Build TLModel instance using hyperparameters from the config.
    tl_model = TLModel(
        n_chans=n_chans,
        n_outputs=cfg.model.n_outputs,
        n_clusters_pretrained=cfg.model.n_clusters_pretrained,
        window_samples=window_samples
    )
    print(f"[DEBUG] TLModel instance created.")

    state_dict = torch.load(cfg.pretrained_mtl_model, map_location=device)
    tl_model.load_state_dict(state_dict)
    print(f"[DEBUG] Pretrained model loaded from {cfg.pretrained_mtl_model}")

    # 4) Add a new head for the new subject using the provided feature dimension directly.
    # Instead of "TL_{cfg.subject}", use a numeric id.
    new_cluster_id = int(cfg.subject)
    feature_dim = 4  # Directly provided feature dimension (from previous backbone computation)
    print(f"[DEBUG] Adding new head with feature dimension: {feature_dim} for cluster id: {new_cluster_id}")
    tl_model.add_new_head(new_cluster_id, feature_dim=feature_dim)

    # 5) Prepare DataLoaders. Pass worker_init_fn if desired.
    train_dataset = TLSubjectDataset(X_train, y_train)
    test_dataset  = TLSubjectDataset(X_test,  y_test)
    train_loader  = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, worker_init_fn=worker_init_fn)
    test_loader   = DataLoader(test_dataset,  batch_size=cfg.batch_size, shuffle=False, worker_init_fn=worker_init_fn)
    print(f"[DEBUG] DataLoaders created.")

    # 6) Initialize and run the TL trainer.
    trainer = TLTrainer(
        model=tl_model,
        device=device,
        freeze_backbone=cfg.freeze_backbone,
        lr=cfg.lr,
        epochs=cfg.epochs
    )
    print(f"[DEBUG] Starting training...")
    trainer.train(train_loader, new_cluster_id)
    print(f"[DEBUG] Training completed.")

    tl_results = trainer.evaluate(test_loader, new_cluster_id)
    print(f"[DEBUG] Evaluation completed.")

    # 7) Save the TL results.
    out_results_path = os.path.join(cfg.out_dir, f"tl_{cfg.subject}_results.pkl")
    tl_results.save(out_results_path)
    print(f"[INFO] TL results saved to {out_results_path}")

    # 8) Evaluate and print metrics.
    evaluator = TLEvaluator()
    # Specify class names if you have them, e.g., class_names=['0', '1', '2', '3']
    metrics = evaluator.evaluate(tl_results, class_names=None, plot_confusion=True, save_plot_path=os.path.join(cfg.out_dir, "confusion_matrix.png"))
    print("[TL Evaluation] =>", metrics)

    # 9) Optionally save the final model.
    out_model_path = os.path.join(cfg.out_dir, f"tl_{cfg.subject}_model.pth")
    torch.save(tl_model.state_dict(), out_model_path)
    print(f"[INFO] TL model saved to {out_model_path}")

if __name__ == "__main__":
    main()
