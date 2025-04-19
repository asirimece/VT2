#!/usr/bin/env python
import os
import pickle
import torch
from torch.utils.data import DataLoader
import hydra
from lib.utils.utils import _prefix_mtl_keys
from omegaconf import DictConfig

from lib.tl.model import TLModel
from lib.tl.dataset import TLSubjectDataset
from lib.tl.trainer import TLTrainer
from lib.tl.evaluator import TLEvaluator

@hydra.main(config_path="../../config/experiment", config_name="tl.yaml", version_base=None)
def main(config: DictConfig):
    device = torch.device(config.device)
    os.makedirs(config.out_dir, exist_ok=True)

    print(f"[DEBUG] Device: {device}")
    print(f"[DEBUG] Output directory: {config.out_dir}")

    # 1) Load new subject data
    with open(config.preprocessed_data, "rb") as f:
        preprocessed_data = pickle.load(f)
    print(f"[DEBUG] Preprocessed data loaded from {config.preprocessed_data}")

    if int(config.subject) not in preprocessed_data:
        raise ValueError(f"Subject '{config.subject}' not found!")
    subj_dict = preprocessed_data[config.subject]
    train_ep = subj_dict["0train"]
    test_ep  = subj_dict["1test"]
    X_train, y_train = train_ep.get_data(), train_ep.events[:, -1]
    X_test,  y_test  = test_ep.get_data(),  test_ep.events[:, -1]

    print(f"[DEBUG] X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"[DEBUG] X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    # 2) Build model
    n_chans       = X_train.shape[1]
    window_samples= X_train.shape[2]
    tl_model = TLModel(
        n_chans=n_chans,
        n_outputs=config.model.n_outputs,
        n_clusters_pretrained=config.model.n_clusters_pretrained,
        window_samples=window_samples
    )
    print(f"[DEBUG] TLModel created.")

    # 3) Load pretrained weights (unless scratch)
    if not config.init_from_scratch:
        raw_state = torch.load(config.pretrained_mtl_model, map_location=device)
        # fix up key names so they match TLModel.mtl_net.*
        raw_sd = torch.load(config.pretrained_mtl_model, map_location=device)
        fixed_sd = _prefix_mtl_keys(raw_sd)
        tl_model.load_state_dict(fixed_sd)
        print(f"[DEBUG] Loaded and remapped pretrained backbone from {config.pretrained_mtl_model}")
    else:
        print(f"[DEBUG] Training from scratch (no pretrained weights)")

    # 4) Add new head for this subject
    new_cid = int(config.subject)
    feature_dim = config.get("feature_dim", None)
    tl_model.add_new_head(new_cluster_id=new_cid, feature_dim=feature_dim)

    # 5) DataLoaders
    train_ds = TLSubjectDataset(X_train, y_train)
    test_ds  = TLSubjectDataset(X_test,  y_test)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size, shuffle=False)
    print(f"[DEBUG] DataLoaders ready.")

    # 6) Train
    trainer = TLTrainer(
        model=tl_model,
        device=device,
        freeze_backbone=config.freeze_backbone,
        lr=config.lr,
        weight_decay=config.weight_decay,
        epochs=config.epochs
    )
    print(f"[DEBUG] Starting TL training...")
    trainer.train(train_loader, new_cid)
    print(f"[DEBUG] Training done.")

    # 7) Evaluate & save
    results = trainer.evaluate(test_loader, new_cid)
    out_res = os.path.join(config.out_dir, f"tl_{config.subject}_results.pkl")
    results.save(out_res)
    print(f"[INFO] Results → {out_res}")

    metrics = TLEvaluator().evaluate(results, plot_confusion=False)
    print("[TL Evaluation] =>", metrics)

    # 8) Save model weights
    out_model = os.path.join(config.out_dir, f"tl_{config.subject}_model.pth")
    torch.save(tl_model.state_dict(), out_model)
    print(f"[INFO] TL model → {out_model}")

if __name__ == "__main__":
    main()
