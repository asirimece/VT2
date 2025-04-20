#!/usr/bin/env python
import os
import argparse
import sys
import yaml
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.dataset.dataset import EEGMultiTaskDataset
from lib.pipeline.cluster.methods import evaluate_k_means  # optional
from lib.pipeline.cluster.cluster import SubjectClusterer
from lib.mtl.train import train_mtl_model
from lib.mtl.model import MultiTaskDeep4Net

def parse_args():
    p = argparse.ArgumentParser(description="Pretrain MTL backbone")
    p.add_argument("--config-path",    type=str, required=True,
                   help="Path to config/experiment/mtl.yaml")
    p.add_argument("--features-file",  type=str, required=True,
                   help="Path to features.pkl for clustering")
    p.add_argument("--out_dir",        type=str, required=True,
                   help="Where to save pretrained model weights")
    p.add_argument("--restrict_to_cluster", action="store_true",
                   help="If set, only train on subjects in `--cluster_id`")
    p.add_argument("--cluster_id", type=int, required="--restrict_to_cluster" in sys.argv,
               help="Cluster label (integer) to restrict to")
    return p.parse_args()

def main():
    args = parse_args()
    # --- load experiment config
    with open(args.config_path) as f:
        cfg = yaml.safe_load(f)["experiment"]
    preproc_file = cfg["preprocessed_file"]
    clustering_cfg = cfg["clustering"]
    mtl_cfg = cfg["mtl"]
    train_cfg = cfg["mtl"]["training"]

    # --- load preprocessed EEG data
    with open(preproc_file, "rb") as f:
        preproc = pickle.load(f)
    data_list, label_list, subj_ids = [], [], []
    for subj, sessions in preproc.items():
        for sess, ep in sessions.items():
            d = ep.get_data()
            lab = ep.events[:, -1]
            data_list.append(d)
            label_list.append(lab)
            subj_ids.extend([str(subj)] * d.shape[0])
    X = np.concatenate(data_list, axis=0)
    y = np.concatenate(label_list, axis=0)

    # --- clustering wrapper
    clusterer = SubjectClusterer(args.features_file, clustering_cfg)
    cluster_wrapper = clusterer.cluster_subjects(method=clustering_cfg.get("method","kmeans"))

    # --- optional: restrict to one cluster
    if args.restrict_to_cluster:
        if args.cluster_id is None:
            raise ValueError("--cluster_id required when --restrict_to_cluster is set")
        mask = [cluster_wrapper.get_cluster_for_subject(sid)==args.cluster_id
                for sid in subj_ids]
        X = X[mask]; y = y[mask]
        subj_ids = [sid for sid, m in zip(subj_ids, mask) if m]
        print(f"[DEBUG] Training only on cluster {args.cluster_id}: {len(y)} samples")

    # --- dataset & loader
    ds = EEGMultiTaskDataset(X, y, subj_ids, cluster_wrapper)
    loader = DataLoader(ds, batch_size=train_cfg["batch_size"], shuffle=True)

    # --- model
    n_chans = X.shape[1]
    n_times = X.shape[2]
    n_clusters = cluster_wrapper.get_num_clusters()
    model = MultiTaskDeep4Net(
        n_chans=n_chans,
        n_outputs=mtl_cfg["model"]["n_outputs"],
        n_clusters=n_clusters,
        backbone_kwargs={"n_times": mtl_cfg["backbone"]["n_times"], **mtl_cfg["backbone"]},
        head_kwargs=None
    )

    opt_cfg = train_cfg["optimizer"]
    lr = train_cfg["learning_rate"][0]
    weight_decay = opt_cfg.get("weight_decay", 0.0)
    optimizer = getattr(torch.optim, opt_cfg["name"])(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    # --- train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_mtl_model(
        model,
        loader,
        criterion,
        optimizer,
        device,
        epochs=train_cfg["epochs"]
    )

    # --- save
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "mtl_model_weights.pth")
    torch.save(model.state_dict(), out_path)
    print(f"[INFO] Saved pretrained MTL weights → {out_path}")

if __name__ == "__main__":
    main()
