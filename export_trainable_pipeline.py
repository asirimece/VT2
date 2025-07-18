from lib.train.algorithms.deep4nettl import Deep4NetTL
#!/usr/bin/env python3
# export_manual_pipelines.py

import joblib
import torch
import torch.nn as nn
from torch import Tensor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from braindecode.models import Deep4Net


# ─── 2) Edit these paths to match your offline .pth files ────────────────────
POOLED   = "./dump/trained_models/tl/tl_pooled_model.pth"
CLUSTER  = "./dump/trained_models/mtl/cluster_model.pth"
HEADS    = {
    0: "./dump/trained_models/mtl/base_model_cluster0.pth",
    1: "./dump/trained_models/mtl/base_model_cluster1.pth",
    2: "./dump/trained_models/mtl/base_model_cluster2.pth",
    3: "./dump/trained_models/mtl/base_model_cluster3.pth"
}
N_STEPS  = 5
LR       = 1e-3
BATCH_SZ = 64
DEVICE   = "cpu"

# ─── 3) Build & dump the two pipelines ───────────────────────────────────────
def export_pipelines():
    # Baseline (no clustering)
    baseline = Pipeline([
        ("mtl", Deep4NetTL(
            pooled_model_path=POOLED,
            select_cfg={"enabled": False, "cluster_model": None},
            model_per_cluster=HEADS,
            n_steps=N_STEPS,
            lr=LR,
            batch_size=BATCH_SZ,
            device=DEVICE
        ))
    ])
    joblib.dump(baseline, "./data/models/trainable/pipeline_baseline.joblib")
    print("✅ Saved pipeline_baseline.joblib (no clustering)")

    # Clustered
    clustered = Pipeline([
        ("mtl", Deep4NetTL(
            pooled_model_path=POOLED,
            select_cfg={"enabled": True, "cluster_model": CLUSTER},
            model_per_cluster=HEADS,
            n_steps=N_STEPS,
            lr=LR,
            batch_size=BATCH_SZ,
            device=DEVICE
        ))
    ])
    joblib.dump(clustered, "./data/models/trainable/pipeline_clustered.joblib")
    print("✅ Saved pipeline_clustered.joblib (with clustering)")

if __name__ == "__main__":
    export_pipelines()
