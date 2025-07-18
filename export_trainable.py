#!/usr/bin/env python3
# export_manual_pipelines.py

import joblib
import torch
import torch.nn as nn
from torch import Tensor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from braindecode.models import Deep4Net

# ─── 1) Copy in your Deep4NetTL estimator ───────────────────────────────────
class Deep4NetTL(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        pooled_model_path: str,
        select_cfg: dict,
        model_per_cluster: dict[int,str],
        n_steps: int = 5,
        lr: float   = 1e-3,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.pooled_model_path  = pooled_model_path
        self.select_cfg          = select_cfg
        self.model_per_cluster   = model_per_cluster
        self.n_steps             = n_steps
        self.lr                  = lr
        self.batch_size          = batch_size
        self.device              = device
        self._net                = None
        self._opt                = None

    def _init_model(self, X):
        device = torch.device(self.device)
        _, n_chans, n_times = X.shape
        # build backbone with 2 outputs
        self._net = Deep4Net(
            n_chans=n_chans, n_outputs=2, n_times=n_times,
            sfreq=250, final_conv_length="auto"
        ).to(device)
        # load pooled weights
        state = torch.load(self.pooled_model_path, map_location=device)
        self._net.load_state_dict(state, strict=False)

        # if clustering, pick and load per-cluster head
        if self.select_cfg.get("enabled", False):
            # simple clustering on flattened X
            Z = X.reshape(len(X), -1)
            grp = int(self._cluster_pipe.predict(Z)[0])
            head_state = torch.load(self.model_per_cluster[grp], map_location=device)
            self._net.load_state_dict(head_state, strict=False)

        self._opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)

    def fit(self, X, y):
        import numpy as np
        X_t = torch.tensor(X.astype(np.float32), device=self.device)
        y_t = torch.tensor(y.astype(int), device=self.device)

        if self._net is None:
            self._init_model(X)

        self._net.train()
        for _ in range(self.n_steps):
            idx = torch.randperm(len(X_t))[: self.batch_size]
            xb, yb = X_t[idx], y_t[idx]
            preds = self._net(xb)
            loss  = nn.functional.cross_entropy(preds, yb)
            self._opt.zero_grad()
            loss.backward()
            self._opt.step()
        return self

    def predict(self, X):
        import numpy as np
        X_t = torch.tensor(X.astype(np.float32), device=self.device)
        self._net.eval()
        with torch.no_grad():
            logits = self._net(X_t)
            return logits.argmax(dim=1).cpu().numpy()

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
