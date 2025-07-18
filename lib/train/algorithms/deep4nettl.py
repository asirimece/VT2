# offline-repo/lib/train/algorithms/dee4nettl.py

import joblib
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from braindecode.models import Deep4Net

class Deep4NetTL(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        *,
        pooled_model_path: str,
        select_cfg: dict,
        model_per_cluster: dict[int,str],
        n_steps: int = 5,
        lr: float   = 1e-3,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.pooled_model_path  = pooled_model_path
        self.select_cfg         = select_cfg
        self.model_per_cluster  = model_per_cluster
        self.n_steps            = n_steps
        self.lr                 = lr
        self.batch_size         = batch_size
        self.device             = device

        # Always load the cluster selector via joblib
        # (we’ll only use it if select_cfg["enabled"] is True)
        if select_cfg.get("cluster_model"):
            self._cluster_pipe = joblib.load(select_cfg["cluster_model"])
        else:
            self._cluster_pipe = None

        self._net = None
        self._opt = None

    def _init_model(self, X):
        device = torch.device(self.device)
        _, n_chans, n_times = X.shape

        # 1) Build backbone
        self._net = Deep4Net(
            n_chans=n_chans,
            n_outputs=2,
            n_times=n_times,
            sfreq=250,
            final_conv_length="auto"
        ).to(device)

        # 2) Load pooled weights
        state = torch.load(self.pooled_model_path, map_location=device, weights_only=False)
        self._net.load_state_dict(state, strict=False)

        # 3) If clustering enabled, pick & load per-cluster head
        if self.select_cfg.get("enabled", False):
            Z = X.reshape(len(X), -1)
            grp = int(self._cluster_pipe.predict(Z)[0])
            head_state = torch.load(
                self.model_per_cluster[grp],
                map_location=device,
                weights_only=False
            )
            self._net.load_state_dict(head_state, strict=False)

        # 4) Prepare optimizer
        self._opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)

    def fit(self, X, y):
        import numpy as _np
        X_t = torch.tensor(X.astype(_np.float32), device=self.device)
        y_t = torch.tensor(y.astype(int),    device=self.device)

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
        import numpy as _np
        X_t = torch.tensor(X.astype(_np.float32), device=self.device)
        self._net.eval()
        with torch.no_grad():
            logits = self._net(X_t)
            return logits.argmax(dim=1).cpu().numpy()
