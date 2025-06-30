# lib/utils/recorder_utils.py

import os
import joblib
import torch
import numpy as np
from typing import Dict, Any
from lib.tl.model import TLModel


def make_tl_init_args(config: Any, n_chans: int, window_samples: int) -> Dict[str, Any]:
    """
    Build the kwargs you need to re-instantiate your TLModel exactly as in TLTrainer.
    """
    return dict(
        n_chans               = n_chans,
        n_outputs             = config.model.n_outputs,
        n_clusters_pretrained = config.model.n_clusters_pretrained,
        window_samples        = window_samples,
        head_kwargs = {
            "hidden_dim": config.head_hidden_dim,
            "dropout":    config.head_dropout,
        },
    )


class TorchSklearnWrapper:
    """
    Thin sklearn‐style wrapper around a saved TLModel .pth.
    Exposes .predict(X: np.ndarray) and .predict_proba(X: np.ndarray).
    """
    def __init__(self,
                 model_cls,
                 init_args: Dict[str, Any],
                 state_dict_path: str,
                 device: str = "cpu"):
        self.device = torch.device(device)
        # instantiate & load
        self.model = model_cls(**init_args).to(self.device)
        state     = torch.load(state_dict_path, map_location=self.device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32, device=self.device)
            # dummy subject IDs (0 for pooled, or pass cluster_id if wrapping a cluster model)
            sub_ids = [0] * x.shape[0]
            logits  = self.model(x, sub_ids)
            return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32, device=self.device)
            sub_ids = [0] * x.shape[0]
            logits  = self.model(x, sub_ids)
            return torch.softmax(logits, dim=1).cpu().numpy()


def save_for_recorder(*,
                      model,                   # your trained TLModel instance
                      init_args: Dict[str,Any],# same kwargs you used to build it
                      pth_path: str,          
                      device: str = "cpu"):
    """
    1) torch.save(model.state_dict(), pth_path)
    2) joblib.dump({"model": wrapper, "metadata": {}}, pth_path.replace('.pth','.joblib'))
    """
    # 1) save the raw weights
    os.makedirs(os.path.dirname(pth_path), exist_ok=True)
    torch.save(model.state_dict(), pth_path)

    # 2) build & dump the sklearn‐wrapper
    wrapper = TorchSklearnWrapper(
        model_cls       = TLModel,
        init_args       = init_args,
        state_dict_path = pth_path,
        device          = device
    )
    jl_path = pth_path.replace(".pth", ".joblib")
    joblib.dump({"model": wrapper, "metadata": {}}, jl_path)
