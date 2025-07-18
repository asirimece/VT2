# torch_wrapper.py
import torch

class TorchScriptWrapper:
    """
    sklearn‐style estimator that lazily loads a TorchScript model from disk.
    The only attributes are ts_path and device, so pickling is trivial.
    """
    def __init__(self, ts_path: str, device: str = "cpu"):
        self.ts_path = ts_path
        self.device = device
        # do NOT load here!

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        # lazy‐load on first call
        if not hasattr(self, "_ts"):
            self._ts = torch.jit.load(self.ts_path, map_location=self.device)
            self._ts.eval()
        # assume X is a numpy array of shape (n_trials, n_chans, n_times)
        import numpy as np
        import torch as th
        tensor = th.tensor(X, dtype=th.float32, device=self.device)
        with th.no_grad():
            out = self._ts(tensor)
        # output shape (n_trials, n_outputs)
        preds = out.argmax(dim=1).cpu().numpy()
        return preds
