
import os
import glob
import json
import argparse
import torch
import joblib
from sklearn.pipeline import Pipeline
from torch import nn
from lib.tl.model import TLModel

# --- 1) Wrapper class: makes TLModel behave like an sklearn classifier.
class Deep4NetTLWrapper:
    def __init__(self, model_path, model_ctor_kwargs, device="cpu"):
        self.model_path        = model_path
        self.model_ctor_kwargs = model_ctor_kwargs
        self.device            = torch.device(device)
        self._load_model()

    def _load_model(self):
        # instantiate your TLModel and load weights
        self.model = TLModel(**self.model_ctor_kwargs).to(self.device)

        # 1) Add a dummy head for subj_0 (so predict(...) will work)
        #    We use the same signature that your TLTrainer did:
        self.model.add_new_head(0)

        # 2) Load the pooled weights, but ignore any heads beyond the dummy one:
        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state, strict=False)

        self.model.eval()


    def fit(self, X, y=None):
        # no training here
        return self

    def predict(self, X):
        import numpy as np
        with torch.no_grad():
            tx = torch.tensor(X, dtype=torch.float32, device=self.device)
            batch = tx.shape[0]
            # single‐subject wrapper: dummy IDs
            sids = [0]*batch
            out = self.model(tx, sids).cpu().numpy()
            return out.argmax(axis=1)

    def predict_proba(self, X):
        import numpy as np
        with torch.no_grad():
            tx = torch.tensor(X, dtype=torch.float32, device=self.device)
            batch = tx.shape[0]
            sids = [0]*batch
            out = self.model(tx, sids).cpu().numpy()
            exp = np.exp(out - out.max(axis=1, keepdims=True))
            return exp / exp.sum(axis=1, keepdims=True)

