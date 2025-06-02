import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from braindecode.models import Deep4Net
from omegaconf import OmegaConf

class DeepFeatureExtractor:
    def __init__(self, 
                 config_path="./config/dataset/customEEG.yaml",
                 model_path="./dump/trained_models/baseline/pooled_training.pth",
                 batch_size=64):
        """
        Loads config and model automatically. Do not pass params from pipeline.
        """
        self.config_path = config_path
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # ---- Load dataset config ----
        cfg = OmegaConf.load(self.config_path)
        self.n_chans = cfg.n_channels
        self.n_outputs = len([v for v in cfg.event_markers.values() if isinstance(v, int)])
        self.n_times = self._infer_n_times(cfg)

        # ---- Load Deep4Net model ----
        self.model = Deep4Net(
            n_chans=self.n_chans,
            n_outputs=self.n_outputs,
            n_times=self.n_times,
            final_conv_length='auto'
        )
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

    def _infer_n_times(self, cfg):
        # Try to infer n_times from crop_window_length and sfreq
        try:
            crop_window = float(cfg.preprocessing['epoching']['kwargs']['crop_window_length'])
            sfreq = float(cfg.sfreq)
            n_times = int(crop_window * sfreq)
        except Exception:
            n_times = 400  # fallback default, change as needed!
        return n_times

    def extract_features(self, X):
        self.model.eval()
        features = []
        ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(self.device)
                # Pass through all layers except the final_layer
                # (Braindecode Deep4Net: everything before final_layer)
                x = batch_X
                for name, module in self.model.named_children():
                    if name == 'final_layer':
                        break
                    x = module(x)
                feats = x
                feats = feats.view(feats.size(0), -1)  # flatten
                features.append(feats.cpu().numpy())
        features = np.concatenate(features, axis=0)
        return features

    def extract_all_subjects(self, preprocessed_data, subset='train'):
        all_features = {}
        for subj_id, splits in preprocessed_data.items():
            if subset not in splits:
                continue
            epochs = splits[subset]
            X = epochs.get_data()  # shape: (n_trials, n_channels, n_times)
            features = self.extract_features(X)
            all_features[subj_id] = features
        return all_features

    def extract_and_save(self, preprocessed_data, output_path, subset='train'):
        features = self.extract_all_subjects(preprocessed_data, subset=subset)
        with open(output_path, 'wb') as f:
            pickle.dump(features, f)

