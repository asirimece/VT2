import os
import pickle
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch import nn
from omegaconf import OmegaConf, DictConfig
from collections import defaultdict
from lib.dataset.dataset import TLSubjectDataset
from lib.tl.model import TLModel
from lib.tl.evaluate import TLEvaluator
from lib.utils.utils import _prefix_mtl_keys
from lib.logging import logger

logger = logger.get()

class TLWrapper:
    def __init__(self, ground_truth, predictions):
        self.ground_truth = ground_truth
        self.predictions = predictions

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


class TLTrainer:
    def __init__(self, config):
        # Allow both flat config or full Hydra config
        if isinstance(config, DictConfig) or isinstance(config, dict):
            if "experiment" in config and "transfer" in config.experiment:
                # expected case
                self.full_cfg = config
                self.config = config.experiment.transfer
                self.root_cfg = config.experiment
            elif "transfer" in config:
                # fallback: config is already the transfer section
                self.full_cfg = None
                self.config = config
                self.root_cfg = None
            else:
                raise ValueError("Missing `transfer` block in config")
        else:
            raise ValueError("Expected dict or DictConfig")

        assert self.config is not None, "Transfer config could not be extracted"
        self.device = torch.device(self.config.get("device", "cpu"))

        # Fallback for top-level paths
        if self.root_cfg:
            self.preprocessed_data_path = self.root_cfg.get("preprocessed_file")
        else:
            self.preprocessed_data_path = self.config.get("preprocessed_file")

        self._pretrained_weights = None
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

    def _set_seed(self, seed: int):
        import random
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run(self):
        with open(self.preprocessed_data_path, "rb") as f:
            preprocessed_data = pickle.load(f)

        subject_ids = sorted(preprocessed_data.keys())
        all_results = defaultdict(list)

        if not self.config.init_from_scratch:
            self._pretrained_weights = _prefix_mtl_keys(
                torch.load(self.config.pretrained_mtl_model, map_location=self.device)
            )

        for run_idx in range(self.config.n_runs):
            seed = self.config.seed_start + run_idx
            self._set_seed(seed)

            out_dir = os.path.join(self.config.tl_model_output, f"run_{run_idx}")
            os.makedirs(out_dir, exist_ok=True)

            for subject_id in subject_ids:
                weights_path = os.path.join(out_dir, f"tl_{subject_id}_model.pth")
                results_path = os.path.join(out_dir, f"tl_{subject_id}_results.pkl")

                X_train, y_train, X_test, y_test = self._load_subject_data(preprocessed_data, subject_id)
                self.model = self._build_model(X_train, subject_id)
                self.optimizer = self._build_optimizer()

                train_loader, test_loader = self._build_dataloaders(X_train, y_train, X_test, y_test)
                self._train(train_loader, subject_id)
                wrapper = self._evaluate(test_loader, subject_id)

                wrapper.save(results_path)
                torch.save(self.model.state_dict(), weights_path)
                all_results[subject_id].append(wrapper)

        return all_results

    def _load_subject_data(self, preprocessed_data, subject_id: int):
        subj_data = preprocessed_data[subject_id]
        train_ep = subj_data["0train"]
        test_ep = subj_data["1test"]
        X_train, y_train = train_ep.get_data(), train_ep.events[:, -1]
        X_test, y_test = test_ep.get_data(), test_ep.events[:, -1]
        return X_train, y_train, X_test, y_test

    def _build_model(self, X_train, subject_id: int):
        n_chans = X_train.shape[1]
        window_samples = X_train.shape[2]

        model = TLModel(
            n_chans=n_chans,
            n_outputs=self.config.model.n_outputs,
            n_clusters_pretrained=self.config.model.n_clusters_pretrained,
            window_samples=window_samples,
            freeze_layers=self.config.model.get("freeze_layers", None)
        )

        if self._pretrained_weights is not None:
            model.load_state_dict(self._pretrained_weights)
            logger.info("Loaded pretrained MTL weights into TL model.")
        else:
            logger.info("Training TL model from scratch.")

        if self.config.freeze_backbone:
            for param in model.shared_backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen.")

        model.add_new_head(subject_id, feature_dim=self.config.get("feature_dim", None))
        return model.to(self.device)

    def _build_optimizer(self):
        lr_backbone = self.config.get("lr_backbone", self.config.lr)
        lr_head = self.config.get("lr_head", self.config.lr)

        decay_params_backbone, no_decay_params_backbone = [], []
        decay_params_head, no_decay_params_head = [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "bn" in name.lower():
                (no_decay_params_backbone if "shared_backbone" in name else no_decay_params_head).append(param)
            else:
                (decay_params_backbone if "shared_backbone" in name else decay_params_head).append(param)

        return torch.optim.Adam([
            {"params": decay_params_backbone,    "lr": lr_backbone, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params_backbone, "lr": lr_backbone, "weight_decay": 0.0},
            {"params": decay_params_head,        "lr": lr_head,     "weight_decay": self.config.weight_decay},
            {"params": no_decay_params_head,     "lr": lr_head,     "weight_decay": 0.0},
        ])

    def _build_dataloaders(self, X_train, y_train, X_test, y_test):
        train_ds = TLSubjectDataset(X_train, y_train)
        test_ds = TLSubjectDataset(X_test, y_test)
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=self.config.batch_size, shuffle=False)
        return train_loader, test_loader

    def _train(self, train_loader, subject_id: int):
        self.model.train()
        for epoch in range(1, self.config.epochs + 1):
            total_loss, correct, count = 0.0, 0, 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.epochs}", leave=False)
            for X, y in pbar:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X, [subject_id] * X.size(0))
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                count += y.size(0)
                total_loss += loss.item() * y.size(0)
                pbar.set_postfix(loss=f"{total_loss/count:.4f}", acc=f"{correct/count:.4f}")
            print(f"[TLTrainer] Epoch {epoch}: Loss={total_loss/count:.4f}, Acc={correct/count:.4f}")

    def _evaluate(self, test_loader: torch.utils.data.DataLoader, subject_id: int):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(self.device)
                outputs = self.model(X, [subject_id] * X.size(0))
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
        return TLWrapper(
            ground_truth=np.array(all_labels, dtype=int),
            predictions=np.array(all_preds, dtype=int)
        )
