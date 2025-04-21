import os
import pickle
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch import nn
from omegaconf import DictConfig
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
    def __init__(self, config: DictConfig):
        self.config = config.experiment.experiment.transfer
        self.device = torch.device(self.config.device)
        self.preprocessed_data_path = config.experiment.experiment.preprocessed_file
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
        logger.info("==== Starting transfer learning for all subjects ====")

        with open(self.preprocessed_data_path, "rb") as f:
            preprocessed_data = pickle.load(f)

        subject_ids = sorted(preprocessed_data.keys())
        all_results = defaultdict(list)

        # Load pretrained MTL weights once
        if not self.config.init_from_scratch:
            logger.info(f"Loading pretrained MTL weights from: {self.config.pretrained_mtl_model}")
            self._pretrained_weights = _prefix_mtl_keys(
                torch.load(self.config.pretrained_mtl_model, map_location=self.device)
            )

        for run_idx in range(self.config.n_runs):
            seed = self.config.seed_start + run_idx
            self._set_seed(seed)
            logger.info(f"\n=== TL Run {run_idx+1}/{self.config.n_runs} | seed={seed} ===")

            out_dir = os.path.join(self.config.model_output_dir, f"run_{run_idx}")
            os.makedirs(out_dir, exist_ok=True)

            for i, subject_id in enumerate(subject_ids):
                logger.info(f"→ [{i+1}/{len(subject_ids)}] Subject {subject_id}")

                weights_path = os.path.join(out_dir, f"tl_{subject_id}_model.pth")
                results_path = os.path.join(out_dir, f"tl_{subject_id}_results.pkl")

                X_train, y_train, X_test, y_test = self._load_subject_data(preprocessed_data, subject_id)
                self.model = self._build_model(X_train, subject_id)
                self.optimizer = self._build_optimizer()

                train_loader, test_loader = self._build_dataloaders(X_train, y_train, X_test, y_test)
                self._train(train_loader, subject_id)
                wrapper = self._evaluate(test_loader, subject_id)

                wrapper.save(results_path)
                logger.info(f"Saved TL results to: {results_path}")

                torch.save(self.model.state_dict(), weights_path)
                logger.info(f"Saved TL model to: {weights_path}")

                all_results[subject_id].append(wrapper)

        return all_results

    def _load_subject_data(self, preprocessed_data, subject_id: int):
        if subject_id not in preprocessed_data:
            raise ValueError(f"Subject '{subject_id}' not found in preprocessed data.")

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
            window_samples=window_samples
        )

        if self._pretrained_weights is not None:
            model.load_state_dict(self._pretrained_weights)
            logger.info("→ Loaded pretrained MTL weights into TL model.")
        else:
            logger.info("→ Training TL model from scratch.")

        # Optionally freeze backbone
        if self.config.freeze_backbone:
            for param in model.shared_backbone.parameters():
                param.requires_grad = False
            logger.info("→ Backbone frozen.")

        feature_dim = self.config.get("feature_dim", None)
        model.add_new_head(new_cluster_id=int(subject_id), feature_dim=feature_dim)
        return model.to(self.device)

    def _build_optimizer(self):
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "bn" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return torch.optim.Adam(
            [
                {"params": decay_params, "weight_decay": self.config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.config.lr
        )

    def _build_dataloaders(self, X_train, y_train, X_test, y_test):
        train_ds = TLSubjectDataset(X_train, y_train)
        test_ds = TLSubjectDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=self.config.batch_size, shuffle=False)
        return train_loader, test_loader

    def _train(self, train_loader: DataLoader, subject_id: int):
        self.model.train()

        for epoch in range(1, self.config.epochs + 1):
            total_loss = 0.0
            correct = 0
            count = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.epochs}", unit="batch", leave=False)
            for X, y in pbar:
                X = X.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X, [subject_id] * X.size(0))
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                count += y.size(0)
                total_loss += loss.item() * y.size(0)

                avg_loss = total_loss / count if count else 0.0
                acc = correct / count if count else 0.0
                pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}")

            print(f"[TLTrainer] Epoch {epoch}: Loss={avg_loss:.4f}, Acc={acc:.4f}")

    def _evaluate(self, test_loader: DataLoader, subject_id: int) -> TLWrapper:
        self.model.eval()
        all_preds = []
        all_labels = []

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






"""# tl_trainer.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import pickle

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
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        freeze_backbone: bool,
        lr: float,
        epochs: int,
        weight_decay: float
    ):
        # Move model to device
        self.device = device
        self.model = model.to(device)

        # Optionally freeze shared backbone parameters
        if freeze_backbone:
            for param in self.model.shared_backbone.parameters():
                param.requires_grad = False

        # Build optimizer on all parameters that require gradients
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay
        )

        # Standard cross-entropy loss
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = epochs

    def train(self, train_loader: torch.utils.data.DataLoader, new_cluster_id: int):
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward + backward + step
                self.optimizer.zero_grad()
                outputs = self.model(X_batch, [new_cluster_id] * X_batch.size(0))
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                # Track metrics
                preds = outputs.argmax(dim=1)
                total_correct += (preds == y_batch).sum().item()
                total_samples += y_batch.size(0)
                total_loss += loss.item() * y_batch.size(0)

            avg_loss = total_loss / total_samples if total_samples else 0.0
            avg_acc = total_correct / total_samples if total_samples else 0.0
            print(f"[TLTrainer] Epoch {epoch}/{self.epochs}, Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")

    def evaluate(self, test_loader: torch.utils.data.DataLoader, new_cluster_id: int) -> TLWrapper:
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch, [new_cluster_id] * X_batch.size(0))
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())

        return TLWrapper(
            ground_truth=np.array(all_labels, dtype=int),
            predictions=np.array(all_preds, dtype=int)
        )"""