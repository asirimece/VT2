# lib/tl/train.py

import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm.auto import tqdm
from collections import defaultdict
from omegaconf import DictConfig

from lib.dataset.dataset import TLSubjectDataset
from lib.tl.model   import TLModel
from lib.tl.evaluate import TLEvaluator
from lib.utils.utils import _prefix_mtl_keys
from lib.logging import logger
from lib.augment.augment import mixup_batch

logger = logger.get()


def freeze_backbone_layers(backbone, freeze_until_layer=None):
    """
    Freezes backbone layers up to (and including) the named layer.
    """
    found = False
    for name, module in backbone.named_children():
        for p in module.parameters():
            p.requires_grad = False
        if freeze_until_layer and name == freeze_until_layer:
            found = True
            break
    if freeze_until_layer and not found:
        raise ValueError(f"freeze_until_layer '{freeze_until_layer}' not found")


class TLWrapper:
    def __init__(self, ground_truth, predictions):
        self.ground_truth = ground_truth
        self.predictions  = predictions

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


class TLTrainer:
    def __init__(self, config: DictConfig):
        # config.experiment.experiment.transfer
        root_cfg = config
        self.cfg     = root_cfg.experiment.experiment.transfer
        self.device  = torch.device(self.cfg.device)
        self.data_fp = root_cfg.experiment.experiment.preprocessed_file

        # Phase2 mixup toggle & alpha
        self.do_mixup    = bool(self.cfg.phase2_aug)
        self.mixup_alpha = float(root_cfg.augment.augmentations.mixup.alpha)

        self._pretrained_weights = None
        self.model     = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run(self):
        # 1) Load all preprocessed epochs
        with open(self.data_fp, "rb") as f:
            preprocessed_data = pickle.load(f)

        subject_ids = sorted(preprocessed_data.keys())
        all_results = defaultdict(list)

        # 2) Load pretrained MTL weights if not from scratch
        if not self.cfg.init_from_scratch:
            state = torch.load(self.cfg.pretrained_mtl_model, map_location=self.device)
            self._pretrained_weights = _prefix_mtl_keys(state)

        # 3) Multiple runs
        for run_idx in range(self.cfg.n_runs):
            seed = self.cfg.seed_start + run_idx
            self._set_seed(seed)

            out_dir = os.path.join(self.cfg.tl_model_output, f"run_{run_idx}")
            os.makedirs(out_dir, exist_ok=True)

            # iterate subjects
            for subj in subject_ids:
                weights_path = os.path.join(out_dir, f"tl_{subj}_model.pth")
                results_path = os.path.join(out_dir, f"tl_{subj}_results.pkl")

                X_tr, y_tr, X_te, y_te = self._load_subject_data(preprocessed_data, subj)
                self.model     = self._build_model(X_tr, subj)
                self.optimizer = self._build_optimizer()

                train_loader, test_loader = self._build_dataloaders(X_tr, y_tr, X_te, y_te)
                self._train(train_loader, subj)
                wrapper = self._evaluate(test_loader, subj)

                wrapper.save(results_path)
                torch.save(self.model.state_dict(), weights_path)
                all_results[subj].append(wrapper)

        return all_results

    def _load_subject_data(self, preprocessed_data, subject_id: int):
        subj_data = preprocessed_data[subject_id]
        X_train = subj_data["train"].get_data()
        y_train = subj_data["train"].events[:, -1]
        X_test  = subj_data["test"].get_data()
        y_test  = subj_data["test"].events[:, -1]
        return X_train, y_train, X_test, y_test

    def _build_model(self, X_train: np.ndarray, subject_id: int):
        # number of EEG channels × timepoints per epoch
        n_chans = X_train.shape[1]
        window  = X_train.shape[2]
        head_kwargs = {
            "hidden_dim": self.cfg.head_hidden_dim,
            "dropout":    self.cfg.head_dropout
        }
        model = TLModel(
            #n_chans=self.cfg.model.n_outputs,  # or X_train.shape[1]
            n_chans=n_chans,   
            n_outputs=self.cfg.model.n_outputs,
            n_clusters_pretrained=self.cfg.model.n_clusters_pretrained,
            window_samples=window,
            head_kwargs=head_kwargs
        )

        # load pretrained
        if self._pretrained_weights is not None:
            model.load_state_dict(self._pretrained_weights)
            logger.info("Loaded pretrained MTL weights")
        else:
            logger.info("Training from scratch")

        # partial or full freeze
        if self.cfg.freeze_until_layer:
            freeze_backbone_layers(model.shared_backbone, self.cfg.freeze_until_layer)
            logger.info(f"Froze backbone up to {self.cfg.freeze_until_layer}")
        elif self.cfg.freeze_backbone:
            for p in model.shared_backbone.parameters():
                p.requires_grad = False
            logger.info("Froze entire backbone")

        # add a new head for this subject
        model.add_new_head(subject_id)
        return model.to(self.device)

    def _build_optimizer(self):
        b_params, h_params = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "shared_backbone" in n:
                b_params.append(p)
            else:
                h_params.append(p)
        return torch.optim.Adam([
            {"params": b_params, "lr": self.cfg.backbone_lr, "weight_decay": self.cfg.weight_decay},
            {"params": h_params, "lr": self.cfg.head_lr,     "weight_decay": 0.0}
        ])

    def _build_dataloaders(self, X_tr, y_tr, X_te, y_te):
        train_ds = TLSubjectDataset(X_tr, y_tr)
        test_ds  = TLSubjectDataset(X_te, y_te)
        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True)
        test_loader  = DataLoader(test_ds, batch_size=self.cfg.batch_size, shuffle=False)
        return train_loader, test_loader

    def _train(self, train_loader, subject_id: int):
        self.model.train()
        for epoch in range(1, self.cfg.epochs + 1):
            total_loss, correct, count = 0, 0, 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.cfg.epochs}", leave=False)
            for X, y in pbar:
                # Phase2 mixup on raw windows
                if self.do_mixup:
                    X_np = X.detach().cpu().numpy()
                    y_np = y.detach().cpu().numpy()
                    Xm, ya, yb, lam = mixup_batch(X_np, y_np, self.mixup_alpha)
                    X  = torch.from_numpy(Xm).to(self.device)
                    ya = torch.from_numpy(ya).long().to(self.device)
                    yb = torch.from_numpy(yb).long().to(self.device)
                else:
                    X, y = X.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X, [subject_id] * X.size(0))
                if self.do_mixup:
                    loss = lam * self.criterion(outputs, ya) + (1 - lam) * self.criterion(outputs, yb)
                else:
                    loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

                preds = outputs.argmax(dim=1)
                correct += (preds == (ya if self.do_mixup else y)).sum().item()
                count += y.size(0)
                total_loss += loss.item() * y.size(0)
                pbar.set_postfix(loss=f"{total_loss/count:.4f}", acc=f"{correct/count:.4f}")

            print(f"[TLTrainer] Epoch {epoch}: Loss={total_loss/count:.4f}, Acc={correct/count:.4f}")

    def _evaluate(self, test_loader, subject_id: int):
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
            predictions =np.array(all_preds,  dtype=int)
        )
