import os
import pickle
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from collections import defaultdict
from lib.dataset.dataset import TLSubjectDataset
from lib.tl.model import TLModel
from lib.logging import logger
from lib.utils.utils import _prefix_mtl_keys

logger = logger.get()

def freeze_backbone_layers(backbone, freeze_until_layer=None):
    found = False
    for name, module in backbone.named_children():
        for param in module.parameters():
            param.requires_grad = False
        if freeze_until_layer is not None and name == freeze_until_layer:
            found = True
            break
    if freeze_until_layer and not found:
        raise ValueError(f"Layer {freeze_until_layer} not found in backbone (got: {[n for n, _ in backbone.named_children()]})")

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
        self.criterion = torch.nn.CrossEntropyLoss()

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

        # Load only the shared backbone weights from MTL
        if not self.config.init_from_scratch:
            self._pretrained_weights = torch.load(self.config.pretrained_mtl_backbone, map_location=self.device)

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
            # Optionally: head_type, head_kwargs
        )
        # Load only backbone weights if available
        if self._pretrained_weights is not None:
            model.shared_backbone.load_state_dict(self._pretrained_weights, strict=True)
            logger.info("Loaded MTL backbone weights into TL model.")
        else:
            logger.info("Training TL model from scratch.")

        # Partial backbone freeze (up to freeze_until_block, inclusive)
        freeze_until = getattr(self.config, "freeze_until_layer", None)
        if freeze_until is not None:
            freeze_backbone_layers(model.shared_backbone, freeze_until_layer=freeze_until)
            logger.info(f"Froze backbone up to {freeze_until}")
        elif getattr(self.config, "freeze_backbone", False):
            for param in model.shared_backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen.")

        # Always add a new (random) head for this subject
        model.add_new_head(subject_id)
        return model.to(self.device)

    def _build_optimizer(self):
        # Differential learning rates for backbone and head(s)
        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'shared_backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        backbone_lr = getattr(self.config, "backbone_lr", self.config.lr)
        head_lr = getattr(self.config, "head_lr", self.config.lr)
        return torch.optim.Adam([
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": self.config.weight_decay},
            {"params": head_params,     "lr": head_lr,     "weight_decay": 0.0}
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
                outputs = self.model(X, torch.full((X.size(0),), subject_id, dtype=torch.long, device=self.device))
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
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
                outputs = self.model(X, torch.full((X.size(0),), subject_id, dtype=torch.long, device=self.device))
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
        return TLWrapper(
            ground_truth=np.array(all_labels, dtype=int),
            predictions=np.array(all_preds, dtype=int)
        )
