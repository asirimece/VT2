import os
import random
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lib.dataset.dataset import EEGMultiTaskDataset
from omegaconf import DictConfig, OmegaConf

from lib.mtl.model import MultiTaskDeep4Net
from lib.pipeline.cluster.cluster import SubjectClusterer
from lib.utils.utils import convert_state_dict_keys


class MTLWrapper:
    """
    Wraps multi‑run, per‑subject MTL results.
    """
    def __init__(self, results_by_subject, cluster_assignments, additional_info):
        self.results_by_subject   = results_by_subject
        self.cluster_assignments  = cluster_assignments
        self.additional_info      = additional_info

    def save(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"MTL results saved to {filename}")


class MTLTrainer:
    def __init__(self,
                 experiment_config_path="config/experiment/mtl.yaml",
                 model_config_path="config/model/deep4net.yaml"):
        # load config files
        self.experiment_cfg = OmegaConf.load(experiment_config_path)
        self.model_cfg      = OmegaConf.load(model_config_path)

        # now pull out the bits you need
        exp = self.experiment_cfg.experiment
        self.features_fp   = exp.features_file
        self.cluster_cfg   = OmegaConf.to_container(exp.clustering, resolve=True)
        self.mtl_cfg       = exp.mtl
        self.train_cfg     = exp.mtl.training

        # where to dump outputs (must live under evaluators)
        eval_dir = exp.output_dir
        os.makedirs(eval_dir, exist_ok=True)
        self.wrapper_path = os.path.join(eval_dir, "mtl_wrapper.pkl")
        self.weights_path = os.path.join(eval_dir, "mtl_weights.pth")

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> MTLWrapper:
        # 1) Load extracted features: expect a pickle of (X, y, subject_ids)
        with open(self.features_fp, "rb") as f:
            X, y, subject_ids = pickle.load(f)

        # 2) Cluster subjects
        subject_clusterer = SubjectClusterer(self.features_fp, self.cluster_cfg)
        cluster_wrapper   = subject_clusterer.cluster_subjects(
            method=self.config.experiment.clustering.method
        )
        n_clusters = cluster_wrapper.get_num_clusters()

        # Prepare DataLoaders
        dataset = EEGMultiTaskDataset(X, y, subject_ids, cluster_wrapper)
        train_loader = DataLoader(
            dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
        )
        eval_loader = DataLoader(
            dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=False,
        )

        # Storage for per-run, per-subject results
        results_by_subject = {sid: [] for sid in set(subject_ids)}

        # 3) Multi‑run training & evaluation
        for run_idx in range(self.train_cfg.n_runs):
            seed = self.train_cfg.seed_start + run_idx
            self._set_seed(seed)
            print(f"\n>>> MTL run {run_idx+1}/{self.train_cfg.n_runs} (seed={seed})")

            model     = self._build_model(X.shape[1], n_clusters)
            optimizer = self._build_optimizer(model)
            criterion = self._build_criterion()

            self._train(model, train_loader, criterion, optimizer)
            sids, trues, preds = self._evaluate(model, eval_loader)

            # Collect per‑subject for this run
            run_map = {}
            for sid, t, p in zip(sids, trues, preds):
                run_map.setdefault(sid, {"ground_truth": [], "predictions": []})
                run_map[sid]["ground_truth"].append(t)
                run_map[sid]["predictions"].append(p)
            for sid, res in run_map.items():
                results_by_subject[sid].append({
                    "ground_truth": np.array(res["ground_truth"]),
                    "predictions":  np.array(res["predictions"])
                })

        # 4) Wrap & save
        wrapper = MTLWrapper(
            results_by_subject   = results_by_subject,
            cluster_assignments  = cluster_wrapper.labels,
            additional_info      = OmegaConf.to_container(self.train_cfg, resolve=True),
        )
        wrapper.save(self.config.output.mtl_wrapper)

        # Save only model weights (using converted keys)
        state = convert_state_dict_keys(model.state_dict())
        torch.save(state, self.config.output.mtl_weights)

        return wrapper

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_model(self, n_chans: int, n_clusters: int):
        backbone_kwargs = OmegaConf.to_container(self.mtl_cfg.backbone, resolve=True)
        return MultiTaskDeep4Net(
            n_chans         = n_chans,
            n_outputs       = self.mtl_cfg.model.n_outputs,
            n_clusters      = n_clusters,
            backbone_kwargs = backbone_kwargs,
        )

    def _build_optimizer(self, model):
        return torch.optim.Adam(
            model.parameters(),
            lr           = self.train_cfg.learning_rate,
            weight_decay = self.train_cfg.optimizer.weight_decay
        )

    def _build_criterion(self):
        if self.train_cfg.loss == "cross_entropy":
            return torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss: {self.train_cfg.loss}")

    def _train(self, model, loader, criterion, optimizer):
        model.to(self.device)
        for epoch in range(self.train_cfg.epochs):
            model.train()
            total_loss, correct, count = 0.0, 0, 0

            for X, y, _, cids in loader:
                X = X.to(self.device, dtype=torch.float)
                y = y.to(self.device, dtype=torch.long)
                if not torch.is_tensor(cids):
                    cids = torch.tensor(cids, dtype=torch.long)
                cids = cids.to(self.device)

                optimizer.zero_grad()
                outputs = model(X, cids)
                loss    = criterion(outputs, y)

                bias_penalty = torch.tensor(0.0, device=self.device)
                for head in model.heads.values():
                    bias_penalty = bias_penalty + torch.sum(head.bias ** 2)
                loss = loss + self.train_cfg.lambda_bias * bias_penalty

                loss.backward()
                optimizer.step()

                bs = X.size(0)
                total_loss += loss.item() * bs
                preds      = outputs.argmax(dim=1)
                correct   += (preds == y).sum().item()
                count     += bs

            avg_loss = total_loss / count
            acc      = correct / count
            print(f"Epoch {epoch+1}/{self.train_cfg.epochs}: Loss={avg_loss:.4f}, Acc={acc:.4f}")

    def _evaluate(self, model, loader):
        model.eval()
        sids_list, true_list, pred_list = [], [], []

        with torch.no_grad():
            for X, y, sids, cids in loader:
                X = X.to(self.device, dtype=torch.float)
                y = y.to(self.device, dtype=torch.long)
                if not torch.is_tensor(cids):
                    cids = torch.tensor(cids, dtype=torch.long)
                cids = cids.to(self.device)

                outputs = model(X, cids)
                preds   = outputs.argmax(dim=1)

                sids_list .extend(sids)
                true_list .extend(y.cpu().numpy().tolist())
                pred_list .extend(preds.cpu().numpy().tolist())

        acc = (np.array(pred_list) == np.array(true_list)).mean()
        print(f"Evaluation Accuracy: {acc:.4f}")
        return sids_list, true_list, pred_list






"""import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from lib.mtl.model import MultiTaskDeep4Net
import pickle


class MTLWrapper:
    def __init__(self, results_by_subject, training_logs=None, cluster_assignments=None, additional_info=None):
        self.results_by_subject = results_by_subject
        self.training_logs = training_logs if training_logs is not None else {}
        self.cluster_assignments = cluster_assignments if cluster_assignments is not None else {}
        self.additional_info = additional_info if additional_info is not None else {}
    
    def get_subject_results(self, subject):
        return self.results_by_subject.get(subject)
    
    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"MTL results saved to {filename}")
    
    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        # If already an instance of MTLWrapper, return it.
        if isinstance(obj, cls):
            return obj
        # If it's a dict with the keys "ground_truth" and "predictions", then wrap it.
        if isinstance(obj, dict) and ("ground_truth" in obj and "predictions" in obj):
            wrapped = {"pooled": obj}
            print("[DEBUG] Loaded results as dict with keys ['ground_truth', 'predictions']. Wrapping under key 'pooled'.")
            return cls(results_by_subject=wrapped)
        # Otherwise, if it's a dict (assumed to be mapping subject IDs to results), wrap it.
        if isinstance(obj, dict):
            return cls(results_by_subject=obj)
        # If it's a list, wrap it as pooled.
        if isinstance(obj, list):
            wrapped = {"pooled": obj}
            return cls(results_by_subject=wrapped)
        return obj


def train_mtl_model(model, dataloader, criterion, optimizer, device, epochs: int, lambda_bias: float):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for data, labels, subject_ids, cluster_ids in dataloader:
            # move everything to device
            data = data.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            if not torch.is_tensor(cluster_ids):
                cluster_ids = torch.tensor(cluster_ids, dtype=torch.long)
            cluster_ids = cluster_ids.to(device)

            optimizer.zero_grad()
            outputs = model(data, cluster_ids)
            loss = criterion(outputs, labels)

            # --- explicit head-bias penalty ---
            bias_penalty = torch.tensor(0.0, device=device)
            for head in model.heads.values():
                bias_penalty = bias_penalty + torch.sum(head.bias ** 2)
            loss = loss + lambda_bias * bias_penalty

            loss.backward()
            optimizer.step()

            # bookkeeping
            batch_size     = data.size(0)
            epoch_loss    += loss.item() * batch_size
            preds          = outputs.argmax(dim=1)
            epoch_correct += (preds == labels).sum().item()
            epoch_samples += batch_size

        avg_loss = epoch_loss / epoch_samples
        accuracy = epoch_correct / epoch_samples
        print(f"End Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")

    return model


def evaluate_mtl_model(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    all_subjects = []   
    with torch.no_grad():
        for data, labels, subject_ids, cluster_ids in dataloader:
            data = data.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            if not torch.is_tensor(cluster_ids):
                cluster_ids = torch.tensor(cluster_ids, dtype=torch.long)
            cluster_ids = cluster_ids.to(device)
            outputs = model(data, cluster_ids)
            _, preds = torch.max(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += data.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_subjects.extend(subject_ids)  # record subject id for each sample
    accuracy = total_correct / total_samples
    print(f"Evaluation Accuracy: {accuracy:.4f}")
    return all_subjects, all_labels, all_preds
"""