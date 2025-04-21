import os
import random
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from omegaconf import OmegaConf, DictConfig
from lib.dataset.dataset import EEGMultiTaskDataset
from lib.mtl.model import MultiTaskDeep4Net
from lib.pipeline.cluster.cluster import SubjectClusterer
from lib.utils.utils import convert_state_dict_keys
from lib.logging import logger

logger = logger.get()


class MTLWrapper:
    """
    Wraps multi‑run, per‑subject MTL results, including an explicit
    subject_id→cluster_id mapping of the SAME TYPE as your raw_dict keys.
    """
    def __init__(self, results_by_subject, cluster_assignments, additional_info):
        self.results_by_subject  = results_by_subject
        self.cluster_assignments = cluster_assignments
        self.additional_info     = additional_info

    def save(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"MTL results saved to {filename}")


class MTLTrainer:
    def __init__(self,
                 experiment_cfg: DictConfig | str = "config/experiment/mtl.yaml",
                 model_cfg:      DictConfig | str = "config/model/deep4net.yaml"):
        self.experiment_cfg = (OmegaConf.load(experiment_cfg)
                               if isinstance(experiment_cfg, str)
                               else experiment_cfg)
        self.model_cfg      = (OmegaConf.load(model_cfg)
                               if isinstance(model_cfg, str)
                               else model_cfg)

        exp = self.experiment_cfg.experiment
        self.raw_fp       = exp.preprocessed_file
        self.features_fp  = exp.features_file

        self.cluster_cfg  = exp.clustering
        self.mtl_cfg      = exp.mtl
        self.train_cfg    = exp.mtl.training

        os.makedirs(exp.model_output_dir, exist_ok=True)
        self.wrapper_path = os.path.join(exp.model_output_dir, "mtl_wrapper.pkl")
        self.weights_path = os.path.join(exp.model_output_dir, "mtl_weights.pth")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> MTLWrapper:
        # 1) Load & flatten raw EEG across subjects
        with open(self.raw_fp, "rb") as f:
            raw_dict = pickle.load(f)  # subject_id → {'0train', '1test'}

        X_tr, y_tr, sid_tr = [], [], []
        X_te, y_te, sid_te = [], [], []
        for subj_id, splits in raw_dict.items():
            ep_tr = splits['0train']
            X_tr.append(ep_tr.get_data())
            y_tr.append(ep_tr.events[:,2])
            sid_tr += [subj_id]*len(ep_tr.events)

            ep_te = splits['1test']
            X_te.append(ep_te.get_data())
            y_te.append(ep_te.events[:,2])
            sid_te += [subj_id]*len(ep_te.events)

        X_tr = np.concatenate(X_tr, axis=0)
        y_tr = np.concatenate(y_tr)
        sid_tr = np.array(sid_tr)

        X_te = np.concatenate(X_te, axis=0)
        y_te = np.concatenate(y_te)
        sid_te = np.array(sid_te)

        # 2) Cluster
        subject_clusterer = SubjectClusterer(
            self.features_fp,
            OmegaConf.to_container(self.cluster_cfg, resolve=True)
        )
        cluster_wrapper = subject_clusterer.cluster_subjects(
            method=self.cluster_cfg.method
        )
        n_clusters = cluster_wrapper.get_num_clusters()

        # Build an explicit subject→cluster dict, keys are EXACTLY the same type
        assignments = {sid: cluster_wrapper.labels[sid]
                       for sid in cluster_wrapper.subject_ids}

        # --- NEW: restrict to one cluster if requested ---
        exp = self.experiment_cfg.experiment
        if getattr(exp, "restrict_to_cluster", False):
            if exp.cluster_id is None:
                raise ValueError("cluster_id must be set when restrict_to_cluster is True")
            # Build assignments map one time
            assignments = {sid: cluster_wrapper.labels[sid]
                        for sid in cluster_wrapper.subject_ids}
            # Now filter by cluster label, not subject ID
            mask_tr = np.array([assignments[s] == exp.cluster_id for s in sid_tr])
            mask_te = np.array([assignments[s] == exp.cluster_id for s in sid_te])
            X_tr, y_tr, sid_tr = X_tr[mask_tr], y_tr[mask_tr], sid_tr[mask_tr]
            X_te, y_te, sid_te = X_te[mask_te], y_te[mask_te], sid_te[mask_te]
            logger.info(f"Restricted MTL data to cluster {exp.cluster_id}: "
                        f"{X_tr.shape[0]} train samples, {X_te.shape[0]} test samples")
        # ----------------------------------------------------

        # 3) Datasets & loaders
        train_ds = EEGMultiTaskDataset(X_tr, y_tr, sid_tr, cluster_wrapper)
        eval_ds  = EEGMultiTaskDataset(X_te, y_te, sid_te, cluster_wrapper)
        train_loader = DataLoader(train_ds, batch_size=self.train_cfg.batch_size, shuffle=True)
        eval_loader  = DataLoader(eval_ds,  batch_size=self.train_cfg.batch_size, shuffle=False)

        # 4) Hyperparam lists
        lrs = OmegaConf.to_container(self.train_cfg.learning_rate, resolve=True)
        lbs = OmegaConf.to_container(self.train_cfg.lambda_bias,   resolve=True)
        if not isinstance(lrs, list): lrs = [lrs]*self.train_cfg.n_runs
        if not isinstance(lbs, list): lbs = [lbs]*self.train_cfg.n_runs

        results_by_subject = {sid: [] for sid in set(sid_tr) | set(sid_te)}

        # 5) Multi‑run train+eval
        for run_idx in range(self.train_cfg.n_runs):
            seed = self.train_cfg.seed_start + run_idx
            lr, λb = float(lrs[run_idx]), float(lbs[run_idx])
            self._set_seed(seed)
            print(f"\n>>> Run {run_idx+1}/{self.train_cfg.n_runs} (seed={seed}, lr={lr}, λ_bias={λb})")

            model     = self._build_model(X_tr.shape[1], n_clusters)
            optimizer = self._build_optimizer(model, lr)
            criterion = self._build_criterion()

            self._train(model, train_loader, criterion, optimizer, λb)
            sids, gt, pred = self._evaluate(model, eval_loader)

            # per‑subject aggregation
            run_map = {}
            for sid, t, p in zip(sids, gt, pred):
                run_map.setdefault(sid, {"ground_truth": [], "predictions": []})
                run_map[sid]["ground_truth"].append(t)
                run_map[sid]["predictions"].append(p)
            for sid, res in run_map.items():
                results_by_subject[sid].append({
                    "ground_truth": np.array(res["ground_truth"]),
                    "predictions":  np.array(res["predictions"])
                })

        # 6) Wrap & save
        wrapper = MTLWrapper(
            results_by_subject  = results_by_subject,
            cluster_assignments = assignments,
            additional_info     = OmegaConf.to_container(self.train_cfg, resolve=True)
        )
        wrapper.save(self.wrapper_path)

        state = convert_state_dict_keys(model.state_dict())
        torch.save(state, self.weights_path)
        self.model = model
        return wrapper

    # — all other methods (_set_seed, _build_model, _build_optimizer,
    #    _build_criterion, _train, _evaluate) remain unchanged — 


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

    def _build_optimizer(self, model, lr: float):
        # Separate weight‑decay for weights vs. biases
        wd = float(self.train_cfg.optimizer.weight_decay)

        decay_params, no_decay_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # treat any bias or norm term as no_decay
            if name.endswith(".bias") or "norm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return torch.optim.Adam([
            {"params": decay_params,    "weight_decay": wd},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=lr)

    def _build_criterion(self):
        if self.train_cfg.loss == "cross_entropy":
            return torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss: {self.train_cfg.loss}")

    def _train(self, model, loader, criterion, optimizer, lambda_bias: float):
        model.to(self.device)
        for epoch in range(self.train_cfg.epochs):
            model.train()
            total_loss, correct, count = 0.0, 0, 0

            batch_iter = tqdm(
                loader,
                desc=f"Epoch {epoch+1}/{self.train_cfg.epochs}",
                unit="batch",
                leave=False,
            )
            for X, y, _, cids in batch_iter:
                X = X.to(self.device, dtype=torch.float)
                y = y.to(self.device, dtype=torch.long)
                cids = torch.tensor(cids, dtype=torch.long, device=self.device)

                optimizer.zero_grad()
                outputs = model(X, cids)
                loss    = criterion(outputs, y)

                # explicit bias‑penalty
                penalty = sum(torch.sum(h.bias**2) for h in model.heads.values())
                loss = loss + lambda_bias * penalty

                loss.backward()
                optimizer.step()

                bs = X.size(0)
                total_loss += loss.item() * bs
                preds      = outputs.argmax(dim=1)
                correct   += (preds == y).sum().item()
                count     += bs

                batch_iter.set_postfix({
                    "loss": f"{(total_loss/count):.4f}",
                    "acc":  f"{(correct/count):.4f}"
                })

            avg_loss = total_loss / count
            acc      = correct / count
            print(f"[Epoch {epoch+1}/{self.train_cfg.epochs}] "
                  f"Loss={avg_loss:.4f}, Acc={acc:.4f}")

    def _evaluate(self, model, loader):
        model.to(self.device)
        model.eval()
        sids_list, true_list, pred_list = [], [], []

        with torch.no_grad():
            for X, y, sids, cids in loader:
                X = X.to(self.device, dtype=torch.float)
                y = y.to(self.device, dtype=torch.long)
                cids = torch.tensor(cids, dtype=torch.long, device=self.device)

                outputs = model(X, cids)
                preds   = outputs.argmax(dim=1)

                # convert tensors to Python ints/lists
                sids_list.extend(sids.cpu().numpy().tolist())
                true_list.extend(y.cpu().numpy().tolist())
                pred_list.extend(preds.cpu().numpy().tolist())

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