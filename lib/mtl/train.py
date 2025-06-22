import os
import random
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from omegaconf import OmegaConf, DictConfig

from braindecode.models import Deep4Net
from lib.dataset.dataset import EEGMultiTaskDataset
from lib.pipeline.cluster.cluster import SubjectClusterer
from lib.utils.utils import convert_state_dict_keys
from lib.logging import logger
from lib.augment.augment import apply_raw_augmentations  # <— import augmentation helper
from lib.mtl.model import MultiTaskDeep4Net

logger = logger.get()


class MTLWrapper:
    def __init__(self, results_by_subject, cluster_assignments, additional_info):
        self.results_by_subject  = results_by_subject
        self.cluster_assignments = cluster_assignments
        self.additional_info     = additional_info

    def save(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)


class MTLTrainer:
    def __init__(self,
                 root_cfg:      DictConfig,
                 model_cfg:     DictConfig | str = "config/model/deep4net.yaml"):
        """
        root_cfg: the full OmegaConf config (with dataset, augment, experiment, etc.)
        model_cfg: path or DictConfig for the model-specific settings
        """
        # 1) Save the root config so we can pass it downstream if needed
        self.root_cfg = root_cfg

        # 2) Experiment settings live under root_cfg.experiment.experiment
        #    (first 'experiment' is the group name, second is the block in transfer.yaml)
        self.experiment_cfg = root_cfg.experiment.experiment

        # 3) Augmentation settings live under root_cfg.augment.augmentations
        self.augment_cfg    = root_cfg.augment.augmentations

        # 4) Load the model-specific config as before
        self.model_cfg = (
            OmegaConf.load(model_cfg) if isinstance(model_cfg, str)
            else model_cfg
        )

        # 5) Unpack experiment sub-blocks
        exp = self.experiment_cfg
        self.raw_fp       = exp.preprocessed_file
        self.features_fp  = exp.features_file
        self.cluster_cfg  = exp.clustering
        self.mtl_cfg      = exp.mtl
        self.train_cfg    = exp.mtl.training

        # 6) Prepare output dirs
        os.makedirs(exp.mtl.mtl_model_output, exist_ok=True)
        self.wrapper_path = os.path.join(exp.mtl.mtl_model_output, "mtl_wrapper.pkl")
        self.weights_path = os.path.join(exp.mtl.mtl_model_output, "mtl_weights.pth")

        # 7) Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> MTLWrapper:
        # 1) Load raw preprocessed epochs
        with open(self.raw_fp, "rb") as f:
            raw_dict = pickle.load(f)

        # 2) Build train/test arrays, applying Phase 1 augmentation on train
        X_tr, y_tr, sid_tr = [], [], []
        X_te, y_te, sid_te = [], [], []

        do_aug = self.experiment_cfg.transfer.phase1_aug
        aug_cfg = self.augment_cfg

        for subj_id, splits in raw_dict.items():
            # ----- TRAIN SPLIT -----
            ep_tr = splits['train']
            X = ep_tr.get_data()           # shape (n_trials, n_ch, n_times)
            if do_aug:
                X = apply_raw_augmentations(X, aug_cfg)
            y = ep_tr.events[:, 2]         # labels
            X_tr.append(X)
            y_tr.append(y)
            sid_tr += [subj_id] * len(y)

            # ----- TEST SPLIT (no augmentation) -----
            ep_te = splits['test']
            Xv = ep_te.get_data()
            yv = ep_te.events[:, 2]
            X_te.append(Xv)
            y_te.append(yv)
            sid_te += [subj_id] * len(yv)

        # Concatenate across subjects
        X_tr   = np.concatenate(X_tr, axis=0)
        y_tr   = np.concatenate(y_tr)
        sid_tr = np.array(sid_tr)

        X_te   = np.concatenate(X_te, axis=0)
        y_te   = np.concatenate(y_te)
        sid_te = np.array(sid_te)

        # 3) Cluster subjects (on saved features)
        subject_clusterer = SubjectClusterer(
            self.features_fp,
            OmegaConf.to_container(self.cluster_cfg, resolve=True)
        )
        cluster_wrapper = subject_clusterer.cluster_subjects(
            method=self.cluster_cfg.method
        )
        n_clusters = cluster_wrapper.get_num_clusters()
        assignments = {sid: cluster_wrapper.labels[sid]
                       for sid in cluster_wrapper.subject_ids}

        # Optionally restrict to a single cluster (if configured)
        exp = self.experiment_cfg
        if getattr(exp, "restrict_to_cluster", False):
            if exp.cluster_id is None:
                raise ValueError("cluster_id must be set when restrict_to_cluster is True")
            mask_tr = np.array([assignments[s] == exp.cluster_id for s in sid_tr])
            mask_te = np.array([assignments[s] == exp.cluster_id for s in sid_te])
            X_tr, y_tr, sid_tr = X_tr[mask_tr], y_tr[mask_tr], sid_tr[mask_tr]
            X_te, y_te, sid_te = X_te[mask_te], y_te[mask_te], sid_te[mask_te]
            logger.info(f"Restricted MTL data to cluster {exp.cluster_id}")

        # 4) Build Datasets and Loaders
        train_ds = EEGMultiTaskDataset(X_tr, y_tr, sid_tr, cluster_wrapper)
        eval_ds  = EEGMultiTaskDataset(X_te, y_te, sid_te, cluster_wrapper)
        train_loader = DataLoader(train_ds, batch_size=self.train_cfg.batch_size, shuffle=True)
        eval_loader  = DataLoader(eval_ds,  batch_size=self.train_cfg.batch_size, shuffle=False)

        # 5) Training loop
        lrs = OmegaConf.to_container(self.train_cfg.learning_rate, resolve=True)
        lbs = OmegaConf.to_container(self.train_cfg.lambda_bias,   resolve=True)
        if not isinstance(lrs, list): lrs = [lrs] * self.train_cfg.n_runs
        if not isinstance(lbs, list): lbs = [lbs] * self.train_cfg.n_runs

        results_by_subject = {sid: [] for sid in set(sid_tr) | set(sid_te)}

        for run_idx in range(self.train_cfg.n_runs):
            seed = self.train_cfg.seed_start + run_idx
            lr, λb = float(lrs[run_idx]), float(lbs[run_idx])
            self._set_seed(seed)

            model     = self._build_model(X_tr.shape[1], n_clusters)
            optimizer = self._build_optimizer(model, lr)
            criterion = self._build_criterion()

            self._train(model, train_loader, criterion, optimizer, λb)
            sids, gt, pred = self._evaluate(model, eval_loader)

            # collect per-subject results
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

        # 6) Package and save
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

    # ... rest of class unchanged (_set_seed, _build_model, _train, _evaluate, etc.) ...



    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_model(self, n_chans: int, n_clusters: int):
        backbone_kwargs = OmegaConf.to_container(self.mtl_cfg.backbone, resolve=True)
        head_kwargs = OmegaConf.to_container(self.mtl_cfg.model.head, resolve=True)

        return MultiTaskDeep4Net(
            n_chans         = n_chans,
            n_outputs       = self.mtl_cfg.model.n_outputs,
            n_clusters      = n_clusters,
            backbone_kwargs = backbone_kwargs,
            head_kwargs     = head_kwargs,
        )

    def _build_optimizer(self, model, lr: float):
        wd = float(self.train_cfg.optimizer.weight_decay)

        decay_params, no_decay_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
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

                penalty = sum(
                    torch.sum(p**2) for h in model.heads.values()
                    for n, p in h.named_parameters() if "bias" in n
                )
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
            print(f"[MTLTrainer] Epoch {epoch+1}/{self.train_cfg.epochs}] "
                  f"Loss={avg_loss:.4f}, Acc={acc:.4f}")

    def _evaluate(self, model, loader):
        model.to(self.device)
        model.eval()
        sids_list, true_list, pred_list = [], [], []
    
        
        def to_numpy_list(x):
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy().tolist()
            return np.asarray(x).tolist()
        
        with torch.no_grad():
            for X, y, sids, cids in loader:
                X = X.to(self.device, dtype=torch.float)
                y = y.to(self.device, dtype=torch.long)
                cids = torch.tensor(cids, dtype=torch.long, device=self.device)

                outputs = model(X, cids)
                preds   = outputs.argmax(dim=1)

                sids_list.extend(to_numpy_list(sids))
                true_list.extend(to_numpy_list(y))
                pred_list.extend(to_numpy_list(preds))

        acc = (np.array(pred_list) == np.array(true_list)).mean()

        return sids_list, true_list, pred_list
