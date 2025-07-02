import os
import pickle
import random
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torch import nn
from omegaconf import DictConfig
from collections import defaultdict
from lib.dataset.dataset import TLSubjectDataset
from lib.tl.model import TLModel
from lib.utils.utils import _prefix_mtl_keys
from lib.logging import logger

logger = logger.get()


def freeze_backbone_layers(backbone, freeze_until_layer=None):
    """
    Freezes all backbone layers up to (and including) `freeze_until_layer`.
    If `freeze_until_layer` is None, freezes the entire backbone.
    """
    found = False
    for name, module in backbone.named_children():
        for p in module.parameters():
            p.requires_grad = False
        if freeze_until_layer is not None and name == freeze_until_layer:
            found = True
            break
    if freeze_until_layer and not found:
        raise ValueError(f"Layer {freeze_until_layer} not found; available: "
                         f"{[n for n,_ in backbone.named_children()]}")


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
    """
    mode='offline' → per‐subject TL (old)
    mode='online'  → pooled + clustered TL (new)
    """
    def __init__(self, config: DictConfig):
        self.transfer_cfg = config.experiment.experiment.transfer
        self.mode   = self.transfer_cfg.get("mode", "offline")
        self.device = torch.device(self.transfer_cfg.device)

        self.preproc_fp = config.experiment.experiment.preprocessed_file

        # where to grab MTL weights & clusters
        mtl_out = config.experiment.experiment.mtl.mtl_model_output
        self.mtl_weights_fp  = self.transfer_cfg.pretrained_mtl_model
        self.mtl_wrapper_fp  = os.path.join(mtl_out, "mtl_wrapper.pkl")

        # TL outputs dir
        self.output_dir = self.transfer_cfg.tl_model_output
        os.makedirs(self.output_dir, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()
        # load & fix prefix of pretrained MTL
        self._init_state = None
        if not self.transfer_cfg.init_from_scratch:
            state = torch.load(self.mtl_weights_fp, map_location=self.device)
            self._init_state = _prefix_mtl_keys(state)

        self.model     = None
        self.optimizer = None

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run(self):
        with open(self.preproc_fp, "rb") as f:
            preproc = pickle.load(f)
        if self.mode == "offline":
            return self._run_offline(preproc)
        else:
            return self._run_online(preproc)

    # ---------------------
    # OLD per‐subject TL
    # ---------------------
    def _run_offline(self, preproc):
        subject_ids = sorted(preproc.keys())
        all_results = defaultdict(list)

        for run_idx in range(self.transfer_cfg.n_runs):
            self._set_seed(self.transfer_cfg.seed_start + run_idx)
            run_dir = os.path.join(self.output_dir, f"run_{run_idx}")
            os.makedirs(run_dir, exist_ok=True)

            for sid in subject_ids:
                # load subject data
                ep = preproc[sid]
                Xtr, ytr = ep["train"].get_data(), ep["train"].events[:,-1]
                Xte, yte = ep["test"].get_data(),  ep["test"].events[:,-1]

                # build, train, eval
                self.model     = self._build_model(Xtr, sid)
                self.optimizer = self._build_optimizer()

                tr_loader, te_loader = self._build_dataloaders(Xtr, ytr, Xte, yte)
                self._train(tr_loader, head_id=sid)
                wrapper = self._evaluate(te_loader, head_id=sid)

                # save weights & results
                wfp = os.path.join(run_dir, f"tl_{sid}_model.pth")
                rfp = os.path.join(run_dir, f"tl_{sid}_results.pkl")
                torch.save(self.model.state_dict(), wfp)
                wrapper.save(rfp)

                all_results[sid].append(wrapper)

        return all_results

    # -----------------------------
    # NEW pooled + clustered TL
    # -----------------------------
    def _run_online(self, preproc):
        # load MTL clustering
        mtl_wrapper = TLWrapper.load(self.mtl_wrapper_fp)
        assignments = mtl_wrapper.cluster_assignments  # subj → cluster
        n_clusters = max(assignments.values()) + 1

        # turn every subject into a TLSubjectDataset
        train_ds = {}
        test_ds  = {}
        for sid, splits in preproc.items():
            Xtr, ytr = splits["train"].get_data(), splits["train"].events[:,-1]
            Xte, yte = splits["test"].get_data(),  splits["test"].events[:,-1]
            train_ds[sid] = TLSubjectDataset(Xtr, ytr)
            test_ds [sid] = TLSubjectDataset(Xte, yte)

        # grab an example to know dims
        example_ds = next(iter(train_ds.values()))
        n_chans       = example_ds.X.shape[1]
        window_samples= example_ds.X.shape[2]

        all_results = defaultdict(list)

        # ----- pooled head (head_id = 0) -----
        self._set_seed(self.transfer_cfg.seed_start)
        pooled_loader = DataLoader(
            ConcatDataset(list(train_ds.values())),
            batch_size=self.transfer_cfg.batch_size, shuffle=True
        )
        # build model + optimizer
        self.model     = TLModel(
            n_chans=n_chans,
            n_outputs=self.transfer_cfg.model.n_outputs,
            n_clusters_pretrained=self.transfer_cfg.model.n_clusters_pretrained,
            window_samples=window_samples,
            head_kwargs={
                "hidden_dim": self.transfer_cfg.head_hidden_dim,
                "dropout":    self.transfer_cfg.head_dropout
            }
        ).to(self.device)
        if self._init_state is not None:
            self.model.load_state_dict(self._init_state, strict=False)
        self.model.add_new_head(0)

        self.optimizer = torch.optim.Adam([
            {"params":[p for n,p in self.model.named_parameters() if "shared_backbone" in n],
             "lr": self.transfer_cfg.backbone_lr, "weight_decay": self.transfer_cfg.weight_decay},
            {"params":[p for n,p in self.model.named_parameters() if "shared_backbone" not in n],
             "lr": self.transfer_cfg.head_lr,     "weight_decay": 0.0},
        ])

        # train + eval pooled
        self._train(pooled_loader, head_id=0)
        for sid, ds in test_ds.items():
            wrapper = self._evaluate(
                DataLoader(ds, batch_size=self.transfer_cfg.batch_size),
                head_id=0
            )
            all_results[sid].append(wrapper)

        # ----- cluster-specific heads -----
        for cid in range(n_clusters):
            members = [s for s,c in assignments.items() if c==cid]
            if not members:
                continue

            self._set_seed(self.transfer_cfg.seed_start + cid)
            loader = DataLoader(
                ConcatDataset([train_ds[s] for s in members]),
                batch_size=self.transfer_cfg.batch_size, shuffle=True
            )

            # build new model + optimizer
            self.model     = TLModel(
                n_chans=n_chans,
                n_outputs=self.transfer_cfg.model.n_outputs,
                n_clusters_pretrained=self.transfer_cfg.model.n_clusters_pretrained,
                window_samples=window_samples,
                head_kwargs={
                    "hidden_dim": self.transfer_cfg.head_hidden_dim,
                    "dropout":    self.transfer_cfg.head_dropout
                }
            ).to(self.device)
            if self._init_state is not None:
                self.model.load_state_dict(self._init_state, strict=False)
            self.model.add_new_head(cid)

            self.optimizer = torch.optim.Adam([
                {"params":[p for n,p in self.model.named_parameters() if "shared_backbone" in n],
                 "lr": self.transfer_cfg.backbone_lr, "weight_decay": self.transfer_cfg.weight_decay},
                {"params":[p for n,p in self.model.named_parameters() if "shared_backbone" not in n],
                 "lr": self.transfer_cfg.head_lr,     "weight_decay": 0.0},
            ])

            self._train(loader, head_id=cid)
            for sid in members:
                wrapper = self._evaluate(
                    DataLoader(test_ds[sid], batch_size=self.transfer_cfg.batch_size),
                    head_id=cid
                )
                all_results[sid].append(wrapper)

        return all_results

    # -----------------
    # Helpers below
    # -----------------
    def _load_subject_data(self, preproc, sid):
        ep = preproc[sid]
        return (ep["train"].get_data(), ep["train"].events[:,-1],
                ep["test"].get_data(),  ep["test"].events[:,-1])

    def _build_model(self, X_train, sid):
        """Used by offline: builds TLModel for subject `sid` and adds head."""
        n_chans, window_samples = X_train.shape[1], X_train.shape[2]
        model = TLModel(
            n_chans=n_chans,
            n_outputs=self.transfer_cfg.model.n_outputs,
            n_clusters_pretrained=self.transfer_cfg.model.n_clusters_pretrained,
            window_samples=window_samples,
            head_kwargs={
                "hidden_dim": self.transfer_cfg.head_hidden_dim,
                "dropout":    self.transfer_cfg.head_dropout
            }
        ).to(self.device)
        if self._init_state is not None:
            model.load_state_dict(self._init_state, strict=False)
        # optional freezing
        freeze_until = self.transfer_cfg.freeze_until_layer
        if freeze_until and freeze_until != "None":
            freeze_backbone_layers(model.shared_backbone, freeze_until_layer=freeze_until)
        elif self.transfer_cfg.freeze_backbone:
            for p in model.shared_backbone.parameters():
                p.requires_grad = False

        model.add_new_head(sid)
        return model

    def _build_optimizer(self):
        bd_params, hd_params = [], []
        for n,p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "shared_backbone" in n:
                bd_params.append(p)
            else:
                hd_params.append(p)
        return torch.optim.Adam([
            {"params": bd_params, "lr": self.transfer_cfg.backbone_lr, "weight_decay": self.transfer_cfg.weight_decay},
            {"params": hd_params, "lr": self.transfer_cfg.head_lr,     "weight_decay": 0.0},
        ])

    def _build_dataloaders(self, Xtr, ytr, Xte, yte):
        train_ds = TLSubjectDataset(Xtr, ytr)
        test_ds  = TLSubjectDataset(Xte, yte)
        return (
            DataLoader(train_ds, batch_size=self.transfer_cfg.batch_size, shuffle=True),
            DataLoader(test_ds,  batch_size=self.transfer_cfg.batch_size, shuffle=False),
        )

    def _train(self, loader, head_id):
        self.model.train()
        for epoch in range(1, self.transfer_cfg.epochs + 1):
            total, correct = 0, 0
            for X,y in tqdm(loader, desc=f"Epoch {epoch}/{self.transfer_cfg.epochs}", leave=False):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(X, [head_id]*X.size(0))
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()
                preds = out.argmax(dim=1)
                correct += (preds==y).sum().item()
                total   += y.size(0)
            logger.info(f"[TLTrainer] Epoch {epoch}: acc={correct/total:.4f}")

    def _evaluate(self, loader, head_id):
        self.model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for X,y in loader:
                X = X.to(self.device)
                out = self.model(X, [head_id]*X.size(0))
                pred = out.argmax(dim=1).cpu().numpy()
                all_p.extend(pred)
                all_y.extend(y.numpy())
        return TLWrapper(
            ground_truth=np.array(all_y, dtype=int),
            predictions =np.array(all_p, dtype=int)
        )
