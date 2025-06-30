import hashlib
import os
import pickle
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torch import nn
from omegaconf import DictConfig
from collections import defaultdict
from lib.dataset.dataset import TLSubjectDataset
from lib.tl.model import TLModel
from lib.tl.evaluate import TLEvaluator
from lib.tl.recorder import make_tl_init_args, save_for_recorder
from lib.utils.utils import _prefix_mtl_keys
from lib.logging import logger

logger = logger.get()


# Helper to freeze backbone up to given conv block index
def freeze_backbone_layers(backbone, freeze_until_layer=None):
    """
    Freezes all backbone layers up to (and including) freeze_until_layer.
    If freeze_until_layer is None, freezes the entire backbone.
    """
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
        tcfg = config.experiment.experiment.transfer
        self.config          = tcfg
        self.device          = torch.device(tcfg.device)
        self.preproc_fp      = config.experiment.experiment.preprocessed_file
        self.mtl_weights_fp  = tcfg.pretrained_mtl_model
        self.mtl_wrapper_fp  = tcfg.pretrained_mtl_wrapper
        self.cluster_model_fp= tcfg.pretrained_cluster_model
        self.tl_output_dir   = tcfg.tl_model_output
        os.makedirs(self.tl_output_dir, exist_ok=True)

        # loss
        self.criterion = nn.CrossEntropyLoss()

        # load MTL init‐weights
        if not tcfg.init_from_scratch:
            state = torch.load(self.mtl_weights_fp, map_location=self.device)
            self.init_state = _prefix_mtl_keys(state)
        else:
            self.init_state = None

    def _set_seed(self, seed: int):
        import random
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run(self):
        # 1) Load all data and clustering info
        with open(self.preproc_fp, "rb") as f:
            preproc = pickle.load(f)  # dict: subj -> {"train", "test"}
        mtl_wrapper = pickle.load(open(self.mtl_wrapper_fp, "rb"))
        assignments = mtl_wrapper.cluster_assignments       # subj -> cluster_id
        n_clusters  = max(assignments.values()) + 1

        # Build per‐subject datasets
        train_ds = {}
        test_ds  = {}
        for sid, splits in preproc.items():
            Xtr = splits["train"].get_data()
            ytr = splits["train"].events[:, -1]
            train_ds[sid] = TLSubjectDataset(Xtr, ytr)

            Xte = splits["test"].get_data()
            yte = splits["test"].events[:, -1]
            test_ds[sid]  = TLSubjectDataset(Xte, yte)

        # Figure out shapes
        example = next(iter(train_ds.values()))
        n_chans, window_samples = example.X.shape[1], example.X.shape[2]

        # Prepare results container
        all_results = defaultdict(list)

        # --- 2) Train pooled model ---
        logger.info("=== Training pooled TL model ===")
        self._set_seed(self.config.seed_start)

        pooled_loader = DataLoader(
            ConcatDataset(list(train_ds.values())),
            batch_size=self.config.batch_size,
            shuffle=True
        )

        pooled_model = TLModel(
            n_chans=n_chans,
            n_outputs=self.config.model.n_outputs,
            n_clusters_pretrained=self.config.model.n_clusters_pretrained,
            window_samples=window_samples,
            head_kwargs={
                "hidden_dim": self.config.head_hidden_dim,
                "dropout":    self.config.head_dropout
            }
        ).to(self.device)

        # load MTL backbone weights
        if self.init_state is not None:
            pooled_model.load_state_dict(self.init_state, strict=False)
            logger.info("Initialized pooled model from MTL weights")

        # create the 'subj_0' head for pooled
        pooled_model.add_new_head(0)

        optimizer = torch.optim.Adam([
            {"params":[p for n,p in pooled_model.named_parameters() if "shared_backbone" in n],
             "lr": self.config.backbone_lr, "weight_decay": self.config.weight_decay},
            {"params":[p for n,p in pooled_model.named_parameters() if "shared_backbone" not in n],
             "lr": self.config.head_lr,     "weight_decay": 0.0},
        ])

        self._train(pooled_model, pooled_loader, optimizer, head_id=0)

        # save
        pooled_fp = os.path.join(self.tl_output_dir, "pooled_model.pth")
        torch.save(pooled_model.state_dict(), pooled_fp)
        
        init_args = make_tl_init_args(self.config, n_chans, window_samples)
        save_for_recorder(
            model      = pooled_model,
            init_args  = init_args,
            pth_path   = os.path.join(self.tl_output_dir, "pooled_model.pth"),
            device     = self.config.device
        )
        logger.info(f"Saved pooled model to {pooled_fp}")

        # evaluate per‐subject
        for sid in test_ds:
            wrapper = self._evaluate_model_on_subject(pooled_model, test_ds[sid], subject_id=0)
            all_results[sid].append(wrapper)

        # --- 3) Train cluster‐specific models ---
        logger.info(f"=== Training {n_clusters} cluster‐specific models ===")
        for cid in range(n_clusters):
            members = [s for s,c in assignments.items() if c==cid]
            if not members:
                logger.warning(f"No subjects in cluster {cid}, skipping")
                continue

            self._set_seed(self.config.seed_start + cid)
            loader = DataLoader(
                ConcatDataset([train_ds[s] for s in members]),
                batch_size=self.config.batch_size,
                shuffle=True
            )

            model = TLModel(
                n_chans=n_chans,
                n_outputs=self.config.model.n_outputs,
                n_clusters_pretrained=self.config.model.n_clusters_pretrained,
                window_samples=window_samples,
                head_kwargs={
                    "hidden_dim": self.config.head_hidden_dim,
                    "dropout":    self.config.head_dropout
                }
            ).to(self.device)

            if self.init_state is not None:
                model.load_state_dict(self.init_state, strict=False)

            # add head for this cluster
            model.add_new_head(cid)

            optimizer = torch.optim.Adam([
                {"params":[p for n,p in model.named_parameters() if "shared_backbone" in n],
                 "lr": self.config.backbone_lr, "weight_decay": self.config.weight_decay},
                {"params":[p for n,p in model.named_parameters() if "shared_backbone" not in n],
                 "lr": self.config.head_lr,     "weight_decay": 0.0},
            ])

            self._train(model, loader, optimizer, head_id=cid)

            out_fp = os.path.join(self.tl_output_dir, f"base_model_cluster{cid}.pth")
            torch.save(model.state_dict(), out_fp)
            init_args = make_tl_init_args(self.config, n_chans, window_samples)
            save_for_recorder(
                model      = model,
                init_args  = init_args,
                pth_path   = os.path.join(self.tl_output_dir, f"base_model_cluster{cid}.pth"),
                device     = self.config.device
            )
            logger.info(f"Saved cluster {cid} model to {out_fp}")

            # evaluate on each member
            for sid in members:
                wrapper = self._evaluate_model_on_subject(model, test_ds[sid], subject_id=cid)
                all_results[sid].append(wrapper)

        logger.info("=== TLTrainer complete ===")
        return all_results


    def _train(self, model, loader, optimizer, head_id: int):
        """Train `model` on `loader`, always routing through `subj_{head_id}`."""
        model.train()
        for epoch in range(1, self.config.epochs + 1):
            total, correct = 0, 0
            for X, y in tqdm(loader, desc=f"Epoch {epoch}/{self.config.epochs}", leave=False):
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                preds = model(X, [head_id]*X.size(0))
                loss  = self.criterion(preds, y)
                loss.backward()
                optimizer.step()

                p = preds.argmax(dim=1)
                correct += (p == y).sum().item()
                total   += y.size(0)
            logger.info(f"[TLTrainer] Epoch {epoch}: acc={correct/total:.4f}")


    def _evaluate_model_on_subject(self, model, dataset: TLSubjectDataset, subject_id: int):
        """Run model on one subject’s test set, return a TLWrapper."""
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X,y in loader:
                X = X.to(self.device)
                out = model(X, [subject_id]*X.size(0))
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
        return TLWrapper(
            ground_truth = np.array(all_labels, dtype=int),
            predictions  = np.array(all_preds, dtype=int)
        )
