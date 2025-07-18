import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from omegaconf import OmegaConf
import mne
import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm
from lib.dataset.dataset import EEGDataset
from lib.dataset.utils import apply_label_filter
from lib.model.deep4net import Deep4NetModel
from lib.base.evaluate import BaselineEvaluator
from lib.logging import logger

matplotlib.use("Agg")

logger = logger.get()


class BaseWrapper:
    """ 
    Wraps baseline training results. 
    """
    def __init__(self, results_by_experiment):
        self.results_by_experiment = results_by_experiment

    def get_experiment_results(self, key):
        return self.results_by_experiment.get(key)


class BaselineTrainer:
    def __init__(self, cfg):
        exp_cfg = cfg.experiment
        self.device     = exp_cfg.device
        self.model_cfg = cfg.model
        self.single_cfg = exp_cfg.single
        self.pooled_cfg = exp_cfg.pooled

        # 2) Logging paths under cfg.logging
        self.single_results_path = exp_cfg.logging.single_results_path
        self.pooled_results_path = exp_cfg.logging.pooled_results_path
        
        # 3) Preprocessed data path under cfg.data
        preproc_path = exp_cfg.data.preprocessed_data
        with open(preproc_path, "rb") as f:
            self.preprocessed_data = pickle.load(f)
            
        # 4) Dataset config under cfg.dataset
        ds = cfg.dataset
        sup = ds.supervised
        # If the user wrote strings, we map them via name2int; if they wrote ints, we take them as-is.
        name2int = {n: i for n, i in ds.event_markers.items() if isinstance(i, int)}

        def to_label(x):
            return name2int[x] if isinstance(x, str) else x

        self.keep_labels       = { to_label(c) for c in sup.classes }
        self.drop_labels       = { to_label(c) for c in sup.ignore_labels }
        self.supervised_enabled = bool(sup.enabled)

    def _train(self,
                             X_train, y_train, train_ids,
                             X_test,  y_test,  test_ids,
                             model_cfg, train_cfg):

        """Core train + eval loop; uses self.device and retains original tqdm/postfix."""
        model = Deep4NetModel(model_cfg).get_model().to(self.device)

        train_ds = EEGDataset(X_train, y_train, train_ids)
        test_ds = EEGDataset(X_test,  y_test,  test_ids)
        train_loader = DataLoader(train_ds,
                                  batch_size=train_cfg.batch_size,
                                  shuffle=True)
        test_loader = DataLoader(test_ds,
                                  batch_size=train_cfg.batch_size,
                                  shuffle=False)

        Optim = getattr(torch.optim, train_cfg.optimizer)
        optimizer = Optim(model.parameters(),
                          lr=train_cfg.learning_rate,
                          weight_decay=train_cfg.weight_decay)

        criterion = nn.CrossEntropyLoss()

        model.train()
        # === Epoch bar only ===
        epoch_bar = trange(train_cfg.epochs, desc="Training epochs", unit="ep")

        for epoch in epoch_bar:
            total_loss = correct = total = 0

            # If you really want a batch bar, uncomment the next two lines:
            # batch_iter = tqdm(train_loader, desc=f" Ep{epoch+1} batches", leave=False)
            # iterator = batch_iter
            # Otherwise:
            iterator = train_loader

            for Xb, yb, _ in iterator:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = model(Xb)
                loss   = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                preds = logits.argmax(dim=1)
                correct    += (preds == yb).sum().item()
                total      += Xb.size(0)
                total_loss += loss.item() * Xb.size(0)

            # compute stats
            avg_loss = total_loss / total
            acc      = correct / total

            # update **only** the epoch bar
            epoch_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc":  f"{acc:.4f}",
                "wd":   f"{train_cfg.weight_decay}"
            })

        # at this point the bar is complete; now you can log once:
        logger.info(f"[BaselineTrainer] Finished training: "
                    f"final loss={avg_loss:.4f}, acc={acc:.4f}")
    
        model.eval()
        all_logits, all_tids, all_y = [], [], []
        with torch.no_grad():
            for Xb, yb, tid in test_loader:
                Xb = Xb.to(self.device)
                out = model(Xb).cpu().numpy()
                all_logits.append(out)
                all_tids.append(tid.numpy())
                all_y.extend(yb.numpy())

        all_logits = np.concatenate(all_logits, axis=0)
        all_tids   = np.concatenate(all_tids,   axis=0)
        unique_t   = np.unique(all_tids)

        preds, labels = [], []
        for t in unique_t:
            idx = np.where(all_tids == t)[0]
            avg_logit = all_logits[idx].mean(axis=0)
            preds.append(int(avg_logit.argmax()))
            labels.append(int(all_y[idx[0]]))

        # after: build a sorted list of labels to pass through
        if self.supervised_enabled:
            # only the binary classes you care about
            label_list = sorted(self.keep_labels)
        else:
            # grab all possible labels from your model config
            # e.g. n_classes = self.model_cfg["n_classes"]
            # or derive from unique(labels) if dynamic
            label_list = list(np.unique(labels))

        acc   = accuracy_score(labels, preds)
        # explicitly pass the same label ordering to both metrics
        kappa = cohen_kappa_score(labels, preds, labels=label_list)
        cm    = confusion_matrix(labels, preds, labels=label_list)
        logger.info(f"Trial‑level Test - Acc: {acc:.4f}, Kappa: {kappa:.4f}")

        return model, {"ground_truth": labels, "predictions": preds}


    def _train_subject(self, subj, subject_data):
        # 1) Load the raw epoch data
        tr = subject_data["train"]
        te = subject_data["test"]
        Xtr, ytr = tr.get_data(), tr.events[:, -1]
        Xte, yte = te.get_data(), te.events[:, -1]
        tid_tr   = tr.events[:, 1]
        tid_te   = te.events[:, 1]

        # 2) Inline supervised filtering
        if self.supervised_enabled:
            logger.info(f"Subject {subj} ▶ raw train labels: {np.unique(ytr)}")
            mask_tr = np.array([lbl in self.keep_labels for lbl in ytr])
            logger.info(f"Subject {subj} ▶ keep_labels: {self.keep_labels}")
            logger.info(f"Subject {subj} ▶ train mask sum: {mask_tr.sum()}/{len(ytr)}")
            Xtr, ytr, tid_tr = Xtr[mask_tr], ytr[mask_tr], tid_tr[mask_tr]

            logger.info(f"Subject {subj} ▶ raw test labels:  {np.unique(yte)}")
            mask_te = np.array([lbl in self.keep_labels for lbl in yte])
            logger.info(f"Subject {subj} ▶ test  mask sum: {mask_te.sum() }/{len(yte)}")
            Xte, yte, tid_te = Xte[mask_te], yte[mask_te], tid_te[mask_te]

        # 3) Build model config and train
        common = {
            k: self.model_cfg[k]
            for k in ("name","in_chans","n_classes","n_times","final_conv_length")
            if k in self.model_cfg
        }
        merged_cfg = {**common, **self.model_cfg.get("single", {})}

        results_runs = []
        for run_i in range(self.single_cfg.n_runs):
            seed = self.single_cfg.seed_start + run_i
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            logger.info(f"Single subject run {run_i+1}/{self.single_cfg.n_runs} for Subject {subj} (seed={seed})")

            _, trial_res = self._train(
                Xtr, ytr, tid_tr,
                Xte, yte, tid_te,
                merged_cfg, self.single_cfg
            )
            results_runs.append(trial_res)

        return results_runs


    def _train_pooled(self):
        # 1) Gather all subjects' data into lists
        Xtr_list, ytr_list, tid_tr_list = [], [], []
        Xte_list, yte_list, tid_te_list = [], [], []

        for idx, (_, data) in enumerate(self.preprocessed_data.items()):
            tr = data["train"]
            te = data["test"]
            offset = idx * 1_000_000

            Xtr_list.append(tr.get_data())
            ytr_list.append(tr.events[:, -1])
            tid_tr_list.append(tr.events[:, 1] + offset)

            Xte_list.append(te.get_data())
            yte_list.append(te.events[:, -1])
            tid_te_list.append(te.events[:, 1] + offset)

        # 2) Concatenate
        Xtr    = np.concatenate(Xtr_list,   axis=0)
        ytr    = np.concatenate(ytr_list,   axis=0)
        tid_tr = np.concatenate(tid_tr_list,axis=0)
        Xte    = np.concatenate(Xte_list,   axis=0)
        yte    = np.concatenate(yte_list,   axis=0)
        tid_te = np.concatenate(tid_te_list,axis=0)

        # 3) Inline supervised filtering
        if self.supervised_enabled:
            logger.info(f"Pooled ▶ raw train labels: {np.unique(ytr)}")
            logger.info(f"Pooled ▶ keep_labels: {self.keep_labels}")
            mask_tr = np.array([lbl in self.keep_labels for lbl in ytr])
            logger.info(f"Pooled ▶ train mask sum: {mask_tr.sum()}/{len(ytr)}")
            Xtr, ytr, tid_tr = Xtr[mask_tr], ytr[mask_tr], tid_tr[mask_tr]

            logger.info(f"Pooled ▶ raw test  labels: {np.unique(yte)}")
            mask_te = np.array([lbl in self.keep_labels for lbl in yte])
            logger.info(f"Pooled ▶ test  mask sum: {mask_te.sum()}/{len(yte)}")
            Xte, yte, tid_te = Xte[mask_te], yte[mask_te], tid_te[mask_te]

        # 4) Build model config
        common = {
            k: self.model_cfg[k]
            for k in ("name","in_chans","n_classes","n_times","final_conv_length")
            if k in self.model_cfg
        }
        merged_cfg = {**common, **self.model_cfg.get("pooled", {})}

        # 5) Pooled runs
        pooled_runs = []
        for run_i in range(self.pooled_cfg.n_runs):
            seed = self.pooled_cfg.seed_start + run_i
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            logger.info(f"Pooled run {run_i+1}/{self.pooled_cfg.n_runs} (seed={seed})")

            _, trial_res = self._train(
                Xtr, ytr, tid_tr,
                Xte, yte, tid_te,
                merged_cfg, self.pooled_cfg
            )
            pooled_runs.append(trial_res)

        return pooled_runs

    def run(self):
        # ── Single-subject training ──
        single_res = {}
        for subj, data in self.preprocessed_data.items():
            single_res[subj] = self._train_subject(subj, data)
        os.makedirs(os.path.dirname(self.single_results_path), exist_ok=True)
        with open(self.single_results_path, "wb") as f:
            pickle.dump(single_res, f)
        logger.info(f"Single-subject training results are saved.")

        # ── Pooled training ──
        pooled_res = self._train_pooled()
        os.makedirs(os.path.dirname(self.pooled_results_path), exist_ok=True)
        with open(self.pooled_results_path, "wb") as f:
            pickle.dump(pooled_res, f)
        logger.info(f"Pooled training results are saved.")

        return BaseWrapper({"single": single_res,
                            "pooled": pooled_res})

