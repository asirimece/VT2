from itertools import product
import types
import os
import pickle
import torch
import numpy as np
from tqdm.auto import tqdm
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from omegaconf import DictConfig
from collections import defaultdict

from lib.dataset.dataset import TLSubjectDataset
from lib.tl.model import TLModel
from lib.utils.utils import _prefix_mtl_keys
from lib.logging import logger

logger = logger.get()


def freeze_backbone_layers(backbone, freeze_until_layer=None):
    found = False
    for name, module in backbone.named_children():
        for param in module.parameters():
            param.requires_grad = False
        if freeze_until_layer and name == freeze_until_layer:
            found = True
            break
    if freeze_until_layer and not found:
        raise ValueError(f"Layer {freeze_until_layer} not found.")


class TLWrapper:
    def __init__(self, ground_truth, predictions):
        self.ground_truth = ground_truth
        self.predictions = predictions

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)


class TLTrainer:
    def __init__(self, config: DictConfig):
        self.cfg = config.experiment.experiment.transfer
        self.device = torch.device(self.cfg.device)
        self.preprocessed_data_path = config.experiment.experiment.preprocessed_file
        self.criterion = nn.CrossEntropyLoss()
        self._pretrained = None
        if not self.cfg.init_from_scratch and self.cfg.pretrained_mtl_model:
            self._pretrained = _prefix_mtl_keys(
                torch.load(self.cfg.pretrained_mtl_model, map_location=self.device)
            )

    def _set_seed(self, seed):
        import random
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run(self):
        # 1) load splits
        with open(self.preprocessed_data_path, "rb") as f:
            data = pickle.load(f)
        subs = sorted(data.keys())
        all_results = defaultdict(list)

        strategy = self.cfg.save_strategy.lower()
        best_acc = -float("inf")
        best_state = None
        best_sub = None
        swa_model = swa_sched = None

        # 1) prepare the list of all (run_index, subject_id) tasks
        runs = list(range(self.cfg.n_runs))
        subs = sorted(data.keys())
        tasks = list(product(runs, subs))

        # Single progress bar over all tasks
        pbar = tqdm(
            tasks,
            desc="TL Training",
            total=len(tasks),
            unit="task",
            dynamic_ncols=True
        )
        
        # 2) per-subject TL
        for run_i, sub in pbar:
            self._set_seed(self.cfg.seed_start + run_i)
            
            Xtr, ytr, Xte, yte = self._load_data(data, sub)
            self._build_model(Xtr, sub)
            self.optimizer = self._build_optimizer()

            if strategy == "swa" and swa_model is None:
                swa_model = AveragedModel(self.model)
                swa_sched = SWALR(self.optimizer, swa_lr=self.cfg.swa_lr)

            tr_ld, te_ld = self._build_dataloaders(Xtr, ytr, Xte, yte)
            self._train(tr_ld, sub,
                        strategy=strategy,
                        swa_model=swa_model,
                        swa_sched=swa_sched,
                        swa_start=self.cfg.swa_start)

            wrap = self._evaluate(te_ld, sub)
            all_results[sub].append(wrap)

            if strategy == "best_run":
                acc = (wrap.predictions == wrap.ground_truth).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_state = self.model.state_dict().copy()
                    best_sub = sub
            # update postfix with useful info
            pbar.set_postfix({
                "run":      f"{run_i+1}/{self.cfg.n_runs}",
                "subject":  sub,
                "last_acc": f"{(wrap.predictions == wrap.ground_truth).mean():.3f}"
            })

        # 3) prepare save
        os.makedirs(self.cfg.tl_model_output, exist_ok=True)
        outp = os.path.join(self.cfg.tl_model_output, "tl_pooled_model.pth")

        # SWA
        if strategy == "swa":
            first_sub = subs[0]
            orig_forward = swa_model.module.forward

            def swa_forward_single(self, x):
                b = x.size(0)
                return orig_forward(x, [first_sub] * b)

            swa_model.module.forward = types.MethodType(swa_forward_single,
                                                       swa_model.module)
            update_bn(tr_ld, swa_model)
            torch.save(swa_model.module.state_dict(), outp)
            logger.info(f"[TLTrainer] Saved SWA model → {outp}")

        # Best-run
        elif strategy == "best_run":
            if best_state is None or best_sub is None:
                raise RuntimeError("No best-run model found")
            new_state = {}
            for k, v in best_state.items():
                if not k.startswith("heads."):
                    new_state[k] = v
            prefix_old = f"heads.{best_sub}."
            for k, v in best_state.items():
                if k.startswith(prefix_old):
                    new_key = k.replace(prefix_old, "heads.0.")
                    new_state[new_key] = v
            torch.save(new_state, outp)
            logger.info(f"[TLTrainer] Saved Best-Run model acc={best_acc:.3f}"
                        f" (subject={best_sub}) → {outp}")

        # Universal: seed from pretrained head, fine-tune on pooled TL data with early stopping
        elif strategy == "universal":
            m = self.model
            # 1) add & seed head 0
            m.add_new_head(0)
            if self._pretrained is not None:
                cur = m.state_dict()
                for k, v in self._pretrained.items():
                    if k.startswith("heads.0."):
                        cur[k] = v.clone()
                m.load_state_dict(cur)
            # 2) optional freeze
            if self.cfg.freeze_backbone or self.cfg.freeze_until_layer:
                freeze_backbone_layers(m.shared_backbone,
                                       self.cfg.freeze_until_layer)
            # 3) pooled TL data & split
            full = ConcatDataset([
                TLSubjectDataset(*self._load_data(data, sub)[:2])
                for sub in subs
            ])
            n = len(full)
            split = int(0.1 * n)
            idx = np.random.permutation(n)
            train_idx, val_idx = idx[split:], idx[:split]
            train_loader = DataLoader(
                full, batch_size=self.cfg.batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(train_idx))
            val_loader = DataLoader(
                full, batch_size=self.cfg.batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(val_idx))

            # 4) class weights & optimizer
            all_labels = np.concatenate([
                self._load_data(data, sub)[1] for sub in subs
            ])
            counts = np.bincount(all_labels, minlength=self.cfg.model.n_outputs)
            weights = torch.tensor(counts.sum() / counts,
                                   dtype=torch.float32,
                                   device=self.device)
            criterion = nn.CrossEntropyLoss(weight=weights)
            optimizer = torch.optim.Adam(
                m.parameters(),
                lr=self.cfg.head_lr,
                weight_decay=self.cfg.weight_decay
            )

            # 5) fine-tune with early stopping
            best_val = float("inf")
            no_improve = 0
            max_bad = self.cfg.early_stop_patience if hasattr(self.cfg, 'early_stop_patience') else 5
            for epoch in range(1, self.cfg.universal_epochs + 1):
                m.train()
                train_loss = 0.0
                for X, y in train_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    out = m(X, [0] * X.size(0))
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * y.size(0)
                train_loss /= len(train_idx)

                # validation
                m.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X, y in val_loader:
                        X, y = X.to(self.device), y.to(self.device)
                        out = m(X, [0] * X.size(0))
                        val_loss += criterion(out, y).item() * y.size(0)
                val_loss /= len(val_idx)

                logger.info(f"[Universal] Ep{epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

                if val_loss + 1e-4 < best_val:
                    best_val = val_loss
                    no_improve = 0
                    torch.save(m.state_dict(), outp + ".best")
                else:
                    no_improve += 1
                    if no_improve >= max_bad:
                        logger.info(f"[Universal] Early stopping at epoch {epoch}")
                        break

            # 6) load & save best
            best_state = torch.load(outp + ".best", map_location=self.device)
            torch.save(best_state, outp)
            logger.info(f"[TLTrainer] Saved Universal model → {outp}")

        else:
            raise ValueError(f"Unknown save_strategy={strategy!r}")

        return all_results

    def _train(self, loader, subject_id,
               strategy="best_run",
               swa_model=None, swa_sched=None, swa_start=1):
        """Exactly your original loop, plus SWA updates if requested."""
        self.model.train()
        for epoch in range(1, self.cfg.epochs + 1):
            tot, corr, cnt = 0.0, 0, 0
            pbar = tqdm(loader, desc=f"Epoch {epoch}/{self.cfg.epochs}", leave=False)
            for X, y in pbar:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(X, [subject_id]*X.size(0))
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()
                preds = out.argmax(dim=1)
                corr += (preds == y).sum().item()
                cnt  += y.size(0)
                tot  += loss.item() * y.size(0)
                pbar.set_postfix(loss=tot/cnt, acc=corr/cnt)
            print(f"[TLTrainer] Epoch {epoch}: Loss={tot/cnt:.4f}, Acc={corr/cnt:.4f}")

            # SWA hook
            if strategy == "swa" and epoch >= swa_start:
                swa_model.update_parameters(self.model)
                swa_sched.step()

    def _load_data(self, data, sub):
        tr, te = data[sub]["train"], data[sub]["test"]
        return tr.get_data(), tr.events[:, -1], te.get_data(), te.events[:, -1]

    def _build_model(self, X, sub):
        n_ch, win = X.shape[1], X.shape[2]
        
        head_kw = {
            "hidden_dim": self.cfg.head_hidden_dim,
            "dropout":    self.cfg.head_dropout
        }
        
        m = TLModel(
            n_chans=n_ch,
            n_outputs=self.cfg.model.n_outputs,
            n_clusters_pretrained=self.cfg.model.n_clusters_pretrained,
            window_samples=win,
            head_kwargs=head_kw
        )
        if self._pretrained:
            m.load_state_dict(self._pretrained)
        if self.cfg.freeze_backbone or self.cfg.freeze_until_layer:
            freeze_backbone_layers(m.shared_backbone, self.cfg.freeze_until_layer)
        m.add_new_head(sub)
        self.model = m.to(self.device)

    def _build_optimizer(self):
        b, h = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad: continue
            (b if "shared_backbone" in n else h).append(p)
        return torch.optim.Adam([
            {"params": b, "lr": self.cfg.backbone_lr, "weight_decay": self.cfg.weight_decay},
            {"params": h, "lr": self.cfg.head_lr,     "weight_decay": 0.0}
        ])

    def _build_dataloaders(self, Xtr, ytr, Xte, yte):
        return (
            DataLoader(TLSubjectDataset(Xtr,ytr), batch_size=self.cfg.batch_size, shuffle=True),
            DataLoader(TLSubjectDataset(Xte,yte), batch_size=self.cfg.batch_size, shuffle=False)
        )

    def _evaluate(self, loader, sub):
        self.model.eval()
        ps, ls = [], []
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                out = self.model(X, [sub]*X.size(0))
                p = out.argmax(dim=1).cpu().numpy()
                ps.extend(p); ls.extend(y.numpy())
        return TLWrapper(ground_truth=np.array(ls), predictions=np.array(ps))

