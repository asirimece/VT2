# tl_trainer.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import pickle

class TLWrapper:
    """
    Container for storing the final predictions and ground truth labels after TL.
    """
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
    """
    Handles transfer learning training for a new subject.

    Freezes the shared backbone if requested and trains only the new head (and optionally fine-tunes the trunk).
    Applies L2 weight decay globally via the optimizer.
    """
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
        """
        Train the TL model.

        Args:
            train_loader: DataLoader yielding (X_batch, y_batch).
            new_cluster_id: int label of the new head’s cluster/subject id.
        """
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
        """
        Evaluate the TL model on held-out data.

        Args:
            test_loader: DataLoader yielding (X_batch, y_batch).
            new_cluster_id: int label of the head to use for prediction.

        Returns:
            A TLWrapper containing numpy arrays of ground truth and predictions.
        """
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
        )