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
    Handles the TL training loop for a new subject:
      - Optionally freezes the shared backbone.
      - Trains the new head (and optionally fine-tunes the trunk).
      - Evaluates the model on test data.
    """
    def __init__(self, model, device, freeze_backbone, lr, epochs):
        self.model = model
        self.device = device
        self.freeze_backbone = freeze_backbone
        self.lr = lr
        self.epochs = epochs

    def freeze_trunk(self):
        for param in self.model.shared_backbone.parameters():
            param.requires_grad = False

    def train(self, train_loader, new_cluster_id):
        if self.freeze_backbone:
            self.freeze_trunk()

        # Only train parameters with requires_grad=True
        params_to_train = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(params_to_train, lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            total_samples = 0
            correct = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch, [new_cluster_id] * len(X_batch))
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total_samples += len(X_batch)
                total_loss += loss.item() * len(X_batch)

            epoch_loss = total_loss / total_samples
            epoch_acc = correct / total_samples
            print(f"[TLTrainer] Epoch {epoch+1}/{self.epochs}, Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")

    def evaluate(self, test_loader, new_cluster_id):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch, [new_cluster_id] * len(X_batch))
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())

        return TLWrapper(np.array(all_labels), np.array(all_preds))
