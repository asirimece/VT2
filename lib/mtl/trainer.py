# multitask_trainer.py

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from lib.mtl.model import MultiTaskDeep4Net

class EEGMultiTaskDataset(Dataset):
    def __init__(self, data, labels, subject_ids, cluster_wrapper):
        """
        Dataset for multi-task EEG classification.

        Parameters:
            data (np.array): EEG samples, shape [N, channels, time].
            labels (np.array): Class labels for each sample, shape [N].
            subject_ids (list): Subject identifier for each sample.
            cluster_wrapper (ClusterWrapper): An instance that provides get_cluster_for_subject(subject_id).
        """
        self.data = data
        self.labels = labels
        self.subject_ids = subject_ids
        self.cluster_wrapper = cluster_wrapper

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        subject_id = self.subject_ids[index]
        cluster_id = self.cluster_wrapper.get_cluster_for_subject(subject_id)
        return sample, label, cluster_id

def train_mtl_model(model, dataloader, criterion, optimizer, device, num_epochs=100):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        total_correct = 0
        total_samples = 0
        for data, labels, cluster_ids in dataloader:
            data = data.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            if not torch.is_tensor(cluster_ids):
                cluster_ids = torch.tensor(cluster_ids, dtype=torch.long)
            cluster_ids = cluster_ids.to(device)
            
            optimizer.zero_grad()
            outputs = model(data, cluster_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = data.size(0)
            epoch_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size
        avg_loss = epoch_loss / total_samples
        accuracy = total_correct / total_samples
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
    return model

def evaluate_mtl_model(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, labels, cluster_ids in dataloader:
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
    accuracy = total_correct / total_samples
    print(f"Evaluation Accuracy: {accuracy:.4f}")
    return all_labels, all_preds
