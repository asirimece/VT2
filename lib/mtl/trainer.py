# multitask_trainer.py

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from lib.mtl.model import MultiTaskDeep4Net
import pickle


class MTLWrapper:
    """
    A container for wrapping MTL training results.
    """
    def __init__(self, results_by_subject, training_logs=None, cluster_assignments=None, additional_info=None):
        # Convert subject keys to strings.
        self.results_by_subject = {str(k): v for k, v in results_by_subject.items()}
        self.training_logs = training_logs if training_logs is not None else {}
        # Convert cluster_assignments keys to strings.
        self.cluster_assignments = {str(k): v for k, v in (cluster_assignments if cluster_assignments is not None else {}).items()}
        self.additional_info = additional_info if additional_info is not None else {}
    
    def get_subject_results(self, subject):
        return self.results_by_subject.get(str(subject))
    
    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f"MTL results saved to {filename}")
    
    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict) and ("ground_truth" in obj and "predictions" in obj):
            wrapped = {"pooled": obj}
            print("[DEBUG] Loaded results as dict with keys ['ground_truth', 'predictions']. Wrapping under key 'pooled'.")
            return cls(results_by_subject=wrapped)
        if isinstance(obj, dict):
            return cls(results_by_subject=obj)
        if isinstance(obj, list):
            wrapped = {"pooled": obj}
            return cls(results_by_subject=wrapped)
        return obj


class EEGMultiTaskDataset(Dataset):
    def __init__(self, data, labels, subject_ids, cluster_wrapper):
        """
        Dataset for multi-task EEG classification.

        Parameters:
            data (np.array): EEG samples, shape [N, channels, time].
            labels (np.array): Class labels for each sample.
            subject_ids (list): Subject identifier for each sample.
            cluster_wrapper (ClusterWrapper): An instance providing get_cluster_for_subject(subject_id).
        """
        self.data = data
        self.labels = labels
        # Convert subject IDs to strings.
        self.subject_ids = [str(sid) for sid in subject_ids]
        self.cluster_wrapper = cluster_wrapper

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        subject_id = self.subject_ids[index]
        cluster_id = self.cluster_wrapper.get_cluster_for_subject(subject_id)
        return sample, label, subject_id, cluster_id


def train_mtl_model(model, dataloader, criterion, optimizer, device, num_epochs=100):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        total_correct = 0
        total_samples = 0
        # Unpack four values now: data, labels, subject_ids, cluster_ids
        for data, labels, subject_ids, cluster_ids in dataloader:
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
    all_subjects = []   # new list to capture subject ids
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


