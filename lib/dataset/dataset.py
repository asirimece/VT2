import torch
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    """
    Wraps X, y, trial_ids for sub-epochs.
    X: shape (n_subepochs, n_channels, n_samples)
    y: shape (n_subepochs,)
    trial_ids: shape (n_subepochs,) points to the original trial index.
    """
    def __init__(self, X, y, trial_ids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.trial_ids = torch.tensor(trial_ids, dtype=torch.long)
        print(f"EEGDataset shapes => X: {self.X.shape}, y: {self.y.shape}, trial_ids: {self.trial_ids.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.trial_ids[idx]
