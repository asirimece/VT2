# tl_dataset.py
import torch
from torch.utils.data import Dataset

class TLSubjectDataset(Dataset):
    """
    Dataset for Transfer Learning on a new subject.
    Expects:
      - X: (n_subepochs, n_channels, n_times)
      - y: (n_subepochs,)
    """
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
