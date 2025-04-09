import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from lib.logging import logger

logger = logger.get()

class EEGDataset(Dataset):
    """
    Wraps EEG data, labels, and trial IDs for sub-epochs.
    
    If epochs_or_data has a get_data() method (i.e. an MNE Epochs object), it will be used to extract 
    the data along with the events. Otherwise, it assumes that pre-computed NumPy arrays are passed in.
    
    Parameters:
        epochs_or_data: Either an MNE Epochs object or a NumPy array.
        labels: (Optional) NumPy array for labels. If provided, then data is assumed precomputed.
        trial_ids: (Optional) NumPy array for trial identifiers.
        transform: (Optional) A transformation to apply.
        subject_id: (Optional) Subject identifier.
        session_key: (Optional) Session identifier.
    """
    def __init__(self, epochs_or_data, labels=None, trial_ids=None, transform=None, subject_id=None, session_key=None):
        self.transform = transform
        
        # Check if the input has get_data() (i.e. an MNE Epochs object)
        if hasattr(epochs_or_data, "get_data"):
            data = epochs_or_data.get_data()  # shape: (n_subepochs, n_channels, n_samples)
            self.labels = epochs_or_data.events[:, -1]
            self.trial_ids = epochs_or_data.events[:, 1]
        else:
            # Else assume pre-computed arrays are passed in.
            data = epochs_or_data
            self.labels = labels
            self.trial_ids = trial_ids
        
        self.X = torch.tensor(data, dtype=torch.float32)
        self.y = torch.tensor(self.labels, dtype=torch.long)
        self.trial_ids = torch.tensor(self.trial_ids, dtype=torch.long)
        
        logger.info(f"EEGDataset for subject {subject_id} session {session_key}: X shape {self.X.shape}, y shape {self.y.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = (self.X[idx], self.y[idx], self.trial_ids[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample
