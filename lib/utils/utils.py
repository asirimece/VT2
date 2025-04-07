from omegaconf import OmegaConf
import pickle
from torch.utils.data import ConcatDataset, DataLoader
from lib.dataset.dataset import EEGDataset

def load_preprocessed_data(path: str):
    """
    Load preprocessed_data.pkl which should be a dict:
      { subject_id: { '0train': mne.Epochs, '1test': mne.Epochs } }
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def create_subject_dataset(preprocessed_data: dict, subject_id: str, session_key: str = '0train', transform=None):
    """
    Create an EEGDataset for one subject.
    """
    sessions = preprocessed_data[subject_id]
    if session_key not in sessions:
        raise ValueError(f"Subject {subject_id} has no session '{session_key}'")
    epochs = sessions[session_key]
    return EEGDataset(epochs, transform=transform, subject_id=subject_id, session_key=session_key)

def create_pooled_dataset(preprocessed_data: dict, session_key: str = '0train', transform=None):
    """
    Create a pooled dataset from all subjects for the given session.
    """
    datasets = []
    for subject_id in sorted(preprocessed_data.keys(), key=lambda x: int(x)):
        sessions = preprocessed_data[subject_id]
        if session_key in sessions:
            ds = EEGDataset(sessions[session_key], transform=transform, subject_id=subject_id, session_key=session_key)
            datasets.append(ds)
        else:
            print(f"Warning: Subject {subject_id} missing session '{session_key}', skipping.")
    if not datasets:
        raise ValueError("No valid datasets found for pooling.")
    return ConcatDataset(datasets)

def create_data_loader(dataset, batch_size=32, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def to_float(x, default):
    """
    Convert x to float if possible using OmegaConf, otherwise return default.
    """
    try:
        return float(OmegaConf.to_container(x, resolve=True))
    except Exception:
        return default