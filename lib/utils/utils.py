from omegaconf import OmegaConf
import pickle

def to_float(x, default):
    """
    Convert x to float if possible using OmegaConf, otherwise return default.
    """
    try:
        return float(OmegaConf.to_container(x, resolve=True))
    except Exception:
        return default