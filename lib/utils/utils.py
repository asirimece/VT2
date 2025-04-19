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
    
def _prefix_mtl_keys(state_dict):
    """
    Prepend 'mtl_net.' to every key that belongs to the shared_backbone or heads,
    so it matches TLModel’s attribute namespace.
    """
    new_sd = {}
    for key, val in state_dict.items():
        if key.startswith("shared_backbone") or key.startswith("heads"):
            new_sd[f"mtl_net.{key}"] = val
        else:
            new_sd[key] = val
    return new_sd
