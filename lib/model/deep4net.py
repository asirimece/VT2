# lib/model/deep4net.py

import torch
from braindecode.models.deep4 import Deep4Net
from braindecode import EEGClassifier

class Deep4NetModel:
    """
    An object-oriented wrapper for building a Deep4Net model using braindecode.
    It takes a Hydra configuration object (or a dict) and extracts the parameters.
    """
    def __init__(self, cfg):
        """
        Initializes the Deep4NetModel and builds the model.
        
        Parameters
        ----------
        cfg : OmegaConf object or dict
            Configuration dictionary with the model parameters.
        """
        self.cfg = cfg
        self.model = self._build_model()
    
    def _build_model(self):
        # Use dictionary access if cfg is a dict; otherwise, use attribute access.
        if isinstance(self.cfg, dict):
            in_chans = self.cfg.get("in_chans", 22)
            n_classes = self.cfg.get("n_classes", 4)
            input_window_samples = self.cfg.get("input_window_samples", 1000)
            final_conv_length = self.cfg.get("final_conv_length", "auto")
        else:
            in_chans = self.cfg.in_chans
            n_classes = self.cfg.n_classes
            input_window_samples = self.cfg.input_window_samples
            final_conv_length = self.cfg.final_conv_length
        
        model = Deep4Net(
            in_chans,
            n_classes,
            input_window_samples,
            final_conv_length=final_conv_length
        )
        print("Deep4Net model constructed with the following parameters:")
        print(f"  in_chans: {in_chans}")
        print(f"  n_classes: {n_classes}")
        print(f"  input_window_samples: {input_window_samples}")
        print(f"  final_conv_length: {final_conv_length}")
        return model
    
    def get_model(self):
        """
        Returns the built Deep4Net model.
        """
        return self.model



"""
lib/model/deep4net.py

This module implements the Deep4Net model using braindecode’s implementation.
It loads hyperparameters from a YAML configuration file (or dict) and builds the model accordingly.

import torch
from braindecode.models.deep4 import Deep4Net
from braindecode import EEGClassifier

def build_deep4net_model(cfg):
    Build a Deep4Net model using hyperparameters from the configuration.
    
    Parameters
    ----------
    cfg : OmegaConf object or dict
        Configuration dictionary with a 'model' section.
    
    Returns
    -------
    model : torch.nn.Module
        An instance of Deep4Net.
        
    # Use dictionary access if cfg is a dict; otherwise, use attribute access.
    if isinstance(cfg, dict):
        in_chans = cfg.get("in_chans", 22)
        n_classes = cfg.get("n_classes", 4)
        input_window_samples = cfg.get("input_window_samples", 1000)
        final_conv_length = cfg.get("final_conv_length", "auto")
    else:
        in_chans = cfg.in_chans
        n_classes = cfg.n_classes
        input_window_samples = cfg.input_window_samples
        final_conv_length = cfg.final_conv_length

    model = Deep4Net(
        in_chans,
        n_classes,
        input_window_samples,
        final_conv_length=final_conv_length
    )

    print("Deep4Net model constructed with the following parameters:")
    print(f"  in_chans: {in_chans}")
    print(f"  n_classes: {n_classes}")
    print(f"  input_window_samples: {input_window_samples}")
    print(f"  final_conv_length: {final_conv_length}")
    
    return model


"""