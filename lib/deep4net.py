"""
lib/models/deep4net.py

This module implements the Deep4Net model using braindecode’s implementation.
It loads hyperparameters from a YAML configuration file and builds the model accordingly.
"""

import torch
import torch.nn as nn
from braindecode.models.deep4 import Deep4Net

def build_deep4net_model(cfg):
    """
    Build a Deep4Net model using hyperparameters from the configuration.
    
    Parameters
    ----------
    cfg : OmegaConf object or dict
        Configuration dictionary with a 'model' section.
    
    Returns
    -------
    model : torch.nn.Module
        An instance of Deep4Net.
    """
    # Read hyperparameters from configuration
    in_chans = cfg.model.in_chans
    n_classes = cfg.model.n_classes
    input_window_samples = cfg.model.input_window_samples
    final_conv_length = cfg.model.final_conv_length

    # You can add dropout and other parameters here if your Deep4Net implementation accepts them.
    # For braindecode’s Deep4Net, the primary parameters are as shown.
    model = Deep4Net(
        n_chans=in_chans,
        n_classes=n_classes,
        input_window_samples=input_window_samples,
        final_conv_length=final_conv_length,
    )
    
    # Optionally, you can print a summary or the model architecture.
    print("Deep4Net model constructed with the following parameters:")
    print(f"  in_chans: {in_chans}")
    print(f"  n_classes: {n_classes}")
    print(f"  input_window_samples: {input_window_samples}")
    print(f"  final_conv_length: {final_conv_length}")
    
    return model

"""
if __name__ == "__main__":
    # Example usage: load config and build the model.
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("vt2/config/model/deep4net.yaml")
    model = build_deep4net_model(cfg)
    print(model)
"""