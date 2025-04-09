# lib/model/deep4net.py

import torch
from braindecode.models.deep4 import Deep4Net
from braindecode import EEGClassifier

class Deep4NetModel:
    """
    An object-oriented wrapper for building a Deep4Net model using braindecode.
    It takes a Hydra configuration object (or a dict) and extracts the parameters.
    """
    def __init__(self, config):
        """
        Initializes the Deep4NetModel and builds the model.
        
        Parameters
        ----------
        config : OmegaConf object or dict
            Configuration dictionary with the model parameters.
        """
        self.config = config
        self.model = self._build_model()
    
    def _build_model(self):
        # Use dictionary access if config is a dict; otherwise, use attribute access.
        if isinstance(self.config, dict):
            in_chans = self.config.get("in_chans", 22)
            n_classes = self.config.get("n_classes", 4)
            n_times = self.config.get("n_times", 1000)
            final_conv_length = self.config.get("final_conv_length", "auto")
        else:
            in_chans = self.config.in_chans
            n_classes = self.config.n_classes
            n_times = self.config.n_times
            final_conv_length = self.config.final_conv_length
        
        model = Deep4Net(
            in_chans,
            n_classes,
            n_times,
            final_conv_length=final_conv_length
        )
        print("Deep4Net model constructed with the following parameters:")
        print(f"  in_chans: {in_chans}")
        print(f"  n_classes: {n_classes}")
        print(f"  n_times: {n_times}")
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

def build_deep4net_model(config):
    Build a Deep4Net model using hyperparameters from the configuration.
    
    Parameters
    ----------
    config : OmegaConf object or dict
        Configuration dictionary with a 'model' section.
    
    Returns
    -------
    model : torch.nn.Module
        An instance of Deep4Net.
        
    # Use dictionary access if config is a dict; otherwise, use attribute access.
    if isinstance(config, dict):
        in_chans = config.get("in_chans", 22)
        n_classes = config.get("n_classes", 4)
        n_times = config.get("n_times", 1000)
        final_conv_length = config.get("final_conv_length", "auto")
    else:
        in_chans = config.in_chans
        n_classes = config.n_classes
        n_times = config.n_times
        final_conv_length = config.final_conv_length

    model = Deep4Net(
        in_chans,
        n_classes,
        n_times,
        final_conv_length=final_conv_length
    )

    print("Deep4Net model constructed with the following parameters:")
    print(f"  in_chans: {in_chans}")
    print(f"  n_classes: {n_classes}")
    print(f"  n_times: {n_times}")
    print(f"  final_conv_length: {final_conv_length}")
    
    return model


"""