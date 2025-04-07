#!/usr/bin/env python
import torch
from lib.tl.model import TLModel

def inspect_mtl_model(weights_path="pretrained_mtl_model_weights.pth", 
                      n_chans=22, n_outputs=4, window_samples=500):
    """
    Loads the re-saved state dict of the MTL model and inspects its keys,
    particularly those corresponding to cluster-specific heads, to determine
    the number of pretrained clusters (n_clusters_pretrained).
    """
    # Instantiate TLModel with a dummy value for n_clusters_pretrained.
    # This dummy value doesn't affect the state dict keys.
    dummy_clusters = 1  
    model = TLModel(n_chans=n_chans, n_outputs=n_outputs, 
                    n_clusters_pretrained=dummy_clusters, window_samples=window_samples)
    
    # Load the saved weights.
    state_dict = torch.load(weights_path, map_location="cpu")
    
    # Print all keys in the state dict.
    print("Full state dict keys:")
    for key in state_dict.keys():
        print(key)
    
    # Filter keys corresponding to the heads.
    head_keys = [k for k in state_dict.keys() if k.startswith("mtl_net.heads.")]
    print("\nHead keys found in state dict:")
    for key in head_keys:
        print(key)
    
    # Extract unique head indices.
    head_indices = set()
    for key in head_keys:
        # Expected key pattern: "mtl_net.heads.<index>.<parameter>"
        parts = key.split(".")
        if len(parts) >= 3:
            try:
                head_index = int(parts[2])
                head_indices.add(head_index)
            except ValueError:
                continue
    
    n_clusters_pretrained = len(head_indices)
    print("\nDetermined n_clusters_pretrained:", n_clusters_pretrained)
    return n_clusters_pretrained

if __name__ == "__main__":
    inspect_mtl_model()
