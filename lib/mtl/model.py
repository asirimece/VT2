# multitask_model.py

import torch
import torch.nn as nn
from braindecode.models import Deep4Net

class MultiTaskDeep4Net(nn.Module):
    def __init__(self, n_chans, n_outputs, n_clusters, backbone_kwargs=None, head_kwargs=None):
        """
        Multi-task Deep4Net with a shared backbone and separate classification heads.

        Parameters:
            n_chans (int): Number of EEG channels.
            n_outputs (int): Number of output classes.
            n_clusters (int): Number of clusters/tasks.
            backbone_kwargs (dict): Additional parameters for Deep4Net.
            head_kwargs (dict): Additional parameters for the classifier head.
        """
        super(MultiTaskDeep4Net, self).__init__()
        # Make a copy of backbone_kwargs so we can modify it.
        backbone_kwargs = backbone_kwargs.copy() if backbone_kwargs is not None else {}
        # Pop n_times so it isn't passed twice.
        window_samples = backbone_kwargs.pop("n_times", 500)
        self.shared_backbone = Deep4Net(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=window_samples,
            final_conv_length="auto",
            **backbone_kwargs
        )
        # Compute the actual feature dimension by performing a dummy forward pass.
        dummy_input = torch.zeros(1, n_chans, window_samples)
        with torch.no_grad():
            dummy_features = self.shared_backbone(dummy_input)
        feature_dim = dummy_features.shape[1]
        print(f"Computed backbone feature dimension: {feature_dim}")
        
        # Create a separate classification head for each cluster.
        self.heads = nn.ModuleDict({
            str(cluster): nn.Linear(feature_dim, n_outputs, **(head_kwargs or {}))
            for cluster in range(n_clusters)
        })

    def forward(self, x, cluster_ids):
        """
        Forward pass.

        Parameters:
            x (Tensor): Input EEG data of shape [batch, channels, time].
            cluster_ids (list or Tensor): Cluster label for each sample in the batch.

        Returns:
            Tensor: Logits for each sample, shape [batch, n_outputs].
        """
        features = self.shared_backbone(x)  # [batch, feature_dim]
        if isinstance(cluster_ids, int) or (torch.is_tensor(cluster_ids) and cluster_ids.unique().numel() == 1):
            head = self.heads[str(int(cluster_ids if isinstance(cluster_ids, int) else cluster_ids[0]))]
            output = head(features)
        else:
            outputs = []
            for i, cid in enumerate(cluster_ids):
                head = self.heads[str(int(cid))]
                outputs.append(head(features[i].unsqueeze(0)))
            output = torch.cat(outputs, dim=0)
        return output
