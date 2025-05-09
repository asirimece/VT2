import torch
import torch.nn as nn
from braindecode.models import Deep4Net

class MultiTaskDeep4Net(nn.Module):
    def __init__(self, n_chans, n_outputs, n_clusters, backbone_kwargs=None, head_kwargs=None):
        """
        Multi-task Deep4Net with a shared backbone and separate deep classification heads per cluster.
        """
        super(MultiTaskDeep4Net, self).__init__()
        backbone_kwargs = backbone_kwargs.copy() if backbone_kwargs is not None else {}
        window_samples = backbone_kwargs.pop("n_times", 500)

        self.shared_backbone = Deep4Net(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=window_samples,
            final_conv_length="auto",
            **backbone_kwargs
        )

        # Get the feature dimension by running a dummy forward pass
        dummy_input = torch.zeros(1, n_chans, window_samples)
        with torch.no_grad():
            dummy_features = self.shared_backbone(dummy_input)
        feature_dim = dummy_features.shape[1]

        # Define deep heads per cluster
        dropout_prob = (head_kwargs or {}).get("dropout", 0.3)
        hidden_dim = (head_kwargs or {}).get("hidden_dim", 128)

        self.heads = nn.ModuleDict({
            str(cluster): nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, n_outputs)
            ) for cluster in range(n_clusters)
        })

    def forward(self, x, cluster_ids):
        features = self.shared_backbone(x)

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
