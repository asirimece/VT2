import torch
import torch.nn as nn
from braindecode.models import Deep4Net



class DeepMTLHead(nn.Module):
    def __init__(self, feature_dim, n_outputs, hidden_dim=None, drop_prob=0.2):
        super().__init__()
        hidden_dim = hidden_dim or feature_dim  # or 2*feature_dim for more capacity
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, n_outputs)
        )
    def forward(self, x):
        return self.net(x)


import torch
import torch.nn as nn
from braindecode.models import Deep4Net
from lib.mtl.deep_head import DeepMTLHead  # Path to the head you just added

class MultiTaskDeep4Net(nn.Module):
    def __init__(self, n_chans, n_outputs, n_clusters, backbone_kwargs=None, head_kwargs=None):
        super().__init__()
        backbone_kwargs = backbone_kwargs.copy() if backbone_kwargs is not None else {}
        window_samples = backbone_kwargs.pop("n_times", 500)
        self.shared_backbone = Deep4Net(
            n_chans=n_chans,
            n_outputs=n_outputs,  # Still needed by Deep4Net, but ignored for custom heads
            n_times=window_samples,
            final_conv_length="auto",
            **backbone_kwargs
        )
        # Infer feature dimension for head input
        dummy_input = torch.zeros(1, n_chans, window_samples)
        with torch.no_grad():
            dummy_features = self.shared_backbone(dummy_input)
        feature_dim = dummy_features.shape[1]

        head_kwargs = head_kwargs or {}
        hidden_dim = head_kwargs.get("hidden_dim", feature_dim)
        drop_prob  = head_kwargs.get("drop_prob", 0.2)

        # Use nonlinear head for each cluster
        self.heads = nn.ModuleDict({
            str(cluster): DeepMTLHead(
                feature_dim=feature_dim,
                n_outputs=n_outputs,
                hidden_dim=hidden_dim,
                drop_prob=drop_prob
            )
            for cluster in range(n_clusters)
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
