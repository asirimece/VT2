import torch
import torch.nn as nn
from braindecode.models import Deep4Net

class DeepMTLHead(nn.Module):
    def __init__(self, feature_dim, n_outputs, hidden_dim=None, drop_prob=0.2, norm_type="layer"):
        super().__init__()
        hidden_dim = hidden_dim or feature_dim
        norm_layer = nn.BatchNorm1d if norm_type == "batch" else nn.LayerNorm
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, n_outputs)
        )
    def forward(self, x):
        return self.net(x)

class MultiTaskDeep4Net(nn.Module):
    def __init__(self, n_chans, n_outputs, n_clusters, backbone_kwargs=None, head_kwargs=None):
        super().__init__()
        backbone_kwargs = backbone_kwargs.copy() if backbone_kwargs is not None else {}
        window_samples = backbone_kwargs.pop("n_times", 500)
        self.shared_backbone = Deep4Net(
            n_chans=n_chans,
            n_outputs=n_outputs,  # needed for Deep4Net
            n_times=window_samples,
            final_conv_length="auto",
            **backbone_kwargs
        )
        # Get feature dim
        dummy_input = torch.zeros(2, n_chans, window_samples)  # must be at least 2 for BN
        with torch.no_grad():
            dummy_features = self.shared_backbone(dummy_input)
        feature_dim = dummy_features.shape[1]

        head_kwargs = head_kwargs or {}
        hidden_dim = head_kwargs.get("hidden_dim", feature_dim)
        drop_prob = head_kwargs.get("drop_prob", 0.2)
        norm_type = head_kwargs.get("norm_type", "batch")

        # Make heads per cluster
        self.heads = nn.ModuleDict({
            str(cluster): DeepMTLHead(
                feature_dim=feature_dim,
                n_outputs=n_outputs,
                hidden_dim=hidden_dim,
                drop_prob=drop_prob,
                norm_type=norm_type
            )
            for cluster in range(n_clusters)
        })

    def forward(self, x, cluster_ids):
        features = self.shared_backbone(x)  # [batch, feat_dim]
        # Convert cluster_ids to tensor if not already
        if not torch.is_tensor(cluster_ids):
            cluster_ids = torch.tensor(cluster_ids, dtype=torch.long, device=x.device)
        else:
            cluster_ids = cluster_ids.to(x.device)
        # Batch mode: forward samples grouped by cluster
        outputs = torch.zeros(features.shape[0], self.heads["0"].net[-1].out_features, device=x.device)
        for cluster in torch.unique(cluster_ids):
            idxs = (cluster_ids == cluster).nonzero(as_tuple=False).flatten()
            if len(idxs) == 0:
                continue
            feats = features[idxs]
            head = self.heads[str(int(cluster))]
            outputs[idxs] = head(feats)
        return outputs

