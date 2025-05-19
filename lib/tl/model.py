import torch
import torch.nn as nn
from lib.mtl.model import MultiTaskDeep4Net  # assumes your updated MTL with deep heads

class TLModel(nn.Module):
    def __init__(self, n_chans, n_outputs, n_clusters_pretrained, window_samples, 
                 head_type="linear", head_kwargs=None, backbone_kwargs=None):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_clusters_pretrained = n_clusters_pretrained
        self.window_samples = window_samples

        # Initialize the MTL backbone
        self.mtl_net = MultiTaskDeep4Net(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_clusters=n_clusters_pretrained,
            backbone_kwargs=backbone_kwargs or {"n_times": window_samples},
            head_kwargs=None  # heads will not be used in TL, only backbone weights loaded
        )

        self.heads = nn.ModuleDict()  # subject-specific heads (added below)
        self.head_type = head_type
        self.head_kwargs = head_kwargs or {}

    def forward(self, x, subject_ids):
        if not torch.is_tensor(subject_ids):
            subject_ids = torch.tensor(subject_ids, dtype=torch.long, device=x.device)
        else:
            subject_ids = subject_ids.to(x.device)
        features = self.mtl_net.shared_backbone(x)
        outputs = torch.zeros(features.shape[0], self.n_outputs, device=x.device)
        for sid in torch.unique(subject_ids):
            idxs = (subject_ids == sid).nonzero(as_tuple=False).flatten()
            if len(idxs) == 0:
                continue
            feats = features[idxs]
            head = self.heads[f"subj_{int(sid)}"]
            outputs[idxs] = head(feats)
        return outputs

    def add_new_head(self, subject_id, feature_dim=None, dummy_input=None):
        # Dynamically infer feature dim from backbone if not given
        if feature_dim is None:
            if dummy_input is None:
                device = next(self.parameters()).device
                dummy_input = torch.zeros(2, self.n_chans, self.window_samples, device=device)
            with torch.no_grad():
                dummy_feat = self.mtl_net.shared_backbone(dummy_input)
                feature_dim = dummy_feat.shape[1]
        # Create subject-specific head
        if self.head_type == "linear":
            head = nn.Linear(feature_dim, self.n_outputs)
        else:
            # Optionally, allow deep/nonlinear head for TL too
            hidden_dim = self.head_kwargs.get("hidden_dim", feature_dim)
            drop_prob = self.head_kwargs.get("drop_prob", 0.2)
            norm_type = self.head_kwargs.get("norm_type", "layer")
            norm_layer = nn.BatchNorm1d if norm_type == "batch" else nn.LayerNorm
            head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                norm_layer(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, self.n_outputs)
            )
        self.heads[f"subj_{subject_id}"] = head

    @property
    def shared_backbone(self):
        return self.mtl_net.shared_backbone
