import torch
import torch.nn as nn
from lib.mtl.model import MultiTaskDeep4Net
from lib.logging import logger

logger = logger.get()


class TLModel(nn.Module):
    def __init__(self, n_chans, n_outputs, n_clusters_pretrained, window_samples, head_kwargs=None):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_clusters_pretrained = n_clusters_pretrained
        self.window_samples = window_samples

        self.head_kwargs = head_kwargs or {}
        self.hidden_dim = self.head_kwargs.get("hidden_dim", 128)
        self.dropout = self.head_kwargs.get("dropout", 0.5)

        # Initialize the pretrained MTL backbone
        self.mtl_net = MultiTaskDeep4Net(
            n_chans=self.n_chans,
            n_outputs=self.n_outputs,
            n_clusters=self.n_clusters_pretrained,
            backbone_kwargs={"n_times": window_samples},
            head_kwargs={
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout
            }
        )

    def forward(self, x, subject_ids):
        if not torch.is_tensor(subject_ids):
            subject_ids = [int(s) if isinstance(s, str) else s for s in subject_ids]
            subject_ids = torch.tensor(subject_ids, dtype=torch.long, device=x.device)
        else:
            subject_ids = subject_ids.to(x.device)

        features = self.mtl_net.shared_backbone(x)
        unique = subject_ids.unique()
        if unique.numel() == 1:
            sid = int(unique[0])
            head = self.mtl_net.heads[f"subj_{sid}"]
            return head(features)
        outputs = []
        for i, sid in enumerate(subject_ids.tolist()):
            head = self.mtl_net.heads[f"subj_{sid}"]
            outputs.append(head(features[i : i + 1]))
        return torch.cat(outputs, dim=0)

    def add_new_head(self, subject_id, feature_dim=None, dummy_input=None):
        if feature_dim is None:
            if dummy_input is None:
                device = next(self.parameters()).device
                dummy_input = torch.zeros(
                    1, self.n_chans, self.window_samples, device=device
                )
            with torch.no_grad():
                dummy_feat = self.mtl_net.shared_backbone(dummy_input)
                feature_dim = dummy_feat.shape[1]

        # MLP head
        mlp_head = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.n_outputs)
        )
        self.mtl_net.heads[f"subj_{subject_id}"] = mlp_head

    @property
    def shared_backbone(self):
        return self.mtl_net.shared_backbone

    @property
    def heads(self):
        return self.mtl_net.heads
