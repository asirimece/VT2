
import torch
import torch.nn as nn
from lib.mtl.model import MultiTaskDeep4Net
from lib.logging import logger

logger = logger.get()


class TLModel(nn.Module):
    def __init__(self, n_chans, n_outputs, n_clusters_pretrained, window_samples):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_clusters_pretrained = n_clusters_pretrained
        self.window_samples = window_samples

        # Initialize the pretrained MTL backbone.
        self.mtl_net = MultiTaskDeep4Net(
            n_chans=self.n_chans,
            n_outputs=self.n_outputs,
            n_clusters=self.n_clusters_pretrained,
            backbone_kwargs={"n_times": window_samples},
            head_kwargs=None
        )
        
    def forward(self, x, subject_ids):
        # Convert subject_ids to integers if they are strings
        if not torch.is_tensor(subject_ids):
            subject_ids = [int(s) if isinstance(s, str) else s for s in subject_ids]
            subject_ids = torch.tensor(subject_ids, dtype=torch.long, device=x.device)
        else:
            subject_ids = subject_ids.to(x.device)

        features = self.mtl_net.shared_backbone(x)
        unique = subject_ids.unique()
        if unique.numel() == 1:
            sid = int(unique[0])
            head_key = f"subj_{sid}"
            head = self.mtl_net.heads[head_key]
            return head(features)
        outputs = []
        for i, sid in enumerate(subject_ids.tolist()):
            head_key = f"subj_{sid}"
            head = self.mtl_net.heads[head_key]
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
        # Create subject-specific head
        new_head = nn.Linear(feature_dim, self.n_outputs)
        head_key = f"subj_{subject_id}"
        self.mtl_net.heads[head_key] = new_head

    @property
    def shared_backbone(self):
        return self.mtl_net.shared_backbone

    @property
    def heads(self):
        return self.mtl_net.heads