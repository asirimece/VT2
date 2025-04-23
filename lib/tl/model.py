
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

        # Initialize the pretrained MTL backbone with its cluster heads
        self.mtl_net = MultiTaskDeep4Net(
            n_chans=self.n_chans,
            n_outputs=self.n_outputs,
            n_clusters=self.n_clusters_pretrained,
            backbone_kwargs={"n_times": window_samples},
            head_kwargs=None
        )
        
    def forward(self, x, subject_ids):
        """
        Vectorized forward using subject-specific heads.
        If the batch contains only one subject, applies that subject's head to all features at once.
        Otherwise falls back to per-sample routing.
        """
        # Ensure subject_ids is a tensor on correct device
        if not torch.is_tensor(subject_ids):
            subject_ids = torch.tensor(subject_ids, dtype=torch.long, device=x.device)
        else:
            subject_ids = subject_ids.to(x.device)

        features = self.mtl_net.shared_backbone(x)
        unique = subject_ids.unique()
        # Fast path: single-subject batch
        if unique.numel() == 1:
            sid = int(unique[0])
            head_key = f"subj_{sid}"
            head = self.mtl_net.heads[head_key]
            return head(features)

        # Fallback: mixed-subject batch
        outputs = []
        for i, sid in enumerate(subject_ids.tolist()):
            head_key = f"subj_{sid}"
            head = self.mtl_net.heads[head_key]
            outputs.append(head(features[i : i + 1]))
        return torch.cat(outputs, dim=0)
    
    """    
    NEW
    def forward(self, x, subject_ids):
        # Debug shapes and routing info
        logger.debug(f"TLModel.forward: x.shape={x.shape}, subject_ids={subject_ids}")
        features = self.mtl_net.shared_backbone(x)
        outputs = []
        for i, sid in enumerate(subject_ids):
            head_key = f"subj_{int(sid)}"
            if head_key not in self.mtl_net.heads:
                logger.warning(f"Head not found for {head_key}; available heads: {list(self.mtl_net.heads.keys())}")
            head = self.mtl_net.heads[head_key]
            outputs.append(head(features[i:i+1]))
        output = torch.cat(outputs, dim=0)
        logger.debug(f"TLModel.forward output shape: {output.shape}")
        return output
    """
    """    
    OLD
    def forward(self, x, subject_ids):
        
        #Forward pass using subject-specific heads. `subject_ids` should be a list or tensor
        #of same length as batch size, indicating which subject head to use for each sample.
        
        features = self.mtl_net.shared_backbone(x)
        outputs = []
        # Loop over batch to route each sample through its subject head
        for i, sid in enumerate(subject_ids):
            key = f"subj_{int(sid)}"
            head = self.mtl_net.heads[key]
            # features[i:i+1] has shape [1, feature_dim]
            outputs.append(head(features[i:i+1]))
        return torch.cat(outputs, dim=0)
        """

    def add_new_head(self, subject_id, feature_dim=None, dummy_input=None):
        """
        Create and add a new linear head for a new subject.
        Keys in `self.mtl_net.heads` are now "subj_<subject_id>".
        """
        # Infer feature dimension if not provided
        if feature_dim is None:
            if dummy_input is None:
                # Build a dummy input to pass through the backbone
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
        logger.info(f"New head added for subject id: {subject_id}")

    @property
    def shared_backbone(self):
        return self.mtl_net.shared_backbone

    @property
    def heads(self):
        return self.mtl_net.heads




"""import torch
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

        self.mtl_net = MultiTaskDeep4Net(
            n_chans=self.n_chans,
            n_outputs=self.n_outputs,
            n_clusters=self.n_clusters_pretrained,
            backbone_kwargs={"n_times": window_samples},
            head_kwargs=None
        )

    def forward(self, x, cluster_id):
        return self.mtl_net(x, cluster_id)

    def add_new_head(self, new_cluster_id, feature_dim=None, dummy_input=None):
        #Create and add a new linear head for the new subject.
        # If feature_dim is none, create dummy_input for scratch training.
        if feature_dim is None:
            if dummy_input is None:
                import torch
                device = next(self.parameters()).device
                dummy_input = torch.zeros(
                    1, self.n_chans, self.window_samples, device=device
                )
            with torch.no_grad():
                dummy_feat = self.mtl_net.shared_backbone(dummy_input)
                feature_dim = dummy_feat.shape[1]
        
        new_head = nn.Linear(feature_dim, self.n_outputs)
        self.mtl_net.heads[str(new_cluster_id)] = new_head
        logger.info(f"New head added for cluster id: {new_cluster_id}")   # heads keyed by cluster id + subject id.

    @property
    def shared_backbone(self):
        return self.mtl_net.shared_backbone

    @property
    def heads(self):
        return self.mtl_net.heads
"""