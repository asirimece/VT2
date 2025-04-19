import torch
import torch.nn as nn
from lib.mtl.model import MultiTaskDeep4Net  # Adjust the import path as needed

class TLModel(nn.Module):
    """
    Wraps the MTL MultiTaskDeep4Net and allows adding a new head for a new subject.
    """
    def __init__(self, n_chans, n_outputs, n_clusters_pretrained, window_samples):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_clusters_pretrained = n_clusters_pretrained
        self.window_samples = window_samples  # Save the original input window length

        # Use "n_times" in the backbone kwargs as per the updated API.
        self.mtl_net = MultiTaskDeep4Net(
            n_chans=self.n_chans,
            n_outputs=self.n_outputs,
            n_clusters=self.n_clusters_pretrained,
            backbone_kwargs={"n_times": window_samples},
            head_kwargs=None
        )
        print(f"[DEBUG] MultiTaskDeep4Net created with n_times={window_samples}")

    def forward(self, x, cluster_id):
        return self.mtl_net(x, cluster_id)

    def add_new_head(self, new_cluster_id, feature_dim=None, dummy_input=None):
        """
        Create and add a new linear head for the new subject.
        If feature_dim is none, create dummy_input for scratch baseline.
        """
        if feature_dim is None:
            # if user didn’t give a dummy, build one from n_chans & window_samples
            if dummy_input is None:
                import torch
                device = next(self.parameters()).device
                dummy_input = torch.zeros(
                    1, self.n_chans, self.window_samples, device=device
                )
            # run it through the shared backbone
            with torch.no_grad():
                dummy_feat = self.mtl_net.shared_backbone(dummy_input)
                feature_dim = dummy_feat.shape[1]
            print(f"[DEBUG] Inferred feature dimension: {feature_dim}")
        else:
            print(f"[DEBUG] Using provided feature dimension: {feature_dim}")
        new_head = nn.Linear(feature_dim, self.n_outputs)
        # Store the new head using the numeric cluster id converted to a string (as expected by the backend)
        self.mtl_net.heads[str(new_cluster_id)] = new_head
        print(f"[DEBUG] New head added for cluster id: {new_cluster_id}")

    @property
    def shared_backbone(self):
        # Convenience property for accessing the trunk (shared backbone).
        return self.mtl_net.shared_backbone

    @property
    def heads(self):
        # Access to the head dictionary.
        return self.mtl_net.heads
