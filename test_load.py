# test_load.py

import torch
import joblib

# 1) Whitelist any custom or built‐in classes your TL pipeline may touch.
from lib.tl.model                import TLModel
from lib.mtl.model               import MultiTaskDeep4Net
from braindecode.models.deep4    import Deep4Net
from braindecode.models.modules  import Ensure4d, CombinedConv
from braindecode.models.modules  import Expression
from einops.layers.torch         import Rearrange
from torch.nn.modules.conv       import Conv2d
from torch.nn.modules.linear     import Linear
from torch.nn.modules.batchnorm  import BatchNorm1d, BatchNorm2d
import torch.nn.functional        as F

# Register them *before* any torch.load or joblib.load
torch.serialization.add_safe_globals([
    TLModel,
    MultiTaskDeep4Net,
    Deep4Net,
    Ensure4d,
    CombinedConv,
    Expression,
    Rearrange,
    Conv2d,
    Linear,
    BatchNorm1d,
    BatchNorm2d,
    F.elu,
])

# 2) Now it’s safe to unpickle your entire Skorch pipeline:
pipeline = joblib.load("./data/models/tl_pooled_model.joblib")
print("✅ Pipeline loaded!")

# 3) Quick sanity‐check a dummy predict call:
import numpy as np
X_dummy = np.zeros((1, 8, 600), dtype=np.float32)
print("Dummy output:", pipeline.predict(X_dummy))
