"""FOG-HNet: unified model with task-specific head."""
import torch.nn as nn
from .backbone import FOGHNetBackbone


class FOGHNet(nn.Module):
    """
    task = "detection" -> binary {Normal, FoG}
    task = "subtype"   -> 3-class {Start Hesitation, Turn, Walking}
    """
    def __init__(self, task="detection", num_classes=None, **backbone_kwargs):
        super().__init__()
        assert task in ("detection", "subtype")
        self.task = task
        self.backbone = FOGHNetBackbone(**backbone_kwargs)

        if num_classes is None:
            num_classes = 2 if task == "detection" else 3
        self.num_classes = num_classes

        d = self.backbone.d_model
        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(d, num_classes),
        )

    def forward(self, x):
        h = self.backbone(x)            # (B, T', D)
        h = h.mean(dim=1)               # global average pooling
        return self.head(h)