"""Per-channel z-score normalization on input accelerometer signal."""
import torch.nn as nn


class ZScoreNormalizer(nn.Module):
    """x: (B, T, C) — standardize each channel independently."""

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6
        return (x - mean) / std