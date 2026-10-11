"""Synthetic tri-axial accelerometer dataset for smoke-testing."""
import torch


def make_synthetic_dataset(n=400, seq_len=256, in_channels=3, seed=0):
    """
    Returns:
        X: (N, T, C) float tensor
        y_det: (N,) binary labels {0=Normal, 1=FoG}
        y_sub: (N,) 3-class subtype labels
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, seq_len, in_channels, generator=g)
    y_det = torch.randint(0, 2, (n,), generator=g)
    y_sub = torch.randint(0, 3, (n,), generator=g)
    return X, y_det, y_sub