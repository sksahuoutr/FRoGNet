"""Dataset wrappers and DataLoader builders."""
import torch
from torch.utils.data import TensorDataset, DataLoader


def _split(n, train_ratio, val_ratio):
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def build_dataloaders(X, y, batch_size=32, train_ratio=0.7, val_ratio=0.15,
                      num_workers=0, seed=42):
    """Return (train_loader, val_loader, test_loader)."""
    n = X.size(0)
    n_train, n_val, n_test = _split(n, train_ratio, val_ratio)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    X, y = X[perm], y[perm]

    X_tr, y_tr = X[:n_train], y[:n_train]
    X_va, y_va = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_te, y_te = X[n_train + n_val:], y[n_train + n_val:]

    mk = lambda X_, y_, sh: DataLoader(
        TensorDataset(X_, y_), batch_size=batch_size,
        shuffle=sh, num_workers=num_workers,
    )
    return mk(X_tr, y_tr, True), mk(X_va, y_va, False), mk(X_te, y_te, False)