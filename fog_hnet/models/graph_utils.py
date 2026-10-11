"""Cosine similarity and directed k-NN graph construction."""
import torch
import torch.nn.functional as F


def build_knn_graph(h, k=8, self_loop=True):
    """
    h:   (B, T, D) contextual embeddings
    Returns:
        adj: (B, T, T) binary adjacency (float)
        S:   (B, T, T) cosine similarity matrix
    """
    h_norm = F.normalize(h, p=2, dim=-1)
    S = torch.bmm(h_norm, h_norm.transpose(1, 2))          # (B, T, T)

    T = S.size(-1)
    eye = torch.eye(T, device=S.device, dtype=torch.bool).unsqueeze(0)
    S_masked = S.masked_fill(eye, float("-inf"))

    k = min(k, T - 1)
    _, topk_idx = torch.topk(S_masked, k=k, dim=-1)

    adj = torch.zeros_like(S)
    adj.scatter_(-1, topk_idx, 1.0)
    if self_loop:
        adj = adj + torch.eye(T, device=S.device).unsqueeze(0)
    adj = (adj > 0).float()
    return adj, S