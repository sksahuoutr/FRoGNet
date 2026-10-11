"""Residual Bidirectional Mamba Encoder (RBME)."""
import torch
import torch.nn as nn
from .mamba import MambaEncoder


class RBME(nn.Module):
    def __init__(self, d_model, d_state=16, num_layers=1, ffn_ratio=4):
        super().__init__()
        self.forward_layers = nn.ModuleList([
            MambaEncoder(d_model, d_state=d_state, ffn_ratio=ffn_ratio)
            for _ in range(num_layers)
        ])
        self.backward_layers = nn.ModuleList([
            MambaEncoder(d_model, d_state=d_state, ffn_ratio=ffn_ratio)
            for _ in range(num_layers)
        ])
        self.fuse = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_ratio * d_model),
            nn.GELU(),
            nn.Linear(ffn_ratio * d_model, d_model),
        )
        self.res_norm = nn.LayerNorm(d_model)

    def forward(self, x):  # (B, T, D)
        h_f = x
        for layer in self.forward_layers:
            h_f = layer(h_f)

        h_b = torch.flip(x, dims=[1])
        for layer in self.backward_layers:
            h_b = layer(h_b)
        h_b = torch.flip(h_b, dims=[1])

        h_cat = torch.cat([h_f, h_b], dim=-1)
        h_hat = self.fuse(h_cat)

        f = self.ffn(self.norm(h_hat))
        return self.res_norm(h_hat + f)