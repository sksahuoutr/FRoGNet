"""Multi-head Graph Attention Network (GAT)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1, negative_slope=0.2):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.W = nn.Linear(d_model, d_model, bias=False)
        self.a_src = nn.Parameter(torch.zeros(num_heads, self.d_head))
        self.a_dst = nn.Parameter(torch.zeros(num_heads, self.d_head))
        self.leaky = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, adj):
        B, N, D = x.shape
        h = self.W(x).view(B, N, self.num_heads, self.d_head)

        e_src = (h * self.a_src).sum(-1)                       # (B, N, M)
        e_dst = (h * self.a_dst).sum(-1)                       # (B, N, M)
        e = self.leaky(e_src.unsqueeze(2) + e_dst.unsqueeze(1))  # (B, N, N, M)

        mask = adj.unsqueeze(-1).bool()                        # (B, N, N, 1)
        e = e.masked_fill(~mask, float("-inf"))
        attn = F.softmax(e, dim=2)
        attn = self.dropout(attn)

        out = torch.einsum("bijm,bjmd->bimd", attn, h)
        out = out.reshape(B, N, D)
        return self.norm(x + out)


class GATBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GATLayer(d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x