"""Multi-Scale CNN (MSCNN) with parallel 1-D conv branches."""
import torch
import torch.nn as nn


class MSCNN(nn.Module):
    def __init__(self, in_channels=3, out_dim=64,
                 branch_channels=(32, 32, 32), kernel_sizes=(3, 5, 7)):
        super().__init__()
        assert len(branch_channels) == len(kernel_sizes)
        self.branches = nn.ModuleList()
        for c_out, k in zip(branch_channels, kernel_sizes):
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_channels, c_out, kernel_size=k,
                          padding=k // 2, bias=False),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
            ))
        total_c = sum(branch_channels)
        self.proj = nn.Conv1d(total_c, out_dim, kernel_size=1, bias=False)
        self.out_dim = out_dim

    def forward(self, x):  # x: (B, T, C_in) -> (B, T, D)
        x = x.transpose(1, 2)                              # (B, C_in, T)
        ys = [branch(x) for branch in self.branches]
        y = torch.cat(ys, dim=1)                           # (B, sum Ci, T)
        y = self.proj(y)                                   # (B, D, T)
        return y.transpose(1, 2)                           # (B, T, D)