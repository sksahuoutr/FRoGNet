"""Selective State-Space (Mamba) block and Mamba encoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """Simplified selective SSM with sequential recurrence (readable)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, 2 * d_state + 1, bias=False)
        self.dt_proj = nn.Linear(2 * d_state + 1, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.expand_proj = nn.Linear(d_model, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):  # x: (B, T, D)
        B, T, _ = x.shape
        h = self.expand_proj(x)                            # (B, T, d_inner)
        h_conv = self.conv1d(h.transpose(1, 2))[..., :T].transpose(1, 2)
        h_conv = F.silu(h_conv)

        params = self.x_proj(h_conv)                       # (B, T, 2*d_state+1)
        _, B_t, C_t = torch.split(params, [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(params))              # (B, T, d_inner)

        A = -torch.exp(self.A_log)                         # (d_inner, d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A)               # (B, T, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B_t.unsqueeze(2)           # (B, T, d_inner, d_state)

        s = torch.zeros(B, self.d_inner, self.d_state,
                        device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            s = dA[:, t] * s + dB[:, t] * h_conv[:, t].unsqueeze(-1)
            y = (s * C_t[:, t].unsqueeze(1)).sum(-1)
            ys.append(y)
        y = torch.stack(ys, dim=1)                         # (B, T, d_inner)
        y = y + self.D * h_conv
        return self.out_proj(y)


class MambaEncoder(nn.Module):
    """Pre-norm Mamba block with FFN."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, ffn_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_ratio * d_model),
            nn.GELU(),
            nn.Linear(ffn_ratio * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.ssm(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x