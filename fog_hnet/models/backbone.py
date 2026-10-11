"""Full MSCNN -> RBME -> GAT backbone."""
import torch.nn as nn
from .normalizer import ZScoreNormalizer
from .mscnn import MSCNN
from .rbme import RBME
from .gat import GATBlock
from .graph_utils import build_knn_graph


class FOGHNetBackbone(nn.Module):
    def __init__(self,
                 in_channels=3,
                 mscnn_channels=(32, 32, 32),
                 kernel_sizes=(3, 5, 7),
                 d_model=64,
                 d_state=16,
                 rbme_layers=1,
                 gat_heads=4,
                 gat_layers=2,
                 knn_k=8,
                 dropout=0.1):
        super().__init__()
        self.normalizer = ZScoreNormalizer()
        self.mscnn = MSCNN(in_channels, out_dim=d_model,
                           branch_channels=mscnn_channels,
                           kernel_sizes=kernel_sizes)
        self.rbme = RBME(d_model, d_state=d_state, num_layers=rbme_layers)
        self.gat = GATBlock(d_model, num_heads=gat_heads,
                            num_layers=gat_layers, dropout=dropout)
        self.gat_norm = nn.LayerNorm(d_model)
        self.knn_k = knn_k
        self.d_model = d_model

    def forward(self, x):  # x: (B, T, C)
        x = self.normalizer(x)
        h = self.mscnn(x)                       # (B, T', D)
        h = self.rbme(h)                        # (B, T', D)
        adj, _ = build_knn_graph(h, k=self.knn_k, self_loop=True)
        h_g = self.gat(h, adj)
        return self.gat_norm(h + h_g)           # (B, T', D)