"""Central configuration for FOG-HNet."""
from dataclasses import dataclass, field, asdict
from typing import Tuple
import yaml


@dataclass
class ModelConfig:
    in_channels: int = 3
    mscnn_channels: Tuple[int, int, int] = (32, 32, 32)
    kernel_sizes: Tuple[int, int, int] = (3, 5, 7)
    d_model: int = 64
    d_state: int = 16
    rbme_layers: int = 1
    gat_heads: int = 4
    gat_layers: int = 2
    knn_k: int = 8
    dropout: float = 0.1
    num_classes_detection: int = 2
    num_classes_subtype: int = 3


@dataclass
class DataConfig:
    batch_size: int = 32
    num_workers: int = 0
    seq_len: int = 256
    fs: int = 128
    window_sec: float = 2.0
    data_root: str = "data"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    use_synthetic: bool = True


@dataclass
class TrainConfig:
    task: str = "detection"        # "detection" | "subtype"
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    seed: int = 42


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        cfg = cls()
        for section in ("model", "data", "train"):
            for k, v in (raw.get(section) or {}).items():
                setattr(getattr(cfg, section), k, v)
        return cfg

    def to_dict(self):
        return asdict(self)