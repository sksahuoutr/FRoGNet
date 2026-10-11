"""
Evaluate a saved FOG-HNet checkpoint.

Usage:
    python test.py --task detection --ckpt checkpoints/best_detection.pt
    python test.py --task subtype   --ckpt checkpoints/best_subtype.pt
"""
import argparse
import torch
from fog_hnet.config import Config
from fog_hnet.data import make_synthetic_dataset, build_dataloaders
from fog_hnet.models import FOGHNet
from fog_hnet.utils import set_seed, Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--task", type=str, required=True,
                   choices=["detection", "subtype"])
    p.add_argument("--ckpt", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    cfg.train.task = args.task
    set_seed(cfg.train.seed)

    # Data
    if cfg.data.use_synthetic:
        X, y_det, y_sub = make_synthetic_dataset(
            n=400, seq_len=cfg.data.seq_len,
            in_channels=cfg.model.in_channels, seed=cfg.train.seed,
        )
    else:
        raise NotImplementedError("Plug your real dataset loader here.")

    if args.task == "detection":
        y = y_det
        num_classes = cfg.model.num_classes_detection
    else:
        mask = (y_det == 1)
        X, y = X[mask], y_sub[mask]
        num_classes = cfg.model.num_classes_subtype

    _, _, test_loader = build_dataloaders(
        X, y, batch_size=cfg.data.batch_size,
        train_ratio=cfg.data.train_split,
        val_ratio=cfg.data.val_split,
        num_workers=cfg.data.num_workers, seed=cfg.train.seed,
    )

    # Model
    model = FOGHNet(
        task=args.task,
        num_classes=num_classes,
        in_channels=cfg.model.in_channels,
        mscnn_channels=tuple(cfg.model.mscnn_channels),
        kernel_sizes=tuple(cfg.model.kernel_sizes),
        d_model=cfg.model.d_model,
        d_state=cfg.model.d_state,
        rbme_layers=cfg.model.rbme_layers,
        gat_heads=cfg.model.gat_heads,
        gat_layers=cfg.model.gat_layers,
        knn_k=cfg.model.knn_k,
        dropout=cfg.model.dropout,
    )
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.ckpt}")

    trainer = Trainer(model, device=cfg.train.device,
                      checkpoint_dir=cfg.train.checkpoint_dir)
    metrics = trainer.evaluate(test_loader)
    print("\n=== Test metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()