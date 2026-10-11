"""
Train FOG-HNet for either task.

Usage:
    python train.py --config configs/default.yaml --task detection
    python train.py --config configs/default.yaml --task subtype
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
    p.add_argument("--task", type=str, default=None,
                   choices=["detection", "subtype"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)

    if args.task is not None:
        cfg.train.task = args.task
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size

    set_seed(cfg.train.seed)

    # -------------------------------------------------
    # Data
    # -------------------------------------------------
    if cfg.data.use_synthetic:
        X, y_det, y_sub = make_synthetic_dataset(
            n=400, seq_len=cfg.data.seq_len,
            in_channels=cfg.model.in_channels,
            seed=cfg.train.seed,
        )
    else:
        raise NotImplementedError(
            "Plug your real dataset loader here. "
            "Return X, y_det, y_sub as tensors."
        )

    if cfg.train.task == "detection":
        y = y_det
        num_classes = cfg.model.num_classes_detection
    else:
        # Subtype model is trained only on FoG episodes
        mask = (y_det == 1)
        X, y = X[mask], y_sub[mask]
        num_classes = cfg.model.num_classes_subtype

    train_loader, val_loader, test_loader = build_dataloaders(
        X, y, batch_size=cfg.data.batch_size,
        train_ratio=cfg.data.train_split,
        val_ratio=cfg.data.val_split,
        num_workers=cfg.data.num_workers,
        seed=cfg.train.seed,
    )
    print(f"Train / Val / Test batches: "
          f"{len(train_loader)} / {len(val_loader)} / {len(test_loader)}")

    # -------------------------------------------------
    # Model
    # -------------------------------------------------
    model = FOGHNet(
        task=cfg.train.task,
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

    # -------------------------------------------------
    # Trainer
    # -------------------------------------------------
    trainer = Trainer(
        model,
        device=cfg.train.device,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        grad_clip=cfg.train.grad_clip,
        checkpoint_dir=cfg.train.checkpoint_dir,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=cfg.train.epochs,
    )
    trainer.fit(train_loader, val_loader,
                epochs=cfg.train.epochs, scheduler=scheduler)

    # Final test metrics
    test_metrics = trainer.evaluate(test_loader)
    print("\n=== Test metrics ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()