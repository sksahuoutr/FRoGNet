"""Generic Trainer for FOG-HNet (detection or subtype)."""
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from .metrics import compute_metrics


class Trainer:
    def __init__(self, model, device="cuda", lr=1e-3, weight_decay=1e-4,
                 grad_clip=1.0, checkpoint_dir="checkpoints"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(),
                                           lr=lr, weight_decay=weight_decay)
        self.grad_clip = grad_clip
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.history = {"train_loss": [], "val_loss": [],
                        "train_acc": [], "val_acc": []}

    # -------------------------------------------------
    def _run_epoch(self, loader, train=True):
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0
        all_true, all_pred = [], []
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            pbar = tqdm(loader, desc="train" if train else "eval", leave=False)
            for x, y in pbar:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       self.grad_clip)
                    self.optimizer.step()
                total_loss += loss.item() * x.size(0)
                preds = logits.argmax(-1)
                correct += (preds == y).sum().item()
                total += x.size(0)
                all_true.extend(y.cpu().tolist())
                all_pred.extend(preds.cpu().tolist())
        return total_loss / total, correct / total, all_true, all_pred

    # -------------------------------------------------
    def fit(self, train_loader, val_loader, epochs=30, scheduler=None):
        best_f1 = -1.0
        ckpt_path = os.path.join(self.checkpoint_dir,
                                 f"best_{self.model.task}.pt")
        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc, _, _ = self._run_epoch(train_loader, train=True)
            va_loss, va_acc, va_true, va_pred = self._run_epoch(val_loader, train=False)
            if scheduler is not None:
                scheduler.step()

            metrics = compute_metrics(va_true, va_pred)
            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_acc"].append(va_acc)

            print(f"[{epoch:03d}] train_loss={tr_loss:.4f} "
                  f"train_acc={tr_acc:.4f} | "
                  f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} "
                  f"val_f1={metrics['macro_f1']:.4f}")

            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"  ↳ saved checkpoint: {ckpt_path} (F1={best_f1:.4f})")
        return self.history

    # -------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader):
        _, _, y_true, y_pred = self._run_epoch(loader, train=False)
        return compute_metrics(y_true, y_pred)