import os
from pathlib import Path
from typing import Dict, Any, Tuple, List
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.utils.seed import set_seed


class Trainer:
    """Trainer pipeline for training and evaluating vision models on CIFAR-10."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device = None,
    ):
        """Initialize the Trainer.

        Args:
            model: PyTorch classification model.
            config: Full experiment configuration dictionary.
            device: Explicit torch.device (if None, auto-selects CUDA when available).
        """
        self.config = config
        self.seed = config.get("experiment", {}).get("seed", 42)
        set_seed(self.seed)

        if device is None:
            pref = config.get("training", {}).get("device", "auto")
            if pref == "cuda" and torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif pref == "auto" and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        train_cfg = config.get("training", {})
        lr = train_cfg.get("learning_rate", 0.001)
        weight_decay = train_cfg.get("weight_decay", 0.0005)
        opt_name = train_cfg.get("optimizer", "adamw").lower()

        if opt_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif opt_name == "sgd":
            momentum = train_cfg.get("momentum", 0.9)
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_name}")

        epochs = train_cfg.get("epochs", 15)
        sched_name = train_cfg.get("scheduler", "cosine").lower()
        if sched_name == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        else:
            self.scheduler = None

        raw_ckpt_dir = config.get("paths", {}).get("checkpoint_dir", "outputs/models")
        self.checkpoint_dir = Path(raw_ckpt_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Execute one training epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images, targets = images.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == targets).sum().item()
            total += images.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Execute validation epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in val_loader:
            images, targets = images.to(self.device), targets.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == targets).sum().item()
            total += images.size(0)

        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_name: str = "baseline_resnet18_best.pth",
    ) -> Dict[str, List[float]]:
        """Fit the model over all configured epochs and save the best checkpoint."""
        epochs = self.config.get("training", {}).get("epochs", 15)
        best_val_acc = 0.0
        best_checkpoint_path = self.checkpoint_dir / save_name

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        print(f"\nStarting model training on device: {self.device}")
        print(f"Total Epochs: {epochs} | Seed: {self.seed}")
        print(f"Checkpoint Output Path: {best_checkpoint_path}")
        print("-" * 65)

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate_epoch(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            elapsed = time.time() - ep_start

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%",
                end="",
                flush=True,
            )

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "feature_dim": getattr(self.model, "feature_dim", 512),
                    "config": self.config,
                }
                torch.save(checkpoint, best_checkpoint_path)
                print(" -> [BEST CHECKPOINT SAVED]", flush=True)
            else:
                print(flush=True)

        total_time = time.time() - start_time
        print("-" * 65)
        
        # Load best weights back into model
        if best_checkpoint_path.exists():
            best_ckpt = torch.load(best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(best_ckpt["model_state_dict"])

        print(
            f"Training finished in {total_time/60:.2f} mins. Best Val Acc: {best_val_acc*100:.2f}%"
        )
        print(f"Best Checkpoint saved to: {best_checkpoint_path}\n")

        return history
