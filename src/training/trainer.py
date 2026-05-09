"""
Training loop for goktugGPT.

Features:
  • Cosine LR schedule with linear warmup
  • Gradient clipping
  • Periodic evaluation on validation set
  • Checkpoint saving (best model + latest)
  • Live loss logging to console + loss_history.txt
  • Automatic device detection (CUDA > MPS > CPU)
"""

import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..model import GoktugGPT


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU — training will be slow; consider a smaller config.")
    return device


def cosine_lr_with_warmup(
    step: int, warmup_steps: int, total_steps: int, min_lr: float, max_lr: float
) -> float:
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


class Trainer:
    """
    Manages the full training loop for goktugGPT.

    Args:
        model:          GoktugGPT instance.
        train_dl:       Training DataLoader.
        val_dl:         Validation DataLoader.
        config:         ModelConfig / TinyConfig / MediumConfig.
        checkpoint_dir: Directory to save checkpoints.
    """

    def __init__(
        self,
        model: GoktugGPT,
        train_dl: DataLoader,
        val_dl: DataLoader,
        config,
        checkpoint_dir: str = "checkpoints",
    ):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.device = _get_device()
        self.model = model.to(self.device)

        self.train_dl = train_dl
        self.val_dl = val_dl

        # AdamW with weight-decay (don't decay biases / norms)
        decay_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and p.dim() >= 2
        ]
        no_decay_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and p.dim() < 2
        ]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Mixed precision scaler (only active when CUDA is available)
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        if self.use_amp:
            print("Mixed precision (AMP fp16) enabled — faster training, lower VRAM usage.")
            # Gradient checkpointing: trades compute for memory (recomputes activations
            # during backward instead of storing them — cuts VRAM usage ~40%)
            self.model.enable_gradient_checkpointing()
            print("Gradient checkpointing enabled — ~40% less VRAM, ~20% slower.")

        # Total steps for LR schedule
        self.total_steps = config.max_epochs * len(train_dl)
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Loss history for plotting
        self.train_losses: list = []
        self.val_losses: list = []

        self._log_path = os.path.join(checkpoint_dir, "loss_history.txt")

    # ------------------------------------------------------------------
    # LR update
    # ------------------------------------------------------------------

    def _update_lr(self):
        lr = cosine_lr_with_warmup(
            self.global_step,
            self.config.warmup_steps,
            self.total_steps,
            min_lr=self.config.learning_rate * 0.1,
            max_lr=self.config.learning_rate,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        for x, y in self.val_dl:
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x, targets=y)
            total_loss += out["loss"].item()
            n_batches += 1
            if n_batches >= 20:  # cap eval at 20 batches for speed
                break
        self.model.train()
        return total_loss / max(1, n_batches)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, resume_from: Optional[str] = None):
        """
        Run the full training loop.

        Args:
            resume_from: Path to a checkpoint to resume from.
        """
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            payload = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(payload["model_state"])
            extra = {k: v for k, v in payload.items() if k != "model_state"}
            self.global_step = extra.get("global_step", 0)
            self.best_val_loss = extra.get("best_val_loss", float("inf"))
            # Calculate correct start epoch from global_step
            steps_per_epoch = len(self.train_dl)
            start_epoch = self.global_step // steps_per_epoch
            # Restore optimizer state (critical for LR schedule continuity)
            if "optimizer_state" in extra and extra["optimizer_state"] is not None:
                self.optimizer.load_state_dict(extra["optimizer_state"])
                print("  Optimizer state restored.")
            if self.use_amp and "scaler_state" in extra and extra["scaler_state"] is not None:
                self.scaler.load_state_dict(extra["scaler_state"])
                print("  AMP scaler state restored.")
            print(
                f"Resumed from checkpoint: {resume_from}\n"
                f"  Epoch: {start_epoch + 1}  |  Global step: {self.global_step}  |"
                f"  Best val loss so far: {self.best_val_loss:.4f}"
            )

        print(
            f"\n{'='*60}\n"
            f"  Training goktugGPT\n"
            f"  Epochs: {self.config.max_epochs}  |  "
            f"Steps/epoch: {len(self.train_dl)}  |  "
            f"Total: {self.total_steps}\n"
            f"  Parameters: {self.model.num_parameters()/1e6:.2f}M\n"
            f"{'='*60}\n"
        )

        self.model.train()
        t0 = time.time()

        # How many steps into the current epoch we already completed
        steps_per_epoch = len(self.train_dl)
        skip_batches = self.global_step % steps_per_epoch

        for epoch in range(start_epoch, self.config.max_epochs):
            epoch_loss = 0.0
            n_batches_counted = 0
            pbar = tqdm(
                self.train_dl,
                desc=f"Epoch {epoch+1}/{self.config.max_epochs}",
                dynamic_ncols=True,
            )

            for batch_idx, (x, y) in enumerate(pbar):
                # Skip batches already completed in this epoch (resume case)
                if skip_batches > 0 and batch_idx < skip_batches:
                    continue
                x, y = x.to(self.device), y.to(self.device)

                # Update LR
                lr = self._update_lr()

                # Forward (mixed precision)
                self.optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    out = self.model(x, targets=y)
                    loss = out["loss"]

                # Backward (scaler handles fp16 gradient scaling)
                self.scaler.scale(loss).backward()

                # Gradient clipping (unscale first)
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                loss_val = loss.item()
                epoch_loss += loss_val
                n_batches_counted += 1
                self.global_step += 1
                self.train_losses.append((self.global_step, loss_val))

                pbar.set_postfix(
                    loss=f"{loss_val:.4f}",
                    lr=f"{lr:.2e}",
                    step=self.global_step,
                )

                # Periodic evaluation
                if self.global_step % self.config.eval_interval == 0:
                    val_loss = self.evaluate()
                    self.val_losses.append((self.global_step, val_loss))
                    elapsed = time.time() - t0
                    print(
                        f"\n[Step {self.global_step}] "
                        f"train={loss_val:.4f}  val={val_loss:.4f}  "
                        f"lr={lr:.2e}  elapsed={elapsed/60:.1f}min"
                    )
                    self._log(self.global_step, loss_val, val_loss, lr)

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save("best_model.pt", epoch)
                        print(f"  → New best model (val_loss={val_loss:.4f})")

                # Periodic checkpoint
                if self.global_step % self.config.save_interval == 0:
                    self._save(f"checkpoint_step_{self.global_step}.pt", epoch)

            avg_epoch_loss = epoch_loss / max(1, n_batches_counted)
            print(
                f"Epoch {epoch+1} done | avg_loss={avg_epoch_loss:.4f}"
            )
            # Only skip batches for the first resumed epoch
            skip_batches = 0

        # Final save
        self._save("final_model.pt", self.config.max_epochs - 1)
        total_time = (time.time() - t0) / 60
        print(f"\nTraining complete in {total_time:.1f} minutes.")
        print(f"Best val loss: {self.best_val_loss:.4f}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save(self, filename: str, epoch: int):
        path = os.path.join(self.checkpoint_dir, filename)
        self.model.save_checkpoint(
            path,
            extra={
                "epoch": epoch,
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
                "optimizer_state": self.optimizer.state_dict(),
                "scaler_state": self.scaler.state_dict() if self.use_amp else None,
                "config": {
                    "vocab_size": self.config.vocab_size,
                    "n_embed": self.config.n_embed,
                    "n_head": self.config.n_head,
                    "n_layer": self.config.n_layer,
                    "max_seq_len": self.config.max_seq_len,
                },
            },
        )

    def _log(self, step: int, train_loss: float, val_loss: float, lr: float):
        with open(self._log_path, "a") as f:
            f.write(f"step={step}\ttrain={train_loss:.6f}\tval={val_loss:.6f}\tlr={lr:.2e}\n")
