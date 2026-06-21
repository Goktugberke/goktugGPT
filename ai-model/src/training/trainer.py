"""
Training loop for goktugGPT.

Features:
  • bf16 / fp16 / fp32 mixed precision (bf16 preferred on RTX 30xx/40xx/50xx —
    no gradient scaler needed, more stable than fp16)
  • torch.compile (big speedup on modern GPUs)
  • Gradient accumulation (large effective batch on a single GPU)
  • Gradient checkpointing (trades compute for VRAM)
  • TF32 matmuls on Ampere+ GPUs
  • Cosine LR schedule with linear warmup, decoupled weight decay (no decay on
    norms / biases / 1-D params)
  • Periodic eval, best/step checkpointing, resume.
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
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Faster matmuls on Ampere+ (incl. RTX 50xx) with negligible quality loss.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU — training will be slow; consider a smaller config.")
    return device


def cosine_lr_with_warmup(step, warmup_steps, total_steps, min_lr, max_lr) -> float:
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


class Trainer:
    def __init__(self, model: GoktugGPT, train_dl: DataLoader, val_dl: DataLoader,
                 config, checkpoint_dir: str = "checkpoints"):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.device = _get_device()
        self.raw_model = model.to(self.device)          # for save/load/methods
        self.grad_accum = max(1, getattr(config, "grad_accum_steps", 1))

        # --- Precision ---
        prec = getattr(config, "precision", "fp16")
        cuda = self.device.type == "cuda"
        if cuda and prec == "bf16" and torch.cuda.is_bf16_supported():
            self.amp_dtype, self.use_amp, self.use_scaler = torch.bfloat16, True, False
            print("Precision: bf16 (no grad scaler needed).")
        elif cuda and prec == "fp16":
            self.amp_dtype, self.use_amp, self.use_scaler = torch.float16, True, True
            print("Precision: fp16 (AMP + grad scaler).")
        else:
            self.amp_dtype, self.use_amp, self.use_scaler = torch.float32, False, False
            print(f"Precision: fp32 ({'cuda' if cuda else self.device.type}).")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)

        if cuda:
            self.raw_model.enable_gradient_checkpointing()
            print("Gradient checkpointing enabled (~40% less VRAM).")

        # --- torch.compile (after grad-checkpointing flag is set) ---
        self.model = self.raw_model
        if getattr(config, "compile", False) and cuda:
            try:
                self.model = torch.compile(self.raw_model)
                print("torch.compile enabled.")
            except Exception as e:
                print(f"torch.compile unavailable ({e}); continuing eager.")

        self.train_dl = train_dl
        self.val_dl = val_dl

        # --- Optimizer: decoupled weight decay, none on 1-D params (norms/bias) ---
        decay = [p for p in self.raw_model.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.raw_model.parameters() if p.requires_grad and p.dim() < 2]
        fused_ok = cuda and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": config.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=config.learning_rate, betas=(0.9, 0.95), eps=1e-8,
            **({"fused": True} if fused_ok else {}),
        )

        # Optimizer-step accounting (LR/eval/save operate on optimizer steps)
        self.steps_per_epoch = max(1, len(train_dl) // self.grad_accum)
        self.total_steps = config.max_epochs * self.steps_per_epoch
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.train_losses, self.val_losses = [], []
        self._log_path = os.path.join(checkpoint_dir, "loss_history.txt")

    def _update_lr(self):
        lr = cosine_lr_with_warmup(
            self.global_step, self.config.warmup_steps, self.total_steps,
            min_lr=self.config.learning_rate * getattr(self.config, "min_lr_ratio", 0.1),
            max_lr=self.config.learning_rate,
        )
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for x, y in self.val_dl:
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                total += self.model(x, targets=y)["loss"].item()
            n += 1
            if n >= 20:
                break
        self.model.train()
        return total / max(1, n)

    def train(self, resume_from: Optional[str] = None):
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            payload = torch.load(resume_from, map_location=self.device)
            self.raw_model.load_state_dict(payload["model_state"])
            extra = {k: v for k, v in payload.items() if k != "model_state"}
            self.global_step = extra.get("global_step", 0)
            self.best_val_loss = extra.get("best_val_loss", float("inf"))
            start_epoch = self.global_step // self.steps_per_epoch
            if extra.get("optimizer_state"):
                self.optimizer.load_state_dict(extra["optimizer_state"])
                print("  Optimizer state restored.")
            if self.use_scaler and extra.get("scaler_state"):
                self.scaler.load_state_dict(extra["scaler_state"])
            print(f"Resumed: epoch {start_epoch+1}, step {self.global_step}, "
                  f"best val {self.best_val_loss:.4f}")

        eff_batch = self.config.batch_size * self.grad_accum
        print(f"\n{'='*60}\n  Training goktugGPT\n"
              f"  Params: {self.raw_model.num_parameters()/1e6:.2f}M  |  "
              f"Effective batch: {eff_batch} ({self.config.batch_size}×{self.grad_accum})\n"
              f"  Optimizer steps/epoch: {self.steps_per_epoch}  |  Total: {self.total_steps}\n{'='*60}\n")

        self.model.train()
        t0 = time.time()
        skip_micro = (self.global_step % self.steps_per_epoch) * self.grad_accum

        for epoch in range(start_epoch, self.config.max_epochs):
            epoch_loss, counted = 0.0, 0
            self.optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(self.train_dl, desc=f"Epoch {epoch+1}/{self.config.max_epochs}", dynamic_ncols=True)

            for batch_idx, (x, y) in enumerate(pbar):
                if skip_micro and batch_idx < skip_micro:
                    continue
                x, y = x.to(self.device), y.to(self.device)

                with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    loss = self.model(x, targets=y)["loss"] / self.grad_accum
                if self.use_scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                loss_val = loss.item() * self.grad_accum
                epoch_loss += loss_val
                counted += 1

                # Optimizer step every grad_accum micro-batches
                if (batch_idx + 1) % self.grad_accum == 0:
                    lr = self._update_lr()
                    if self.use_scaler:
                        self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.config.grad_clip)
                    if self.use_scaler:
                        self.scaler.step(self.optimizer); self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    self.global_step += 1
                    self.train_losses.append((self.global_step, loss_val))
                    pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr:.2e}", step=self.global_step)

                    if self.global_step % self.config.eval_interval == 0:
                        val_loss = self.evaluate()
                        self.val_losses.append((self.global_step, val_loss))
                        print(f"\n[Step {self.global_step}] train={loss_val:.4f} val={val_loss:.4f} "
                              f"lr={lr:.2e} elapsed={(time.time()-t0)/60:.1f}min")
                        self._log(self.global_step, loss_val, val_loss, lr)
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self._save("best_model.pt", epoch)
                            print(f"  -> New best (val={val_loss:.4f})")
                    if self.global_step % self.config.save_interval == 0:
                        self._save(f"checkpoint_step_{self.global_step}.pt", epoch)

            print(f"Epoch {epoch+1} done | avg_loss={epoch_loss/max(1,counted):.4f}")
            skip_micro = 0

        self._save("final_model.pt", self.config.max_epochs - 1)
        print(f"\nTraining complete in {(time.time()-t0)/60:.1f} min. Best val: {self.best_val_loss:.4f}")

    def _save(self, filename: str, epoch: int):
        path = os.path.join(self.checkpoint_dir, filename)
        self.raw_model.save_checkpoint(path, extra={
            "epoch": epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict() if self.use_scaler else None,
            "config": {
                "vocab_size": self.config.vocab_size,
                "n_embed": self.config.n_embed,
                "n_head": self.config.n_head,
                "n_layer": self.config.n_layer,
                "max_seq_len": self.config.max_seq_len,
                "rope_theta": getattr(self.config, "rope_theta", 10000.0),
            },
        })

    def _log(self, step, train_loss, val_loss, lr):
        with open(self._log_path, "a") as f:
            f.write(f"step={step}\ttrain={train_loss:.6f}\tval={val_loss:.6f}\tlr={lr:.2e}\n")
