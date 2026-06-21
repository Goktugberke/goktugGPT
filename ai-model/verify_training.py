"""Smoke test for the trainer (grad accumulation + precision + optimizer cadence)
on synthetic data, CPU. Run: python verify_training.py"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from config import TinyConfig
from src.model import GoktugGPT
from src.training import Trainer

torch.manual_seed(0)
cfg = TinyConfig()
cfg.vocab_size = 200; cfg.n_embed = 64; cfg.n_head = 4; cfg.n_layer = 2
cfg.max_seq_len = 32; cfg.batch_size = 4; cfg.grad_accum_steps = 2
cfg.max_epochs = 1; cfg.warmup_steps = 2; cfg.eval_interval = 4
cfg.save_interval = 1000; cfg.precision = "bf16"; cfg.compile = False

def make_dl(n):
    x = torch.randint(0, cfg.vocab_size, (n, cfg.max_seq_len))
    y = torch.randint(0, cfg.vocab_size, (n, cfg.max_seq_len))
    return DataLoader(TensorDataset(x, y), batch_size=cfg.batch_size)

model = GoktugGPT(cfg.vocab_size, cfg.n_embed, cfg.n_head, cfg.n_layer,
                  dropout=0.0, max_seq_len=cfg.max_seq_len, rope_theta=cfg.rope_theta)
tr = Trainer(model, make_dl(40), make_dl(8), cfg, checkpoint_dir="checkpoints_smoketest")
tr.train()
assert tr.global_step > 0, "no optimizer steps taken"
print(f"\n[OK] trainer ran: {tr.global_step} optimizer steps, grad_accum={tr.grad_accum}")
print("TRAINING SMOKE TEST PASSED")
