from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    # --- Tokenizer ---
    vocab_size: int = 8000

    # --- Architecture ---
    n_embed: int = 256       # embedding dimension (d_model)
    n_head: int = 8          # number of attention heads
    n_layer: int = 6         # number of transformer decoder layers
    dropout: float = 0.1
    max_seq_len: int = 512   # maximum context length
    rope_theta: float = 10000.0  # RoPE base frequency (raise for long context)

    # --- Special tokens ---
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    user_token: str = "<user>"
    assistant_token: str = "<assistant>"
    think_start_token: str = "<think>"
    think_end_token: str = "</think>"

    # --- Training ---
    batch_size: int = 8              # micro-batch (per step)
    grad_accum_steps: int = 1        # effective batch = batch_size * grad_accum_steps
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1        # cosine decays to lr * this
    weight_decay: float = 0.1
    max_epochs: int = 50
    grad_clip: float = 1.0
    warmup_steps: int = 200
    eval_interval: int = 250
    save_interval: int = 500
    precision: str = "bf16"          # "bf16" | "fp16" | "fp32" (bf16 best on RTX 30xx+/40xx/50xx)
    compile: bool = False            # torch.compile the model (big speedup, needs Triton)

    # --- Generation ---
    max_new_tokens: int = 200
    max_think_tokens: int = 60
    temperature: float = 0.7
    top_k: int = 30
    top_p: float = 0.9
    repetition_penalty: float = 1.3

    # --- Paths ---
    data_path: str = "data/train.txt"
    checkpoint_dir: str = "checkpoints"
    tokenizer_path: str = "checkpoints/tokenizer.json"

    @property
    def special_tokens(self) -> List[str]:
        return [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
            self.user_token,
            self.assistant_token,
            self.think_start_token,
            self.think_end_token,
        ]


@dataclass
class TinyConfig(ModelConfig):
    """Tiny config for quick training on CPU."""
    vocab_size: int = 4000
    n_embed: int = 128
    n_head: int = 4
    n_layer: int = 3
    max_seq_len: int = 256
    batch_size: int = 4
    max_epochs: int = 80   # more epochs for small data
    warmup_steps: int = 100
    eval_interval: int = 100
    save_interval: int = 200


@dataclass
class MediumConfig(ModelConfig):
    """Medium config — good balance of quality and speed. Needs GPU (4GB+ VRAM).
    Designed for use with data/train_clean.txt (Dolly + Alpaca + synthetic QA).
    Run prepare_data.py first to generate the training data.
    """
    vocab_size: int = 16000
    n_embed: int = 512
    n_head: int = 8
    n_layer: int = 8
    max_seq_len: int = 512   # 512 is enough for QA; reduces memory vs 1024
    batch_size: int = 16
    max_epochs: int = 80     # more epochs — dataset is larger now
    warmup_steps: int = 500  # longer warmup for stability on more data
    eval_interval: int = 500
    save_interval: int = 1000
    data_path: str = "data/train_clean.txt"


@dataclass
class LargeConfig(ModelConfig):
    """Large config — high quality. Needs GPU (8GB+ VRAM). ~85M parameters."""
    vocab_size: int = 32000
    n_embed: int = 768
    n_head: int = 12
    n_layer: int = 12
    max_seq_len: int = 1024
    batch_size: int = 16
    max_epochs: int = 30
    warmup_steps: int = 1000
    eval_interval: int = 500
    save_interval: int = 1000


@dataclass
class XLConfig(ModelConfig):
    """
    XL config — designed to push a single high-end GPU (e.g. RTX 5090 / 32GB)
    with the modern stack (RoPE · RMSNorm · SwiGLU · Flash-Attn · bf16).
    ~0.75B parameters. Larger vocab for richer Turkish + English coverage.

    Fits in 32GB with bf16 + gradient checkpointing + gradient accumulation.
    Bump grad_accum_steps for a larger effective batch; lower batch_size if OOM.
    """
    vocab_size: int = 48000
    n_embed: int = 1536
    n_head: int = 24
    n_layer: int = 24
    max_seq_len: int = 2048
    rope_theta: float = 50000.0      # higher base for the 2048 context
    dropout: float = 0.05
    batch_size: int = 12             # micro-batch
    grad_accum_steps: int = 8        # effective batch = 96
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_epochs: int = 3              # big data → few epochs
    warmup_steps: int = 2000
    eval_interval: int = 1000
    save_interval: int = 2000
    precision: str = "bf16"
    compile: bool = True
    data_path: str = "data/train_clean.txt"
