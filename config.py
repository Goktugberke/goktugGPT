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
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_epochs: int = 50
    grad_clip: float = 1.0
    warmup_steps: int = 200
    eval_interval: int = 250
    save_interval: int = 500

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
    """Medium config — good balance of quality and speed."""
    vocab_size: int = 16000
    n_embed: int = 512
    n_head: int = 8
    n_layer: int = 8
    max_seq_len: int = 1024
    batch_size: int = 4
    max_epochs: int = 100
