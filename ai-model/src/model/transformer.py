"""
Transformer Decoder Block — the repeating unit of goktugGPT.

Each block follows the Pre-LN (pre-layer-norm) variant used in GPT-2:

  x = x + Attention(LayerNorm(x))
  x = x + FFN(LayerNorm(x))

Why Pre-LN over Post-LN?
  - More stable gradients during early training
  - Less sensitive to learning rate

The Feed-Forward Network (FFN) is a 2-layer MLP with GELU activation:

  FFN(x) = W_2 · GELU(W_1 · x + b_1) + b_2

  Hidden dim = 4 × n_embed  (standard GPT convention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import MultiHeadSelfAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    FFN(x) = W_2 · GELU(W_1 · x)

    Expansion factor 4 is standard (from "Attention is All You Need").
    GELU (Gaussian Error Linear Unit) outperforms ReLU on language tasks.
    """

    def __init__(self, n_embed: int, dropout: float = 0.1, expansion: int = 4):
        super().__init__()
        hidden = n_embed * expansion
        self.net = nn.Sequential(
            nn.Linear(n_embed, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, n_embed, bias=True),
            nn.Dropout(dropout),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Single Pre-LN Transformer Decoder block:

      x ← x + MHSA(LayerNorm(x))   [self-attention sub-layer]
      x ← x + FFN(LayerNorm(x))    [feed-forward sub-layer]
    """

    def __init__(
        self,
        n_embed: int,
        n_head: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = MultiHeadSelfAttention(n_embed, n_head, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffn = FeedForward(n_embed, dropout)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Self-attention sub-layer (Pre-LN)
        if return_attn:
            attn_out, attn_weights = self.attn(self.ln1(x), return_attn=True)
        else:
            attn_out = self.attn(self.ln1(x))
        x = x + attn_out

        # Feed-forward sub-layer (Pre-LN)
        x = x + self.ffn(self.ln2(x))

        if return_attn:
            return x, attn_weights
        return x


class TransformerDecoder(nn.Module):
    """
    Stack of N TransformerBlock layers.

    This is the "backbone" of goktugGPT — it transforms the embedded
    token sequence into rich contextual representations.
    """

    def __init__(
        self,
        n_embed: int,
        n_head: int,
        n_layer: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(n_embed, n_head, dropout, max_seq_len)
                for _ in range(n_layer)
            ]
        )
        self.n_layer = n_layer
        self.use_gradient_checkpointing = False

    def forward(
        self, x: torch.Tensor, return_all_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
        all_attn = []
        for block in self.blocks:
            if return_all_attn:
                x, attn = block(x, return_attn=True)
                all_attn.append(attn)
            elif self.use_gradient_checkpointing and x.requires_grad:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        if return_all_attn:
            return x, all_attn
        return x
