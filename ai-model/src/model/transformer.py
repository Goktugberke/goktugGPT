"""
Transformer Decoder Block — the repeating unit of goktugGPT (LLaMA-style).

Modernisations over the original GPT-2-style block:
  • RMSNorm instead of LayerNorm — normalises by the root-mean-square only
    (no mean subtraction, no bias). Fewer ops, as good or better in practice;
    used by LLaMA/Mistral.
  • SwiGLU feed-forward instead of GELU MLP — a gated activation
    (SiLU(W1·x) ⊙ W3·x) projected back down by W2. Consistently beats plain
    GELU MLPs at the same parameter budget.
  • Pre-norm residual structure (kept from GPT-2 — stable training):
        x = x + Attention(RMSNorm(x))
        x = x + SwiGLU(RMSNorm(x))
  • KV-cache + RoPE position are threaded through to the attention layer.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attention import MultiHeadSelfAttention


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalisation (Zhang & Sennrich, 2019).

        RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight

    Unlike LayerNorm there is no mean subtraction and no bias term.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for numerical stability, then cast back.
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm.to(dtype)) * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network (Shazeer, 2020), as used in LLaMA.

        FFN(x) = W2( SiLU(W1·x) ⊙ W3·x )

    The hidden dim is set to ~2/3 · 4 · n_embed so the 3-matrix SwiGLU has a
    similar parameter count to a standard 2-matrix 4× GELU MLP.
    """

    def __init__(self, n_embed: int, dropout: float = 0.1, multiple_of: int = 64):
        super().__init__()
        hidden = int(2 * (4 * n_embed) / 3)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)  # round up
        self.w1 = nn.Linear(n_embed, hidden, bias=False)   # gate
        self.w3 = nn.Linear(n_embed, hidden, bias=False)   # up
        self.w2 = nn.Linear(hidden, n_embed, bias=False)   # down
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for lin in (self.w1, self.w3, self.w2):
            nn.init.normal_(lin.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """
    Single pre-norm Transformer decoder block (LLaMA-style):

        x ← x + Attention(RMSNorm(x))
        x ← x + SwiGLU(RMSNorm(x))
    """

    def __init__(
        self,
        n_embed: int,
        n_head: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.ln1 = RMSNorm(n_embed)
        self.attn = MultiHeadSelfAttention(n_embed, n_head, dropout, max_seq_len, rope_theta)
        self.ln2 = RMSNorm(n_embed)
        self.ffn = SwiGLU(n_embed, dropout)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        return_attn: bool = False,
    ):
        # --- Self-attention sub-layer (pre-norm) ---
        attn_ret = self.attn(
            self.ln1(x),
            start_pos=start_pos,
            past_kv=past_kv,
            use_cache=use_cache,
            return_attn=return_attn,
        )
        present_kv = None
        attn_weights = None
        if use_cache or return_attn:
            attn_out = attn_ret[0]
            idx = 1
            if use_cache:
                present_kv = attn_ret[idx]; idx += 1
            if return_attn:
                attn_weights = attn_ret[idx]
        else:
            attn_out = attn_ret
        x = x + attn_out

        # --- Feed-forward sub-layer (pre-norm) ---
        x = x + self.ffn(self.ln2(x))

        if not use_cache and not return_attn:
            return x
        result = (x,)
        if use_cache:
            result += (present_kv,)
        if return_attn:
            result += (attn_weights,)
        return result


class TransformerDecoder(nn.Module):
    """Stack of N TransformerBlock layers, with optional KV-cache."""

    def __init__(
        self,
        n_embed: int,
        n_head: int,
        n_layer: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(n_embed, n_head, dropout, max_seq_len, rope_theta)
                for _ in range(n_layer)
            ]
        )
        self.n_layer = n_layer
        self.use_gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_all_attn: bool = False,
    ):
        all_attn: List[torch.Tensor] = []
        present_kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None

            if return_all_attn:
                x, attn = block(x, start_pos=start_pos, return_attn=True)
                all_attn.append(attn)
            elif use_cache:
                x, present = block(x, start_pos=start_pos, past_kv=past_kv, use_cache=True)
                present_kvs.append(present)
            elif self.use_gradient_checkpointing and x.requires_grad:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x, start_pos=start_pos)

        if return_all_attn:
            return x, all_attn
        if use_cache:
            return x, present_kvs
        return x
