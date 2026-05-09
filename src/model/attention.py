"""
Multi-Head Causal Self-Attention — built from scratch.

Theory recap:
  Given input X ∈ R^(B×T×d_model):
    Q = X · W_Q,  K = X · W_K,  V = X · W_V       (linear projections)
    Attention(Q,K,V) = softmax(QK^T / √d_k) · V    (scaled dot-product)

  We split Q,K,V into `n_head` independent heads, compute attention for
  each head in parallel, then concatenate and project back.

  The *causal* mask forces each position to attend only to itself and
  earlier positions (prevents the model from "seeing the future").
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Single-head scaled dot-product attention.

    Attention(Q, K, V) = softmax(QK^T / √d_k) · V
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: (B, H, T, d_k)
            k: (B, H, T, d_k)
            v: (B, H, T, d_v)
            mask: (1, 1, T, T) boolean mask  (True = keep, False = mask out)

        Returns:
            out:  (B, H, T, d_v)
            attn: (B, H, T, T)  — attention weights (for visualisation)
        """
        d_k = q.size(-1)
        scale = math.sqrt(d_k)

        # (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        # Replace NaNs (from all-masked rows) with zeros
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, T, d_v)
        return out, attn


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head *causal* self-attention.

    Architecture:
      • Single combined W_qkv projection  →  split into Q, K, V
      • Reshape to (B, H, T, d_k) for each head
      • Compute scaled dot-product attention with causal mask
      • Concatenate heads, project with W_o
    """

    def __init__(
        self,
        n_embed: int,
        n_head: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        assert n_embed % n_head == 0, (
            f"n_embed ({n_embed}) must be divisible by n_head ({n_head})"
        )

        self.n_head = n_head
        self.n_embed = n_embed
        self.d_k = n_embed // n_head  # dimension per head

        # --- Projections ---
        # Fused QKV projection: one matrix instead of three → more efficient
        self.W_qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        # Output projection
        self.W_o = nn.Linear(n_embed, n_embed, bias=False)

        self.attn = ScaledDotProductAttention(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # --- Causal mask (lower-triangular) ---
        # Shape: (1, 1, max_seq_len, max_seq_len)
        # 1 = attend, 0 = masked
        causal = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", causal.view(1, 1, max_seq_len, max_seq_len))

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.W_qkv.weight, std=0.02)
        nn.init.normal_(self.W_o.weight, std=0.02 / math.sqrt(2))

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:           (B, T, n_embed)
            return_attn: if True, also return attention weights

        Returns:
            out:  (B, T, n_embed)
            attn: (B, H, T, T)  only if return_attn=True
        """
        B, T, C = x.shape

        # --- Project to Q, K, V ---
        qkv = self.W_qkv(x)                        # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embed, dim=-1)   # each (B, T, C)

        # --- Reshape for multi-head: (B, T, C) → (B, H, T, d_k) ---
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # --- Causal mask (trim to current seq length) ---
        mask = self.causal_mask[:, :, :T, :T]       # (1, 1, T, T)

        # --- Scaled dot-product attention ---
        out, attn_weights = self.attn(q, k, v, mask)  # (B, H, T, d_k)

        # --- Merge heads: (B, H, T, d_k) → (B, T, C) ---
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # --- Output projection + residual dropout ---
        out = self.resid_drop(self.W_o(out))

        if return_attn:
            return out, attn_weights
        return out
