"""
Multi-Head Causal Self-Attention — built from scratch, LLaMA-style.

Modernisations over the original GPT-2-style attention:
  • RoPE (Rotary Positional Embeddings) — position is injected by *rotating*
    Q and K, instead of adding a learned positional vector to the tokens.
    Better length generalisation; used by LLaMA, Mistral, Qwen, etc.
  • Flash Attention — uses torch's fused `F.scaled_dot_product_attention`
    (memory-efficient, faster). A manual softmax path is kept for attention
    visualisation.
  • KV-cache — during generation, past keys/values are cached so each new
    token only computes attention against the cache instead of recomputing
    the whole sequence (huge speedup for long outputs).

Theory recap:
    Q = X·W_Q,  K = X·W_K,  V = X·W_V
    Attention(Q,K,V) = softmax(QKᵀ / √d_k)·V          (with causal mask)
RoPE rotates Q,K by an angle proportional to their absolute position so that
the dot product QKᵀ depends only on the *relative* position (i − j).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE)
# ----------------------------------------------------------------------

def build_rope_cache(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    """
    Precompute the cos/sin tables for RoPE.

    Returns two (max_seq_len, head_dim) tensors. We duplicate the half-dim
    frequencies so they line up with the `rotate_half` convention below.
    """
    # Frequencies for each pair of dimensions: (head_dim/2,)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()           # (max_seq_len,)
    freqs = torch.outer(t, inv_freq)                # (max_seq_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)         # (max_seq_len, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dim: [x1, x2] -> [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to q and k.

    q, k:    (B, H, T, head_dim)
    cos,sin: (T, head_dim)  — already sliced to the absolute positions of q/k.
    """
    cos = cos[None, None, :, :]   # (1, 1, T, head_dim)
    sin = sin[None, None, :, :]
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE, Flash Attention and KV-cache.
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
        assert n_embed % n_head == 0, (
            f"n_embed ({n_embed}) must be divisible by n_head ({n_head})"
        )

        self.n_head = n_head
        self.n_embed = n_embed
        self.d_k = n_embed // n_head
        self.dropout_p = dropout

        # Fused QKV projection + output projection (no biases, LLaMA-style)
        self.W_qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.W_o = nn.Linear(n_embed, n_embed, bias=False)
        self.resid_drop = nn.Dropout(dropout)

        # RoPE cos/sin tables (buffers — moved to device with the model)
        cos, sin = build_rope_cache(self.d_k, max_seq_len, rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.W_qkv.weight, std=0.02)
        nn.init.normal_(self.W_o.weight, std=0.02 / math.sqrt(2))

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        return_attn: bool = False,
    ):
        """
        Args:
            x:          (B, T, n_embed)
            start_pos:  absolute position of the first token in x (for RoPE +
                        cache). 0 during training / prefill.
            past_kv:    optional (past_k, past_v) each (B, H, T_past, d_k).
            use_cache:  if True, also return the updated (k, v) cache.
            return_attn: if True, return attention weights (manual path, no Flash).

        Returns:
            out (B, T, n_embed), plus optionally present_kv and/or attn weights.
        """
        B, T, C = x.shape

        qkv = self.W_qkv(x)                              # (B, T, 3C)
        q, k, v = qkv.split(self.n_embed, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)   # (B, H, T, d_k)

        # --- RoPE at absolute positions [start_pos, start_pos + T) ---
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]
        q, k = apply_rope(q, k, cos, sin)

        # --- Prepend cached K/V (incremental decoding) ---
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present_kv = (k, v) if use_cache else None

        if return_attn:
            # Manual path (for attention visualisation only)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            T_k = k.size(2)
            # causal mask: query i (abs start_pos+i) may attend key j (abs j) iff j <= start_pos+i
            qpos = torch.arange(T, device=x.device).view(T, 1) + start_pos
            kpos = torch.arange(T_k, device=x.device).view(1, T_k)
            mask = (kpos <= qpos)                        # (T, T_k)
            scores = scores.masked_fill(~mask, float("-inf"))
            attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
            out = torch.matmul(attn, v)
        else:
            # Flash Attention (fused). Causal only needed for the full-seq /
            # prefill case; single-token decode attends all cached keys.
            is_causal = past_kv is None and T > 1
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal,
            )
            attn = None

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.W_o(out))

        # Build return tuple
        if not use_cache and not return_attn:
            return out
        result = (out,)
        if use_cache:
            result += (present_kv,)
        if return_attn:
            result += (attn,)
        return result
