"""
Token Embeddings — built from scratch.

In the modern (LLaMA-style) architecture, position information is NOT added
here. It is injected later, inside the attention layer, via RoPE (Rotary
Positional Embeddings — see attention.py). So this module is just a learned
token lookup table plus dropout.

The token embedding matrix is weight-tied with the LM head (see gpt.py).
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Lookup table: token id → dense vector. Shape (vocab_size, n_embed).
    Each row is the learned embedding for one token.
    """

    def __init__(self, vocab_size: int, n_embed: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.weight = nn.Embedding(vocab_size, n_embed)
        nn.init.normal_(self.weight.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # (B, T) -> (B, T, n_embed)
        return self.weight(token_ids)


class Embeddings(nn.Module):
    """
    Token embedding + dropout. Positions are handled by RoPE in attention,
    so there is no positional table here (LLaMA-style).
    """

    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        max_seq_len: int = 512,   # kept for signature compatibility
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_embed = n_embed
        self.token_embed = TokenEmbedding(vocab_size, n_embed)
        self.drop = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.drop(self.token_embed(token_ids))
