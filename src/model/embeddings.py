"""
Word Embeddings + Positional Encoding — built from scratch.

Two positional strategies are implemented:
  - Learned positional embeddings  (default, like GPT-2)
  - Sinusoidal positional encoding (like the original "Attention is All You Need")

The final Embeddings module combines both and applies dropout.
"""

import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Lookup table: token id → dense vector.

    This is essentially a learned word-vector matrix of shape
    (vocab_size, n_embed).  Each row is the embedding for one token.
    """

    def __init__(self, vocab_size: int, n_embed: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        # The actual parameter matrix
        self.weight = nn.Embedding(vocab_size, n_embed)
        # Scale embeddings (standard practice)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, T)  →  output: (B, T, n_embed)
        return self.weight(token_ids)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al. 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    This is *not* learned — it injects position information via a
    deterministic pattern.  Advantage: can extrapolate to unseen lengths.
    """

    def __init__(self, n_embed: int, max_seq_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build the (max_seq_len, n_embed) encoding table once
        pe = torch.zeros(max_seq_len, n_embed)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(
            torch.arange(0, n_embed, 2, dtype=torch.float)
            * (-math.log(10000.0) / n_embed)
        )  # (n_embed/2,)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, T, n_embed)
        # Register as buffer so it moves to GPU with .to(device) but
        # is NOT updated by the optimizer
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_embed)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embedding table (GPT-2 style).

    Each position 0..max_seq_len-1 gets its own trainable vector.
    More expressive than sinusoidal but cannot extrapolate beyond
    max_seq_len.
    """

    def __init__(self, n_embed: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos_embed = nn.Embedding(max_seq_len, n_embed)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_embed)
        T = x.size(1)
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        return self.dropout(x + self.pos_embed(positions))


class Embeddings(nn.Module):
    """
    Full embedding layer: token embedding + positional encoding.

    Combines TokenEmbedding + LearnedPositionalEncoding (GPT-2 style).
    The scale factor sqrt(n_embed) comes from the original Transformer
    paper to balance the magnitudes of the two contributions.
    """

    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        use_learned_pos: bool = True,
    ):
        super().__init__()
        self.n_embed = n_embed
        self.token_embed = TokenEmbedding(vocab_size, n_embed)

        if use_learned_pos:
            self.pos_enc = LearnedPositionalEncoding(n_embed, max_seq_len, dropout)
        else:
            self.pos_enc = PositionalEncoding(n_embed, max_seq_len, dropout)

        self.scale = math.sqrt(n_embed)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, T)
        tok_emb = self.token_embed(token_ids) * self.scale  # (B, T, E)
        return self.pos_enc(tok_emb)                         # (B, T, E)
