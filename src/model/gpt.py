"""
GoktugGPT — the full GPT-style language model.

Architecture (GPT-2 style decoder-only transformer):

  Input tokens
      │
  [Token Embedding + Positional Encoding]
      │
  [TransformerBlock] × n_layer
      │
  [LayerNorm]
      │
  [Linear head → logits over vocabulary]
      │
  softmax → next token probabilities

The language-modelling objective is to predict the next token at each
position given all previous tokens (causal / autoregressive).

Weight tying:
  The output projection (head) shares weights with the token embedding
  matrix.  This is a well-known trick that reduces parameters and often
  improves performance.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import Embeddings
from .transformer import TransformerDecoder


class GoktugGPT(nn.Module):
    """
    Decoder-only GPT-style language model.

    Args:
        vocab_size:  Size of the token vocabulary.
        n_embed:     Embedding / hidden dimension (d_model).
        n_head:      Number of attention heads per block.
        n_layer:     Number of stacked TransformerBlock layers.
        dropout:     Dropout probability applied throughout.
        max_seq_len: Maximum sequence length the model can handle.
    """

    def __init__(
        self,
        vocab_size: int,
        n_embed: int = 256,
        n_head: int = 8,
        n_layer: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.max_seq_len = max_seq_len

        # --- Embedding layer ---
        self.embed = Embeddings(vocab_size, n_embed, max_seq_len, dropout)

        # --- Transformer backbone ---
        self.decoder = TransformerDecoder(n_embed, n_head, n_layer, dropout, max_seq_len)

        # --- Final layer norm (Pre-LN style) ---
        self.ln_f = nn.LayerNorm(n_embed)

        # --- LM head: projects hidden states → logits ---
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        # Weight tying: share token embedding weights with the LM head
        self.lm_head.weight = self.embed.token_embed.weight.weight

        # Initialise weights
        self.apply(self._init_weights)

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GoktugGPT initialised — {n_params/1e6:.2f}M parameters")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_all_attn: bool = False,
    ) -> dict:
        """
        Args:
            input_ids:      (B, T) long tensor of token IDs.
            targets:        (B, T) long tensor; if provided, computes CE loss.
            return_all_attn: if True, returns attention weights for all layers.

        Returns:
            dict with keys:
              'logits'  : (B, T, vocab_size)
              'loss'    : scalar (only if targets provided)
              'attn'    : list of (B, H, T, T) tensors (only if requested)
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"
        )

        # Embed tokens + positions
        x = self.embed(input_ids)  # (B, T, E)

        # Pass through transformer
        if return_all_attn:
            x, all_attn = self.decoder(x, return_all_attn=True)
        else:
            x = self.decoder(x)  # (B, T, E)
            all_attn = None

        # Final norm + project to vocabulary
        x = self.ln_f(x)          # (B, T, E)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        result = {"logits": logits}

        # --- Compute loss if targets provided ---
        if targets is not None:
            # Shift so we predict position i+1 from position i
            # logits: (B, T, V)  targets: (B, T)
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,  # -1 means "don't compute loss here"
            )
            result["loss"] = loss

        if return_all_attn:
            result["attn"] = all_attn

        return result

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_k: int = 30,
        top_p: float = 0.9,
        repetition_penalty: float = 1.3,
        eos_token_id: Optional[int] = None,
        stop_token_ids: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation with temperature, top-k, top-p
        (nucleus) sampling, and repetition penalty.

        Args:
            input_ids:          (1, T) prompt token IDs.
            max_new_tokens:     maximum tokens to generate.
            temperature:        >1 = more random, <1 = more greedy.
            top_k:              keep only top-k tokens before sampling.
            top_p:              nucleus sampling threshold.
            repetition_penalty: penalise tokens already in the context (>1 = more penalty).
            eos_token_id:       stop generating when this token is produced.
            stop_token_ids:     list of additional stop tokens.

        Returns:
            (1, T + new_tokens) tensor.
        """
        self.eval()
        stop_ids = set(stop_token_ids or [])
        if eos_token_id is not None:
            stop_ids.add(eos_token_id)

        for _ in range(max_new_tokens):
            # Crop context to max_seq_len
            ctx = input_ids[:, -self.max_seq_len:]

            out = self(ctx)
            logits = out["logits"][:, -1, :]  # (1, vocab_size) — last position

            # --- Repetition penalty ---
            # Divide logits for tokens that already appear in the context.
            # Positive logits are divided (made less likely);
            # negative logits are multiplied (pushed further negative).
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            # --- Temperature scaling ---
            if temperature != 1.0:
                logits = logits / temperature

            # --- Top-k filtering ---
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                top_k_vals = torch.topk(logits, k).values[:, -1:]
                logits = logits.masked_fill(logits < top_k_vals, float("-inf"))

            # --- Top-p (nucleus) filtering ---
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens beyond nucleus
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                # Unsort
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            # --- Sample next token ---
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() in stop_ids:
                break

        return input_ids

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save_checkpoint(self, path: str, extra: Optional[dict] = None):
        payload = {"model_state": self.state_dict()}
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"Checkpoint saved → {path}")

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        vocab_size: int,
        n_embed: int,
        n_head: int,
        n_layer: int,
        dropout: float = 0.0,
        max_seq_len: int = 512,
        device: str = "cpu",
    ) -> tuple["GoktugGPT", dict]:
        payload = torch.load(path, map_location=device)
        model = cls(vocab_size, n_embed, n_head, n_layer, dropout, max_seq_len)
        model.load_state_dict(payload["model_state"])
        model.to(device)
        extra = {k: v for k, v in payload.items() if k != "model_state"}
        print(f"Checkpoint loaded ← {path}")
        return model, extra
