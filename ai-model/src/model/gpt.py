"""
GoktugGPT — the full GPT-style language model (LLaMA-style modernised).

Architecture (decoder-only transformer):

  Input tokens
      │
  [Token Embedding]                         (positions via RoPE in attention)
      │
  [TransformerBlock] × n_layer              (RMSNorm + RoPE attention + SwiGLU)
      │
  [RMSNorm]
      │
  [Linear head → logits over vocabulary]    (weight-tied with token embedding)
      │
  softmax → next-token probabilities

Modern components (all implemented from scratch — see the respective files):
  • RoPE rotary positional embeddings  (attention.py)
  • RMSNorm normalisation              (transformer.py)
  • SwiGLU feed-forward                 (transformer.py)
  • Flash Attention (fused SDPA)        (attention.py)
  • KV-cache for fast generation        (this file's generate())

The objective is next-token prediction (causal / autoregressive language
modelling). Weight tying shares the output projection with the token
embedding matrix.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import Embeddings
from .transformer import TransformerDecoder, RMSNorm


class GoktugGPT(nn.Module):
    """Decoder-only GPT-style language model (LLaMA-style internals)."""

    def __init__(
        self,
        vocab_size: int,
        n_embed: int = 256,
        n_head: int = 8,
        n_layer: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.max_seq_len = max_seq_len

        self.embed = Embeddings(vocab_size, n_embed, max_seq_len, dropout)
        self.decoder = TransformerDecoder(n_embed, n_head, n_layer, dropout, max_seq_len, rope_theta)
        self.ln_f = RMSNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        # Weight tying: LM head shares the token embedding matrix.
        self.lm_head.weight = self.embed.token_embed.weight.weight

        n_params = sum(p.numel() for p in self.parameters())
        print(f"GoktugGPT initialised — {n_params/1e6:.2f}M parameters (RoPE · RMSNorm · SwiGLU)")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_all_attn: bool = False,
    ) -> dict:
        """
        Args:
            input_ids:      (B, T) token IDs.
            targets:        (B, T) for training; computes cross-entropy loss.
            start_pos:      absolute position of the first input token (RoPE/cache).
            past_kvs:       per-layer (k, v) cache from a previous step.
            use_cache:      return the updated KV-cache (for generation).
            return_all_attn: return per-layer attention weights (visualisation).
        """
        B, T = input_ids.shape
        assert start_pos + T <= self.max_seq_len, (
            f"position {start_pos + T} exceeds max_seq_len {self.max_seq_len}"
        )

        x = self.embed(input_ids)

        present_kvs = None
        all_attn = None
        if return_all_attn:
            x, all_attn = self.decoder(x, start_pos=start_pos, return_all_attn=True)
        elif use_cache:
            x, present_kvs = self.decoder(x, start_pos=start_pos, past_kvs=past_kvs, use_cache=True)
        else:
            x = self.decoder(x, start_pos=start_pos)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        result = {"logits": logits}
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )
            result["loss"] = loss
        if use_cache:
            result["past_kvs"] = present_kvs
        if return_all_attn:
            result["attn"] = all_attn
        return result

    # ------------------------------------------------------------------
    # Sampling helper
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_next(
        logits: torch.Tensor,
        context_ids: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> torch.Tensor:
        """Apply repetition penalty + temperature + top-k/top-p, then sample. logits: (1, V)."""
        # --- Repetition penalty ---
        if repetition_penalty != 1.0:
            for tid in set(context_ids[0].tolist()):
                if logits[0, tid] > 0:
                    logits[0, tid] /= repetition_penalty
                else:
                    logits[0, tid] *= repetition_penalty

        # --- Temperature ---
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-6)

        # --- Top-k ---
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            kth = torch.topk(logits, k).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        # --- Top-p (nucleus) ---
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            remove = (cum - probs) > top_p
            sorted_logits[remove] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    # ------------------------------------------------------------------
    # Text generation (KV-cached)
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
        Autoregressive generation with a KV-cache: the prompt is encoded once
        (prefill), then each new token attends to the cached keys/values instead
        of recomputing the whole sequence.
        """
        self.eval()
        stop_ids = set(stop_token_ids or [])
        if eos_token_id is not None:
            stop_ids.add(eos_token_id)

        # Crop the prompt to the context window.
        generated = input_ids[:, -self.max_seq_len:]

        # --- Prefill: encode the whole prompt, fill the cache ---
        out = self(generated, use_cache=True, start_pos=0)
        past = out["past_kvs"]
        logits = out["logits"][:, -1, :]
        cur_pos = generated.shape[1]

        for _ in range(max_new_tokens):
            next_token = self._sample_next(
                logits, generated, temperature, top_k, top_p, repetition_penalty
            )
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() in stop_ids:
                break
            if cur_pos >= self.max_seq_len:
                break  # context window full (sliding-window cache is Phase 3)

            # --- Decode step: feed ONLY the new token, reuse the cache ---
            out = self(next_token, use_cache=True, past_kvs=past, start_pos=cur_pos)
            past = out["past_kvs"]
            logits = out["logits"][:, -1, :]
            cur_pos += 1

        return generated

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def enable_gradient_checkpointing(self):
        self.decoder.use_gradient_checkpointing = True

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save_checkpoint(self, path: str, extra: Optional[dict] = None):
        payload = {"model_state": self.state_dict()}
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"Checkpoint saved -> {path}")

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
        print(f"Checkpoint loaded <- {path}")
        return model, extra
