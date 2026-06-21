"""
Quick smoke test for the modernised architecture (RoPE + RMSNorm + SwiGLU +
Flash Attention + KV-cache). Verifies forward/loss/generate run with correct
shapes, and that KV-cached generation is consistent with the no-cache path.
Run:  python verify_architecture.py
"""
import torch
from src.model import GoktugGPT

torch.manual_seed(0)

V, B, T = 256, 2, 16
model = GoktugGPT(vocab_size=V, n_embed=128, n_head=4, n_layer=3, dropout=0.0, max_seq_len=64)
model.eval()

# --- 1. Forward + loss ---
ids = torch.randint(0, V, (B, T))
tgt = torch.randint(0, V, (B, T))
out = model(ids, targets=tgt)
assert out["logits"].shape == (B, T, V), out["logits"].shape
assert out["loss"].dim() == 0 and torch.isfinite(out["loss"]), out["loss"]
print(f"[OK] forward: logits {tuple(out['logits'].shape)}  loss {out['loss'].item():.3f}")

# --- 2. return_all_attn path (manual attention) ---
out_a = model(ids, return_all_attn=True)
assert len(out_a["attn"]) == 3 and out_a["attn"][0].shape == (B, 4, T, T)
print(f"[OK] attn path: {len(out_a['attn'])} layers, attn {tuple(out_a['attn'][0].shape)}")

# --- 3. KV-cache correctness: cached prefill+step == full forward ---
prompt = torch.randint(0, V, (1, 8))
full = model(prompt)["logits"][:, -1, :]              # logits for next token, no cache
pre = model(prompt, use_cache=True, start_pos=0)      # prefill with cache
cached_last = pre["logits"][:, -1, :]
assert torch.allclose(full, cached_last, atol=1e-4), (full - cached_last).abs().max()
# one decode step with a fixed next token, compare against full recompute
nxt = torch.tensor([[42]])
step = model(nxt, use_cache=True, past_kvs=pre["past_kvs"], start_pos=8)["logits"][:, -1, :]
recompute = model(torch.cat([prompt, nxt], dim=1))["logits"][:, -1, :]
assert torch.allclose(step, recompute, atol=1e-4), (step - recompute).abs().max()
print("[OK] KV-cache matches full recompute (prefill + decode step)")

# --- 4. generate runs end to end ---
gen = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=20, top_p=0.9,
                     repetition_penalty=1.3, eos_token_id=None)
assert gen.shape[1] == prompt.shape[1] + 20, gen.shape
print(f"[OK] generate: {prompt.shape[1]} -> {gen.shape[1]} tokens")

print(f"\nParams: {model.num_parameters()/1e6:.2f}M")
print("ALL CHECKS PASSED")
