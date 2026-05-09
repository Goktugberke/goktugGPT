# goktugGPT

A language model built **entirely from scratch** — no OpenAI, no Hugging Face, no pretrained weights.
Every component (tokenizer, attention, transformer, training loop, GUI) is implemented in pure Python and PyTorch.

---

## Table of Contents

- [What is goktugGPT?](#what-is-goktugGPT)
- [Capabilities](#capabilities)
- [Architecture](#architecture)
  - [BPE Tokenizer](#1-bpe-tokenizer)
  - [Word Embeddings](#2-word-embeddings)
  - [Multi-Head Self-Attention](#3-multi-head-self-attention)
  - [Transformer Decoder Blocks](#4-transformer-decoder-blocks)
  - [Full GPT Model](#5-full-gpt-model)
  - [Embedded Thinking Stage](#6-embedded-thinking-stage)
- [Training Data](#training-data)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Training the Model](#training-the-model)
- [Running the Chat](#running-the-chat)
- [Running the GUI](#running-the-gui)
- [Configuration](#configuration)
- [How Generation Works](#how-generation-works)
- [Training on Google Colab](#training-on-google-colab)

---

## What is goktugGPT?

goktugGPT is a **local, fully offline** GPT-style language model. It does not call any API, does not require an internet connection, and does not use any pretrained model weights. Everything — from how it splits words into tokens to how it generates a response — is built from scratch and runs entirely on your own machine.

The goal of this project is to demonstrate how a modern language model actually works at every layer:
tokenization → embeddings → attention → transformer → generation → reasoning → UI.

---

## Capabilities

| Capability | Status |
|---|---|
| General question answering | ✅ |
| Conversational multi-turn chat | ✅ |
| Embedded chain-of-thought reasoning (`<think>` stage) | ✅ |
| Math and logic reasoning | ✅ (scales with training data) |
| Factual recall (from training data) | ✅ |
| Web chat GUI (Gradio) | ✅ |
| Terminal chat interface | ✅ |
| Fully offline / no API | ✅ |
| GPU acceleration (CUDA / Apple MPS) | ✅ |
| Checkpoint saving and resuming | ✅ |
| Configurable model size (tiny → large) | ✅ |
| Mixed precision training (AMP fp16) | ✅ |
| Gradient checkpointing (low VRAM) | ✅ |
| Google Colab training support | ✅ |

> **Note:** Quality of responses is directly tied to the size of your training data and how long you train. `TinyConfig` is for fast CPU experiments — for real results, use `MediumConfig` or `LargeConfig` on a GPU with the full dataset (94K+ lines).

---

## Architecture

goktugGPT uses a **decoder-only transformer** — the same family of architecture as GPT-2, GPT-3, LLaMA, and most modern chat models. Every component is implemented from scratch.

```
Input Text
    │
    ▼
┌─────────────────────────────────┐
│         BPE Tokenizer           │  text → list of integer token IDs
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Token Embedding × √(d_model)   │  IDs → dense vectors (word vectors)
│  + Learned Positional Encoding  │  + position information
└─────────────────────────────────┘
    │
    ▼  (repeated N times)
┌─────────────────────────────────┐
│      Transformer Block          │
│                                 │
│  ┌─ LayerNorm                   │
│  └─ Multi-Head Self-Attention ──┤  each token attends to all previous
│     (causal mask)               │
│  ┌─ Residual connection         │
│  ├─ LayerNorm                   │
│  └─ Feed-Forward Network ───────┤  GELU activation, 4× expansion
│     Residual connection         │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Final LayerNorm                │
│  LM Head (Linear, weight-tied)  │  hidden → logits over vocab
└─────────────────────────────────┘
    │
    ▼
  Logits → top-k/top-p sampling → next token
```

### 1. BPE Tokenizer

**File:** `src/tokenizer/bpe_tokenizer.py`

Byte-Pair Encoding (BPE) is the tokenization algorithm used by GPT-2 and most modern LLMs. It is trained from scratch on the same text data as the model.

**How it works:**
1. Start by splitting every word into individual characters plus a special end-of-word marker `</w>`.
2. Count every adjacent pair of symbols in the entire training corpus.
3. Merge the most frequent pair into a single new symbol.
4. Repeat until the vocabulary reaches the target size (`vocab_size`).
5. To encode new text, apply the learned merge rules in order — rare words get split into known subword pieces.

**Why BPE?**
- Handles unknown words by decomposing them into known subpieces.
- Balances vocabulary size vs. sequence length better than pure word or character tokenization.
- No external dependency — trained entirely on your data.

### 2. Word Embeddings

**File:** `src/model/embeddings.py`

Two components are combined:

**Token Embedding:** A learned matrix of shape `(vocab_size, d_model)`. Each row is the vector representation of one token. These vectors are what the model learns — similar tokens end up with similar vectors.

**Positional Encoding:** Since transformers process all tokens simultaneously (not sequentially like RNNs), they need explicit position information. goktugGPT uses **learned positional embeddings** (GPT-2 style) — a second embedding table indexed by position (0, 1, 2, ...) rather than token ID. The token vectors and position vectors are summed to give the final embedding.

The embeddings are scaled by `√(d_model)` before adding position information, following the original Transformer paper.

### 3. Multi-Head Self-Attention

**File:** `src/model/attention.py`

Attention is the core mechanism that lets every token "look at" other tokens to build context.

**Scaled Dot-Product Attention:**

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```

Where:
- `Q` (Queries) — what this token is looking for
- `K` (Keys) — what each token offers to be matched against
- `V` (Values) — what each token contributes when matched

**Multi-Head:**
Instead of one attention computation, goktugGPT runs `n_head` parallel attention heads, each operating on a `d_model / n_head`-dimensional subspace. This lets the model attend to different aspects of the sequence simultaneously. The outputs are concatenated and projected.

**Causal Mask:**
A lower-triangular mask ensures that position `i` can only attend to positions `0..i`. This is essential for autoregressive (left-to-right) text generation — the model cannot "see the future."

**Implementation note:** The Q, K, V projections are fused into a single `W_qkv` linear layer for efficiency. The output projection `W_o` has a scaled initialisation (`std = 0.02 / √2`) to prevent exploding activations in deep networks.

### 4. Transformer Decoder Blocks

**File:** `src/model/transformer.py`

Each block has two sub-layers:

```
x ← x + Attention( LayerNorm(x) )    ← self-attention sub-layer
x ← x + FFN( LayerNorm(x) )          ← feed-forward sub-layer
```

**Pre-Layer Normalization (Pre-LN):** LayerNorm is applied *before* each sub-layer (not after, as in the original paper). This is the GPT-2 convention and makes training significantly more stable.

**Feed-Forward Network (FFN):**

```
FFN(x) = W₂ · GELU( W₁ · x )
```

The hidden dimension is `4 × d_model`. GELU (Gaussian Error Linear Unit) is used instead of ReLU — it is smoother and consistently outperforms ReLU on language tasks.

**Residual Connections:** Each sub-layer adds its output back to its input. This allows gradients to flow through many layers without vanishing.

### 5. Full GPT Model

**File:** `src/model/gpt.py`

The `GoktugGPT` class assembles all components:

- Embeddings → N Transformer Blocks → LayerNorm → LM Head
- **Weight Tying:** The LM Head shares its weight matrix with the Token Embedding. This reduces parameters and is a well-known technique that improves performance.
- **Loss:** Cross-entropy over next-token prediction. Padding positions (target = -1) are ignored.

**Text Generation** uses:
| Technique | Effect |
|---|---|
| **Temperature** | Scale logits before softmax. `< 1.0` = more focused, `> 1.0` = more creative |
| **Top-K** | Keep only the K highest probability tokens before sampling |
| **Top-P (Nucleus)** | Keep the smallest set of tokens whose cumulative probability exceeds P |

### 6. Embedded Thinking Stage

**File:** `src/thinking/chain_of_thought.py`

Inspired by DeepSeek-R1 and OpenAI's o1, goktugGPT has an embedded reasoning stage.

**How it works:**

When the model sees a user message, it does not jump straight to an answer. Instead:

1. The model is prompted to generate a `<think>` token.
2. It freely generates internal reasoning — a scratchpad — until it produces a `</think>` token.
3. Only then does it generate the final visible answer.

```
<user> What is 15 × 23?
<assistant> <think>
  I need to multiply 15 by 23.
  15 × 20 = 300, 15 × 3 = 45.
  300 + 45 = 345.
</think>
15 × 23 equals 345.
<eos>
```

**Why this works:**
The model is trained on examples that include `<think>...</think>` segments. It learns to associate certain types of questions with generating reasoning first. The thinking tokens give the model extra "compute budget" before committing to a final answer — the same principle behind chain-of-thought prompting.

The thinking output is displayed in a collapsible panel in the GUI, or shown in a distinct color in the terminal.

---

## Training Data

**Script:** `data/download_conversations.py`
**Output:** `data/train_chat.txt`

The training data is built automatically from multiple sources using `download_conversations.py`:

| Source | Lines | Description |
|--------|-------|-------------|
| Synthetic Q&A | ~12,600 | Math (1889), capitals (348), animals (120), definitions (60), greetings, science, history |
| Cornell Movie Dialogs | ~80,000 | Real movie conversations (~220K pairs available) |
| DailyDialog | ~80,000 | Daily life conversations |
| **Total** | **~94,000+** | **~4.1M tokens, ~18MB** |

**Generate the dataset:**
```bash
python data/download_conversations.py              # Full dataset (Cornell + DailyDialog + synthetic)
python data/download_conversations.py --synthetic-only  # Only synthetic pairs (~12K lines, fast)
```

**Format:**
```
<user> [question] <assistant> <think> [reasoning] </think> [answer] <eos>
```

Every example includes a thinking stage, teaching the model to reason before answering.

**Extending the training data:**
Add more lines to `data/train_chat.txt` in the same format. More data = better, more knowledgeable responses. You can also add new sources to `download_conversations.py`.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.10+** | Primary language |
| **PyTorch** | Neural network framework (tensors, autograd, GPU support) |
| **Gradio** | Web-based chat GUI |
| **tqdm** | Training progress bars |
| **NumPy** | Numerical utilities |
| **matplotlib** | Loss curve plotting (optional) |

**Everything else is built from scratch:**
- BPE tokenizer: `tokenizers` library for fast training, custom implementation for encoding
- Transformer: pure PyTorch `nn.Module`, no `transformers` library
- Attention: manual Q/K/V projections, not `nn.MultiheadAttention`
- Training loop: manual gradient updates, not `Trainer` from Hugging Face

---

## Project Structure

```
goktugGPT/
│
├── train.py                    # Entry point: train the model
├── chat.py                     # Entry point: terminal chat
├── gui.py                      # Entry point: launch web GUI
├── config.py                   # Model size configurations
├── requirements.txt
│
├── data/
│   ├── train_chat.txt          # Generated training data (94K+ lines)
│   └── download_conversations.py  # Dataset builder (Cornell, DailyDialog, synthetic)
│
├── checkpoints/                # Saved model weights (created after training)
│   ├── tokenizer.json
│   ├── best_model.pt
│   └── final_model.pt
│
└── src/
    ├── tokenizer/
    │   └── bpe_tokenizer.py    # BPE tokenizer from scratch
    │
    ├── model/
    │   ├── embeddings.py       # Token + positional embeddings
    │   ├── attention.py        # Multi-head causal self-attention
    │   ├── transformer.py      # TransformerBlock + TransformerDecoder
    │   └── gpt.py              # GoktugGPT: full model + generation
    │
    ├── thinking/
    │   └── chain_of_thought.py # ThinkingEngine: think-then-answer loop
    │
    ├── training/
    │   ├── dataset.py          # ConversationDataset + DataLoader
    │   └── trainer.py          # Training loop, LR schedule, checkpointing
    │
    └── gui/
        └── app.py              # Gradio chat interface
```

---

## Setup & Installation

**Requirements:** Python 3.10 or newer.

```bash
# 1. Clone / navigate to the project
cd goktugGPT

# 2. (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

**GPU support (optional but recommended for larger configs):**

```bash
# For NVIDIA GPU (CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For Apple Silicon (MPS)
# PyTorch MPS backend is included in standard torch >= 2.0
```

---

## Training the Model

```bash
# 1. First, generate the dataset
python data/download_conversations.py

# 2. Train (pick a config based on your hardware)
python train.py --config tiny                        # CPU, fast experiment
python train.py --config default                     # Small GPU (4GB VRAM)
python train.py --config medium --batch-size 2       # GTX 1650 (4GB VRAM)
python train.py --config large --batch-size 4        # T4/RTX 3060+ (8GB+ VRAM)

# Override specific hyperparameters
python train.py --epochs 100 --batch-size 4 --lr 3e-4

# Use a custom training data file
python train.py --data path/to/your/data.txt

# Resume from the last saved checkpoint
python train.py --resume

# Save checkpoints to a custom directory (useful for Google Drive)
python train.py --checkpoint-dir /content/drive/MyDrive/goktugGPT/checkpoints

# Only train the tokenizer (skip model training)
python train.py --tokenizer-only
```

**What happens during training:**

1. The BPE tokenizer is trained on the dataset and saved to `checkpoints/tokenizer.json`.
2. Training and validation datasets are built from the tokenized text.
3. The GoktugGPT model is initialized with random weights.
4. **AMP mixed precision (fp16)** is automatically enabled on GPU — halves VRAM usage and speeds up training.
5. **Gradient checkpointing** is enabled on GPU — trades ~20% speed for ~40% less VRAM.
6. The training loop runs for `max_epochs` epochs:
   - Each batch: forward pass → compute cross-entropy loss → backprop → weight update
   - Learning rate warms up linearly then follows a cosine decay schedule
   - Every `eval_interval` steps the model is evaluated on the validation set
   - The best checkpoint (lowest validation loss) is saved to `checkpoints/best_model.pt`
7. Training finishes and `checkpoints/final_model.pt` is saved.

**Expected training time (94K lines dataset):**

| Config | Parameters | Hardware | Approx. Time |
|---|---|---|---|
| TinyConfig | ~1.3M | CPU | ~15–30 min |
| ModelConfig | ~8.5M | CPU / any GPU | ~1–3 hours |
| MediumConfig | ~42M | GTX 1650 (4GB) | ~8–15 hours |
| LargeConfig | ~110M | T4 / RTX 3060+ (8GB+) | ~4–8 hours |

---

## Running the Chat

### Terminal interface

```bash
python chat.py

# Use a specific config and checkpoint
python chat.py --config large --checkpoint checkpoints/best_model.pt

# Disable the thinking stage
python chat.py --no-thinking

# Use a more creative temperature
python chat.py --temperature 1.2
```

**Terminal commands during chat:**

| Command | Action |
|---|---|
| `/quit` or `/exit` | Exit the program |
| `/clear` | Clear conversation history |
| `/think` | Toggle the thinking stage on/off |
| `/info` | Show model info (parameters, device, etc.) |

### Web GUI

```bash
python gui.py

# Create a public shareable link (via Gradio)
python gui.py --share

# Use a different port
python gui.py --port 8080
```

The GUI opens automatically at `http://localhost:7860` and provides:
- A chat interface with message history
- A collapsible **Thinking Stage** panel showing the model's reasoning
- Sliders to adjust **Temperature**, **Top-K**, **Top-P**, and token limits
- A toggle to enable/disable the thinking stage
- Model info panel (parameters, device, vocab size)

---

## Configuration

Four preset configurations are available in `config.py`:

| Setting | TinyConfig | ModelConfig | MediumConfig | LargeConfig |
|---|---|---|---|---|
| `vocab_size` | 4,000 | 8,000 | 16,000 | 32,000 |
| `n_embed` (d_model) | 128 | 256 | 512 | 768 |
| `n_head` | 4 | 8 | 8 | 12 |
| `n_layer` | 3 | 6 | 8 | 12 |
| `max_seq_len` | 256 | 512 | 1024 | 1024 |
| `batch_size` | 4 | 8 | 16 | 16 |
| `max_epochs` | 30 | 50 | 50 | 30 |
| Approx. parameters | **~1.3M** | **~8.5M** | **~42M** | **~110M** |
| Min. VRAM | CPU OK | CPU OK | ~3.5 GB | ~8 GB |

**Which config should I use?**

| Your Hardware | Recommended Config |
|---|---|
| CPU only (no GPU) | `tiny` or `default` |
| GTX 1650 / 4GB VRAM | `medium` with `--batch-size 2` |
| T4 (Colab free) / RTX 3060 | `large` with `--batch-size 4` |
| RTX 3090 / A100 | `large` with `--batch-size 16` |

To customize further, edit `config.py` directly or subclass `ModelConfig`.

---

## How Generation Works

At inference time, generation is autoregressive: the model predicts one token at a time, appends it to the sequence, and then predicts the next token from that extended sequence.

```
prompt → [token 1, token 2, ..., token N]
                                         ↓
                               model predicts token N+1
                                         ↓
                    [token 1, ..., token N, token N+1]
                                                      ↓
                                            predicts token N+2
                                                      ...
```

**Sampling strategy (controls creativity vs. precision):**

1. **Temperature:** Divide all logits by `T`. Low `T` makes the highest-probability token much more likely (focused). High `T` flattens the distribution (more random).
2. **Top-K:** Zero out all but the `K` most probable tokens before sampling.
3. **Top-P (Nucleus):** Sort tokens by probability. Keep only the smallest set of tokens whose cumulative probability exceeds `P`. This adapts dynamically — it keeps fewer tokens when the model is confident and more when it is uncertain.

Generation stops when an `<eos>` token is produced or the maximum token limit is reached.

---

## Training on Google Colab

You can train goktugGPT for free on Google Colab using a T4 GPU (14.5 GB VRAM).

### First-time setup

1. Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook
2. Set runtime: `Runtime → Change runtime type → T4 GPU → Save`

```python
# Cell 1 — Mount Google Drive (persistent storage)
from google.colab import drive
drive.mount('/content/drive')
import os
os.makedirs('/content/drive/MyDrive/goktugGPT/checkpoints', exist_ok=True)
```

```python
# Cell 2 — Clone repo
!git clone https://github.com/YOUR_USERNAME/goktugGPT.git /content/goktugGPT
%cd /content/goktugGPT
!pip install tokenizers tqdm -q
```

```python
# Cell 3 — Generate dataset
!python data/download_conversations.py
# Backup to Drive so you don't have to regenerate
!cp data/train_chat.txt /content/drive/MyDrive/goktugGPT/train_chat.txt
```

```python
# Cell 4 — Train
!python train.py \
    --config large \
    --data data/train_chat.txt \
    --epochs 20 \
    --batch-size 4 \
    --checkpoint-dir /content/drive/MyDrive/goktugGPT/checkpoints
```

### Resuming after Colab disconnects

When Colab resets your runtime, just re-run:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/goktugGPT
!cp /content/drive/MyDrive/goktugGPT/train_chat.txt data/train_chat.txt

!python train.py \
    --config large \
    --data data/train_chat.txt \
    --epochs 20 \
    --batch-size 4 \
    --checkpoint-dir /content/drive/MyDrive/goktugGPT/checkpoints \
    --resume
```

Checkpoints are saved to Google Drive, so progress is never lost.

### Download trained model to local

After training, download from Google Drive:
- Go to `drive.google.com` → `MyDrive/goktugGPT/checkpoints/`
- Download `best_model.pt` and `tokenizer.json`
- Place them in your local `goktugGPT/checkpoints/` folder

---

## Academic References

The following papers directly inspired the architecture of goktugGPT:

- **Attention Is All You Need** — Vaswani et al. (2017) — Original transformer architecture, scaled dot-product attention, multi-head attention, positional encoding
- **Language Models are Unsupervised Multitask Learners** — Radford et al. (2019) — GPT-2: decoder-only transformer, pre-LN, weight tying, large-scale language modelling
- **Neural Machine Translation of Rare Words with Subword Units** — Sennrich et al. (2016) — BPE tokenization algorithm
- **Gaussian Error Linear Units (GELUs)** — Hendrycks & Gimpel (2016) — GELU activation function
- **DeepSeek-R1** — DeepSeek AI (2025) — Embedded chain-of-thought thinking stage with `<think>` tokens

---

*goktugGPT — A language model built from scratch, for learning and experimentation.*
