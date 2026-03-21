"""
chat.py — Terminal chat interface for goktugGPT

Usage:
    python chat.py                          # Default settings
    python chat.py --config medium          # Use medium config
    python chat.py --no-thinking            # Disable thinking stage
    python chat.py --temperature 1.0        # More creative responses
    python chat.py --checkpoint checkpoints/best_model.pt

The terminal interface shows:
  • The model's thinking process (collapsible)
  • The model's response
  • Conversation history is maintained for the session
"""

import argparse
import os
import sys

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with goktugGPT in the terminal")
    parser.add_argument("--config", choices=["tiny", "default", "medium", "large"], default="tiny")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--no-thinking", action="store_true", help="Disable thinking stage")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--max-think", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


BANNER = """
╔════════════════════════════════════════════════╗
║              goktugGPT Terminal Chat           ║
║   Local LLM · Transformer · Thinking Stage    ║
╠════════════════════════════════════════════════╣
║  Commands:                                     ║
║    /quit  or  /exit   — Exit                   ║
║    /clear             — Clear conversation     ║
║    /think             — Toggle thinking stage  ║
║    /info              — Show model info        ║
╚════════════════════════════════════════════════╝
"""


def _color(text: str, code: str) -> str:
    """Apply ANSI color code if the terminal supports it."""
    return f"\033[{code}m{text}\033[0m"


def _find_checkpoint(checkpoint_dir: str):
    if not os.path.isdir(checkpoint_dir):
        return None
    for name in ["best_model.pt", "final_model.pt"]:
        p = os.path.join(checkpoint_dir, name)
        if os.path.exists(p):
            return p
    pts = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")], reverse=True
    )
    return os.path.join(checkpoint_dir, pts[0]) if pts else None


def main():
    args = parse_args()

    from config import ModelConfig, TinyConfig, MediumConfig, LargeConfig
    from src.tokenizer import BPETokenizer
    from src.model import GoktugGPT
    from src.thinking import ThinkingEngine

    if args.config == "tiny":
        config = TinyConfig()
    elif args.config == "medium":
        config = MediumConfig()
    elif args.config == "large":
        config = LargeConfig()
    else:
        config = ModelConfig()

    # --- Device ---
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # --- Tokenizer ---
    tok_path = config.tokenizer_path
    if not os.path.exists(tok_path):
        print(
            f"Error: Tokenizer not found at '{tok_path}'.\n"
            "Please run 'python train.py' first."
        )
        sys.exit(1)
    tokenizer = BPETokenizer.load(tok_path)
    config.vocab_size = len(tokenizer)

    # --- Checkpoint ---
    ckpt_path = args.checkpoint or _find_checkpoint(config.checkpoint_dir)
    if ckpt_path is None:
        print(
            f"Error: No checkpoint found in '{config.checkpoint_dir}'.\n"
            "Please run 'python train.py' first."
        )
        sys.exit(1)

    print(f"Loading model from {ckpt_path}...")
    model, extra = GoktugGPT.load_checkpoint(
        ckpt_path,
        vocab_size=config.vocab_size,
        n_embed=config.n_embed,
        n_head=config.n_head,
        n_layer=config.n_layer,
        dropout=0.0,
        max_seq_len=config.max_seq_len,
        device=device,
    )

    engine = ThinkingEngine(model, tokenizer, config)
    use_thinking = not args.no_thinking

    print(BANNER)
    print(
        _color(
            f"  Model: {model.num_parameters()/1e6:.2f}M params  |  "
            f"Device: {device}  |  Vocab: {len(tokenizer)}\n",
            "36",
        )
    )

    history = []

    while True:
        try:
            user_input = input(_color("You: ", "32")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # --- Commands ---
        if user_input.lower() in ("/quit", "/exit"):
            print("Goodbye!")
            break
        elif user_input.lower() == "/clear":
            history = []
            print(_color("Conversation cleared.\n", "33"))
            continue
        elif user_input.lower() == "/think":
            use_thinking = not use_thinking
            state = "ON" if use_thinking else "OFF"
            print(_color(f"Thinking stage: {state}\n", "33"))
            continue
        elif user_input.lower() == "/info":
            print(
                _color(
                    f"  Parameters: {model.num_parameters()/1e6:.2f}M\n"
                    f"  Vocab size: {len(tokenizer)}\n"
                    f"  Device:     {device}\n"
                    f"  Layers:     {config.n_layer}  Heads: {config.n_head}  Dim: {config.n_embed}\n"
                    f"  Checkpoint: {ckpt_path}\n"
                    f"  Thinking:   {'ON' if use_thinking else 'OFF'}\n",
                    "36",
                )
            )
            continue

        # --- Generate ---
        print(_color("goktugGPT: ", "34"), end="", flush=True)

        thinking, answer = engine.generate_with_thinking(
            user_message=user_input,
            history=history,
            use_thinking=use_thinking,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_tokens,
            max_think_tokens=args.max_think,
        )

        if thinking:
            print()
            print(_color("  [Thinking]", "35"))
            # Word-wrap the thinking text
            words = thinking.split()
            line, width = "  ", 0
            for w in words:
                if width + len(w) + 1 > 70:
                    print(_color(line, "90"))
                    line, width = "  " + w + " ", len(w) + 1
                else:
                    line += w + " "
                    width += len(w) + 1
            if line.strip():
                print(_color(line, "90"))
            print()
            print(_color("goktugGPT: ", "34"), end="")

        print(answer)
        print()

        # Update history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})

        # Keep history from growing too large
        if len(history) > 20:
            history = history[-20:]


if __name__ == "__main__":
    main()
