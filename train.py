"""
train.py — Main training script for goktugGPT

Usage:
    python train.py                        # Train with default TinyConfig
    python train.py --config medium        # Train with MediumConfig
    python train.py --epochs 100           # Override epoch count
    python train.py --resume               # Resume from latest checkpoint
    python train.py --tokenizer-only       # Only train the tokenizer, no model training

Steps performed:
    1. Load / train the BPE tokenizer on the training data
    2. Build the training and validation datasets
    3. Initialise the GoktugGPT model
    4. Run the training loop
    5. Save the final model checkpoint
"""

import argparse
import os
import sys

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Train goktugGPT from scratch")
    parser.add_argument(
        "--config",
        choices=["tiny", "default", "medium"],
        default="tiny",
        help="Model size configuration (default: tiny — fastest for local training)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override max_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning_rate")
    parser.add_argument("--data", type=str, default=None, help="Path to training data file")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint"
    )
    parser.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="Only train the tokenizer and exit",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    return parser.parse_args()


def _find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """
    Find the most advanced checkpoint available.
    Priority: highest step number > best_model > final_model
    """
    import glob
    ckpt_dir = checkpoint_dir

    # Look for step checkpoints — pick the one with the highest step
    step_files = glob.glob(os.path.join(ckpt_dir, "checkpoint_step_*.pt"))
    if step_files:
        # Extract step numbers and pick the highest
        def _step(p):
            m = __import__("re").search(r"checkpoint_step_(\d+)\.pt", p)
            return int(m.group(1)) if m else 0
        latest = max(step_files, key=_step)
        return latest

    # Fall back to named checkpoints
    for name in ["best_model.pt", "final_model.pt"]:
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            return p
    return None


def main():
    args = parse_args()

    # --- Load config ---
    from config import ModelConfig, TinyConfig, MediumConfig

    if args.config == "tiny":
        config = TinyConfig()
    elif args.config == "medium":
        config = MediumConfig()
    else:
        config = ModelConfig()

    # Apply CLI overrides
    if args.epochs is not None:
        config.max_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.data is not None:
        config.data_path = args.data
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  goktugGPT — Training from scratch")
    print("=" * 60)
    print(f"  Config:       {args.config}")
    print(f"  Data path:    {config.data_path}")
    print(f"  Checkpoints:  {config.checkpoint_dir}")
    print(f"  Vocab size:   {config.vocab_size}")
    print(f"  Embed dim:    {config.n_embed}")
    print(f"  Layers:       {config.n_layer}")
    print(f"  Heads:        {config.n_head}")
    print(f"  Max seq len:  {config.max_seq_len}")
    print(f"  Batch size:   {config.batch_size}")
    print(f"  Epochs:       {config.max_epochs}")
    print(f"  Learning rate:{config.learning_rate}")
    print("=" * 60 + "\n")

    # --- Step 1: Tokenizer ---
    from src.tokenizer import BPETokenizer

    tok_path = config.tokenizer_path
    if os.path.exists(tok_path):
        print(f"Loading existing tokenizer from {tok_path}")
        tokenizer = BPETokenizer.load(tok_path)
    else:
        print("Training BPE tokenizer...")
        tokenizer = BPETokenizer(vocab_size=config.vocab_size)
        with open(config.data_path, "r", encoding="utf-8") as f:
            texts = [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
        tokenizer.train(texts, special_tokens=config.special_tokens)
        tokenizer.save(tok_path)

    print(f"Tokenizer ready. Vocab size: {len(tokenizer)}")
    # Update config with actual vocab size
    config.vocab_size = len(tokenizer)

    if args.tokenizer_only:
        print("--tokenizer-only flag set. Done.")
        return

    # --- Step 2: Dataset ---
    print("\nBuilding datasets...")
    from src.training import build_dataloaders

    train_dl, val_dl = build_dataloaders(
        file_path=config.data_path,
        tokenizer=tokenizer,
        block_size=config.max_seq_len,
        batch_size=config.batch_size,
    )

    # --- Step 3: Model ---
    print("\nInitialising model...")
    from src.model import GoktugGPT

    model = GoktugGPT(
        vocab_size=config.vocab_size,
        n_embed=config.n_embed,
        n_head=config.n_head,
        n_layer=config.n_layer,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
    )

    # --- Step 4: Train ---
    from src.training import Trainer

    trainer = Trainer(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        config=config,
        checkpoint_dir=config.checkpoint_dir,
    )

    resume_path = None
    if args.resume:
        resume_path = _find_latest_checkpoint(config.checkpoint_dir)
        if resume_path:
            print(f"Resuming from: {resume_path}")
        else:
            print("No checkpoint found to resume from. Starting fresh.")

    trainer.train(resume_from=resume_path)

    print("\nTraining complete!")
    print(f"Model saved in: {config.checkpoint_dir}")
    print("Run 'python chat.py' to start chatting, or 'python gui.py' to open the GUI.")


if __name__ == "__main__":
    main()
