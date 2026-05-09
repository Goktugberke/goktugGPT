"""
prepare_data.py — GoktugGPT Data Preparation

What this script does:
  1. Reads existing synthetic QA data (data/train_chat.txt) — keeps <think> blocks
  2. Downloads Dolly-15k (databricks/dolly-15k) from HuggingFace
  3. Downloads Alpaca-cleaned (yahma/alpaca-cleaned) from HuggingFace
  4. Filters out movie/creative dialog (Cornell data skipped entirely)
  5. Formats everything as single-line conversation examples
  6. Shuffles and splits 95% train / 5% val
  7. Writes data/train_clean.txt and data/val_clean.txt

Usage:
    python prepare_data.py
    python prepare_data.py --no-dolly      # Skip Dolly-15k
    python prepare_data.py --no-alpaca     # Skip Alpaca-cleaned
    python prepare_data.py --max-length 300  # Skip examples > 300 words
"""

import argparse
import os
import random
import re
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Strip whitespace and collapse internal newlines to spaces."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def format_qa(user_text: str, assistant_text: str) -> str:
    """Format a single Q&A pair as a training line (no <think> block)."""
    user_text = clean_text(user_text)
    assistant_text = clean_text(assistant_text)
    return f"<user> {user_text} <assistant> {assistant_text} <eos>"


def word_count(line: str) -> int:
    return len(line.split())


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_existing_chat(path: str) -> list:
    """Load existing train_chat.txt — already properly formatted."""
    if not os.path.exists(path):
        print(f"  [skip] {path} not found.")
        return []
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            # Only keep lines that look like valid conversation examples
            if "<user>" in line and "<assistant>" in line and "<eos>" in line:
                examples.append(line)
    print(f"  Loaded {len(examples):,} examples from {path}")
    return examples


def load_dolly(max_length: int) -> list:
    """Download and format Dolly-15k."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [error] 'datasets' package not found. Run: pip install datasets")
        return []

    print("  Downloading databricks/dolly-15k from HuggingFace...")
    try:
        ds = load_dataset("databricks/dolly-15k", split="train", trust_remote_code=False)
    except Exception as e:
        print(f"  [error] Could not download Dolly-15k: {e}")
        return []

    examples = []
    skipped = 0
    for row in ds:
        instruction = clean_text(row.get("instruction", ""))
        context = clean_text(row.get("context", ""))
        response = clean_text(row.get("response", ""))

        if not instruction or not response:
            skipped += 1
            continue

        # If context is provided, append it to the instruction
        if context:
            user_text = f"{instruction}\n\nContext: {context}"
        else:
            user_text = instruction

        line = format_qa(user_text, response)
        if word_count(line) > max_length:
            skipped += 1
            continue
        examples.append(line)

    print(f"  Loaded {len(examples):,} examples from Dolly-15k  ({skipped} skipped as too long)")
    return examples


def load_alpaca(max_length: int) -> list:
    """Load Alpaca-cleaned — uses local data/alpaca_data.json if present, otherwise downloads."""
    LOCAL_PATH = os.path.join("data", "alpaca_data.json")

    rows = None

    # Try local file first (saves time if already downloaded)
    if os.path.exists(LOCAL_PATH):
        print(f"  Found local {LOCAL_PATH} — loading from disk (skipping download)")
        import json
        with open(LOCAL_PATH, "r", encoding="utf-8") as f:
            rows = json.load(f)
    else:
        try:
            from datasets import load_dataset
        except ImportError:
            print("  [error] 'datasets' package not found. Run: pip install datasets")
            return []
        print("  Downloading yahma/alpaca-cleaned from HuggingFace...")
        try:
            ds = load_dataset("yahma/alpaca-cleaned", split="train", trust_remote_code=False)
            rows = list(ds)
        except Exception as e:
            print(f"  [error] Could not download Alpaca-cleaned: {e}")
            return []

    examples = []
    skipped = 0
    for row in rows:
        instruction = clean_text(row.get("instruction", ""))
        inp = clean_text(row.get("input", ""))
        output = clean_text(row.get("output", ""))

        if not instruction or not output:
            skipped += 1
            continue

        # Combine instruction + input if input is non-empty
        if inp:
            user_text = f"{instruction}\n\n{inp}"
        else:
            user_text = instruction

        line = format_qa(user_text, output)
        if word_count(line) > max_length:
            skipped += 1
            continue
        examples.append(line)

    print(f"  Loaded {len(examples):,} examples from Alpaca  ({skipped} skipped as too long)")
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare clean training data for GoktugGPT")
    parser.add_argument("--no-dolly", action="store_true", help="Skip Dolly-15k download")
    parser.add_argument("--no-alpaca", action="store_true", help="Skip Alpaca-cleaned download")
    parser.add_argument(
        "--max-length",
        type=int,
        default=250,
        help="Skip examples with more than this many words (default: 250)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="Fraction of data to use as validation set (default: 0.05)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs("data", exist_ok=True)

    print("\n" + "=" * 60)
    print("  GoktugGPT — Data Preparation")
    print("=" * 60)

    all_examples = []

    # 1. Existing synthetic QA data (has <think> blocks → keep as-is)
    print("\n[1/3] Loading existing synthetic QA data...")
    existing = load_existing_chat("data/train_chat.txt")
    all_examples.extend(existing)

    # 2. Dolly-15k
    if not args.no_dolly:
        print("\n[2/3] Downloading Dolly-15k...")
        dolly = load_dolly(args.max_length)
        all_examples.extend(dolly)
    else:
        print("\n[2/3] Dolly-15k skipped (--no-dolly)")

    # 3. Alpaca-cleaned
    if not args.no_alpaca:
        print("\n[3/3] Downloading Alpaca-cleaned...")
        alpaca = load_alpaca(args.max_length)
        all_examples.extend(alpaca)
    else:
        print("\n[3/3] Alpaca-cleaned skipped (--no-alpaca)")

    if not all_examples:
        print("\n[error] No examples collected. Aborting.")
        sys.exit(1)

    # Shuffle
    random.shuffle(all_examples)

    # Split train / val
    split_at = max(1, int(len(all_examples) * (1 - args.val_fraction)))
    train_examples = all_examples[:split_at]
    val_examples = all_examples[split_at:]

    # Write
    train_path = "data/train_clean.txt"
    val_path = "data/val_clean.txt"

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("# GoktugGPT clean training data\n")
        f.write(f"# Sources: train_chat.txt + Dolly-15k + Alpaca-cleaned\n")
        f.write(f"# Total train examples: {len(train_examples):,}\n")
        for line in train_examples:
            f.write(line + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        f.write("# GoktugGPT clean validation data\n")
        f.write(f"# Total val examples: {len(val_examples):,}\n")
        for line in val_examples:
            f.write(line + "\n")

    print("\n" + "=" * 60)
    print(f"  Done!")
    print(f"  Train:      {len(train_examples):,} examples → {train_path}")
    print(f"  Validation: {len(val_examples):,} examples  → {val_path}")
    print(f"  Total:      {len(all_examples):,} examples")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Train a new model:")
    print("     python train.py --config medium --data data/train_clean.txt --val-data data/val_clean.txt")
    print("  2. Or resume from existing checkpoint:")
    print("     python train.py --config medium --data data/train_clean.txt --val-data data/val_clean.txt --resume")
    print()


if __name__ == "__main__":
    main()
