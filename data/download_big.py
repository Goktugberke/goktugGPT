"""
download_big.py — 1M+ line dataset builder for goktugGPT

Downloads large-scale datasets from Hugging Face and combines them
into a massive training file.

Sources (conversation data):
  1. SODA (allenai/soda)           — 1.5M social dialogues     -> ~800K pairs
  2. OpenAssistant (oasst1)        — High-quality Q&A           -> ~10K pairs
  3. Alpaca (tatsu-lab/alpaca)     — Instruction following      -> ~52K pairs
  4. Dolly (databricks-dolly-15k)  — Instruction pairs          -> ~15K pairs
  5. Existing train_chat.txt       — Cornell + DailyDialog + synthetic -> ~94K lines

Target: 1,000,000+ training lines

Requirements:
  pip install datasets

Usage:
  python data/download_big.py                    # Full download (~1M+ lines)
  python data/download_big.py --max-soda 300000  # Limit SODA pairs
  python data/download_big.py --skip-soda        # Skip SODA (if disk limited)

Output:
  data/train_chat_big.txt

2-Stage Training (recommended):
  Stage 1 — Train on big conversation data:
    python train.py --config large --data data/train_chat_big.txt --epochs 5 \\
        --batch-size 4 --checkpoint-dir /kaggle/working/checkpoints

  Stage 2 — Continue training for more epochs:
    python train.py --config large --data data/train_chat_big.txt --epochs 15 \\
        --batch-size 4 --checkpoint-dir /kaggle/working/checkpoints --resume
"""

import argparse
import os
import random
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR
OUTPUT_FILE = os.path.join(DATA_DIR, "train_chat_big.txt")


def ensure_datasets_library():
    """Check that the datasets library is installed."""
    try:
        import datasets  # noqa: F401
        return True
    except ImportError:
        print("ERROR: 'datasets' library not installed.")
        print("Run:  pip install datasets")
        sys.exit(1)


# ---------------------------------------------------------------------------
# SODA — Social Dialogue Dataset (allenai/soda)
# ---------------------------------------------------------------------------

def download_soda(max_pairs=800_000):
    """Download SODA social dialogues from Hugging Face."""
    print("\n" + "=" * 60)
    print("  SODA -- Social Dialogue Dataset")
    print("  Source: allenai/soda (Hugging Face)")
    print("=" * 60)

    from datasets import load_dataset

    print("  Downloading SODA (streaming mode)...")
    ds = load_dataset("allenai/soda", split="train", streaming=True)

    lines = []
    count = 0

    for example in ds:
        if count >= max_pairs:
            break

        dialogue = example.get("dialogue", [])

        # Create pairs from consecutive turns
        for i in range(0, len(dialogue) - 1):
            if count >= max_pairs:
                break

            user_msg = dialogue[i].strip()
            assistant_msg = dialogue[i + 1].strip()

            # Skip very short or empty messages
            if len(user_msg) < 3 or len(assistant_msg) < 3:
                continue

            # Skip very long messages (likely noise)
            if len(user_msg) > 500 or len(assistant_msg) > 500:
                user_msg = user_msg[:500]
                assistant_msg = assistant_msg[:500]

            line = (
                f"<user> {user_msg} "
                f"<assistant> <think> Natural conversation exchange. "
                f"Respond contextually. </think> {assistant_msg} <eos>"
            )
            lines.append(line)
            count += 1

        if count % 100_000 == 0 and count > 0:
            print(f"    {count:,} pairs processed...")

    print(f"  -> {len(lines):,} conversation pairs loaded")
    return lines


# ---------------------------------------------------------------------------
# OpenAssistant (OpenAssistant/oasst1)
# ---------------------------------------------------------------------------

def download_oasst():
    """Download OpenAssistant Q&A conversations."""
    print("\n" + "=" * 60)
    print("  OpenAssistant -- Q&A Conversations")
    print("  Source: OpenAssistant/oasst1 (Hugging Face)")
    print("=" * 60)

    from datasets import load_dataset

    print("  Downloading OpenAssistant...")
    ds = load_dataset("OpenAssistant/oasst1", split="train")

    # Build message lookup
    messages = {}
    for row in ds:
        messages[row["message_id"]] = row

    # Find prompter -> assistant pairs
    lines = []

    for msg_id, msg in messages.items():
        if msg["role"] != "prompter":
            continue

        user_text = msg["text"].strip()
        if len(user_text) < 5:
            continue

        # Find best assistant reply (highest rank)
        best_reply = None
        best_rank = -1

        for reply_id, reply in messages.items():
            if reply["parent_id"] == msg_id and reply["role"] == "assistant":
                rank = reply.get("rank", 0) or 0
                if rank > best_rank or best_reply is None:
                    best_reply = reply
                    best_rank = rank

        if best_reply is None:
            continue

        assistant_text = best_reply["text"].strip()
        if len(assistant_text) < 5:
            continue

        # Truncate very long responses
        if len(assistant_text) > 800:
            assistant_text = assistant_text[:800]
        if len(user_text) > 500:
            user_text = user_text[:500]

        line = (
            f"<user> {user_text} "
            f"<assistant> <think> Process the question carefully "
            f"and give a helpful, accurate response. </think> "
            f"{assistant_text} <eos>"
        )
        lines.append(line)

    print(f"  -> {len(lines):,} Q&A pairs loaded")
    return lines


# ---------------------------------------------------------------------------
# Alpaca (tatsu-lab/alpaca)
# ---------------------------------------------------------------------------

def download_alpaca():
    """Download Stanford Alpaca instruction-following dataset."""
    print("\n" + "=" * 60)
    print("  Alpaca -- Instruction Following")
    print("  Source: tatsu-lab/alpaca (Hugging Face)")
    print("=" * 60)

    from datasets import load_dataset

    print("  Downloading Alpaca...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    lines = []

    for example in ds:
        instruction = example.get("instruction", "").strip()
        inp = example.get("input", "").strip()
        output = example.get("output", "").strip()

        if not instruction or not output:
            continue
        if len(output) < 5:
            continue

        # Combine instruction + input
        if inp:
            user_text = f"{instruction} {inp}"
        else:
            user_text = instruction

        # Truncate
        if len(user_text) > 500:
            user_text = user_text[:500]
        if len(output) > 800:
            output = output[:800]

        line = (
            f"<user> {user_text} "
            f"<assistant> <think> Process the question carefully "
            f"and give a helpful, accurate response. </think> "
            f"{output} <eos>"
        )
        lines.append(line)

    print(f"  -> {len(lines):,} instruction pairs loaded")
    return lines


# ---------------------------------------------------------------------------
# Dolly (databricks/databricks-dolly-15k)
# ---------------------------------------------------------------------------

def download_dolly():
    """Download Databricks Dolly instruction dataset."""
    print("\n" + "=" * 60)
    print("  Dolly -- Instruction Dataset")
    print("  Source: databricks/databricks-dolly-15k (Hugging Face)")
    print("=" * 60)

    from datasets import load_dataset

    print("  Downloading Dolly...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    lines = []

    for example in ds:
        instruction = example.get("instruction", "").strip()
        context = example.get("context", "").strip()
        response = example.get("response", "").strip()

        if not instruction or not response:
            continue
        if len(response) < 5:
            continue

        if context:
            user_text = f"{instruction} Context: {context}"
        else:
            user_text = instruction

        if len(user_text) > 500:
            user_text = user_text[:500]
        if len(response) > 800:
            response = response[:800]

        line = (
            f"<user> {user_text} "
            f"<assistant> <think> Process the question carefully "
            f"and give a helpful, accurate response. </think> "
            f"{response} <eos>"
        )
        lines.append(line)

    print(f"  -> {len(lines):,} instruction pairs loaded")
    return lines


# ---------------------------------------------------------------------------
# Existing data
# ---------------------------------------------------------------------------

def load_existing():
    """Load existing train_chat.txt if available."""
    existing_file = os.path.join(DATA_DIR, "train_chat.txt")
    if os.path.exists(existing_file):
        with open(existing_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        print(f"\n  Existing train_chat.txt: {len(lines):,} lines")
        return lines
    print("\n  No existing train_chat.txt found (run download_conversations.py first)")
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download 1M+ line dataset for goktugGPT"
    )
    parser.add_argument(
        "--max-soda", type=int, default=800_000,
        help="Max SODA conversation pairs (default: 800K)"
    )
    parser.add_argument(
        "--skip-soda", action="store_true",
        help="Skip SODA dataset"
    )
    parser.add_argument(
        "--skip-oasst", action="store_true",
        help="Skip OpenAssistant dataset"
    )
    parser.add_argument(
        "--skip-alpaca", action="store_true",
        help="Skip Alpaca dataset"
    )
    parser.add_argument(
        "--skip-dolly", action="store_true",
        help="Skip Dolly dataset"
    )
    args = parser.parse_args()

    ensure_datasets_library()

    print("=" * 60)
    print("  goktugGPT -- 1M+ Dataset Builder")
    print("=" * 60)

    all_lines = []

    # 1. Existing data
    existing = load_existing()
    all_lines.extend(existing)

    # 2. SODA (biggest source)
    if not args.skip_soda:
        soda = download_soda(max_pairs=args.max_soda)
        all_lines.extend(soda)

    # 3. OpenAssistant
    if not args.skip_oasst:
        oasst = download_oasst()
        all_lines.extend(oasst)

    # 4. Alpaca
    if not args.skip_alpaca:
        alpaca = download_alpaca()
        all_lines.extend(alpaca)

    # 5. Dolly
    if not args.skip_dolly:
        dolly = download_dolly()
        all_lines.extend(dolly)

    # --- Deduplicate ---
    print(f"\n  Total before dedup: {len(all_lines):,}")
    all_lines = list(set(all_lines))
    print(f"  Total after dedup:  {len(all_lines):,}")

    # --- Shuffle and write ---
    random.shuffle(all_lines)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(line + "\n")

    size_mb = os.path.getsize(OUTPUT_FILE) / 1e6
    total_chars = sum(len(l) for l in all_lines)
    est_tokens = total_chars // 4

    print(f"\n{'=' * 60}")
    print(f"  Dataset Statistics")
    print(f"{'=' * 60}")
    print(f"  Total training lines  : {len(all_lines):,}")
    print(f"  Total characters      : {total_chars / 1e6:.1f}M")
    print(f"  Estimated tokens      : ~{est_tokens / 1e6:.1f}M")
    print(f"  File size             : {size_mb:.1f} MB")
    print(f"  Output                : {OUTPUT_FILE}")

    print(f"\n{'=' * 60}")
    print(f"  NEXT STEPS")
    print(f"{'=' * 60}")
    print()
    print("  1. Delete old checkpoint (IMPORTANT — new tokenizer needed):")
    print("       rm checkpoints/tokenizer.json checkpoints/best_model.pt")
    print()
    print("  2. Train on Kaggle (Save & Run All for background):")
    print("       python train.py --config large --data data/train_chat_big.txt \\")
    print("           --epochs 10 --batch-size 4 \\")
    print("           --checkpoint-dir /kaggle/working/checkpoints")
    print()
    print("  3. Resume if interrupted:")
    print("       python train.py --config large --data data/train_chat_big.txt \\")
    print("           --epochs 10 --batch-size 4 \\")
    print("           --checkpoint-dir /kaggle/working/checkpoints --resume")
    print()


if __name__ == "__main__":
    main()
