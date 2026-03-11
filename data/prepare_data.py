"""
prepare_data.py — Clean, merge, and format all downloaded data for goktugGPT.

Reads from:
  data/raw/wikitext-2/  or  data/raw/wikitext-103/
  data/raw/books/
  data/raw/wikipedia/
  data/train.txt          (existing Q&A examples — always included)

Outputs:
  data/train_big.txt      (everything combined, shuffled, formatted)

Run:
  python data/prepare_data.py
  python data/prepare_data.py --include-wikitext103   # if you downloaded the big one
  python data/prepare_data.py --stats                  # just show stats, don't write

Then train:
  python train.py --config tiny --data data/train_big.txt --epochs 10
"""

import argparse
import os
import re
import random
from pathlib import Path
from typing import List

random.seed(42)

RAW_DIR = Path(__file__).parent / "raw"
OUT_DIR = Path(__file__).parent
QA_FILE = OUT_DIR / "train.txt"
OUT_FILE = OUT_DIR / "train_big.txt"

# ─────────────────────────────────────────────────────────────────────────────
# WikiText cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_wikitext(text: str) -> List[str]:
    """
    Extract meaningful paragraphs from WikiText-2/103 raw format.

    WikiText format:
      = Title =
      = = Subsection = =
      Plain paragraph text...
      @-@ (hyphenation marker)
    """
    paragraphs = []
    for line in text.splitlines():
        line = line.strip()
        # Skip headings (= Title =, = = Sub = =, etc.)
        if re.match(r'^=+\s.*\s=+$', line):
            continue
        # Skip empty lines and very short lines
        if len(line) < 40:
            continue
        # Clean WikiText artifacts
        line = line.replace(" @-@ ", "-")
        line = line.replace(" @.@ ", ".")
        line = line.replace(" @,@ ", ",")
        # Remove wiki-style citations like [1], [2], etc.
        line = re.sub(r'\[\d+\]', '', line)
        # Remove excessive whitespace
        line = re.sub(r' +', ' ', line).strip()
        if len(line) > 40:
            paragraphs.append(line)
    return paragraphs


def load_wikitext(prefer_large: bool = False) -> List[str]:
    """Load and clean the WikiText training set."""
    variants = ["wikitext-103", "wikitext-2"] if prefer_large else ["wikitext-2", "wikitext-103"]
    for variant in variants:
        candidate = RAW_DIR / variant / "wiki.train.raw"
        if candidate.exists():
            print(f"  Loading {variant}...")
            text = candidate.read_text(encoding="utf-8", errors="ignore")
            paras = clean_wikitext(text)
            size_mb = candidate.stat().st_size / 1_048_576
            print(f"  → {len(paras)} paragraphs from {size_mb:.1f} MB")
            return paras
    print("  WikiText not found. Run: python data/download_data.py")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Project Gutenberg cleaning
# ─────────────────────────────────────────────────────────────────────────────

_GUTENBERG_START = re.compile(
    r'\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG', re.IGNORECASE
)
_GUTENBERG_END = re.compile(
    r'\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG', re.IGNORECASE
)


def clean_gutenberg_book(text: str) -> List[str]:
    """
    Strip Gutenberg header/footer and extract clean paragraphs.
    """
    lines = text.splitlines()
    # Find content boundaries
    start_idx = 0
    end_idx = len(lines)
    for i, line in enumerate(lines):
        if _GUTENBERG_START.search(line):
            start_idx = i + 1
        if _GUTENBERG_END.search(line):
            end_idx = i
            break

    content_lines = lines[start_idx:end_idx]
    content = "\n".join(content_lines)

    # Split into paragraphs (double newline)
    raw_paragraphs = re.split(r'\n{2,}', content)

    paragraphs = []
    for para in raw_paragraphs:
        # Collapse internal newlines
        para = re.sub(r'\n', ' ', para)
        para = re.sub(r' +', ' ', para).strip()

        # Skip chapter headings and very short text
        if len(para) < 60:
            continue
        # Skip lines that look like chapter titles (ALL CAPS or Roman numerals only)
        if re.match(r'^[IVXLCDM]+\.?\s*$', para):
            continue
        if re.match(r'^CHAPTER\s+', para, re.IGNORECASE) and len(para) < 40:
            continue
        paragraphs.append(para)

    return paragraphs


def load_gutenberg_books() -> List[str]:
    """Load and clean all downloaded Gutenberg books."""
    books_dir = RAW_DIR / "books"
    if not books_dir.exists():
        print("  No Gutenberg books found. Run: python data/download_data.py --books-only")
        return []

    all_paragraphs = []
    book_files = list(books_dir.glob("*.txt"))
    print(f"  Loading {len(book_files)} Gutenberg books...")
    for book_path in sorted(book_files):
        text = book_path.read_text(encoding="utf-8", errors="ignore")
        paras = clean_gutenberg_book(text)
        size_kb = book_path.stat().st_size / 1024
        print(f"    {book_path.stem:35s} {len(paras):5d} paragraphs  ({size_kb:.0f} KB)")
        all_paragraphs.extend(paras)

    print(f"  → {len(all_paragraphs)} total paragraphs from books")
    return all_paragraphs


# ─────────────────────────────────────────────────────────────────────────────
# Wikipedia cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_wikipedia_article(text: str) -> List[str]:
    """
    Clean a Wikipedia plaintext extract.
    Remove section headers, citations, and short fragments.
    """
    paragraphs = []
    for para in re.split(r'\n{2,}', text):
        para = re.sub(r'\n', ' ', para)
        para = re.sub(r' +', ' ', para).strip()
        # Skip section headers (== Header ==)
        if re.match(r'^=+\s', para):
            continue
        # Skip very short lines
        if len(para) < 50:
            continue
        # Remove citation brackets like [1], [2]
        para = re.sub(r'\[\d+\]', '', para)
        paragraphs.append(para)
    return paragraphs


def load_wikipedia() -> List[str]:
    """Load and clean all downloaded Wikipedia articles."""
    wiki_dir = RAW_DIR / "wikipedia"
    if not wiki_dir.exists():
        print("  No Wikipedia data found. Run: python data/download_data.py --wikipedia-only")
        return []

    all_paragraphs = []
    wiki_files = list(wiki_dir.glob("*.txt"))
    print(f"  Loading {len(wiki_files)} Wikipedia articles...")
    for wiki_path in sorted(wiki_files):
        text = wiki_path.read_text(encoding="utf-8", errors="ignore")
        paras = clean_wikipedia_article(text)
        all_paragraphs.extend(paras)

    print(f"  → {len(all_paragraphs)} total paragraphs from Wikipedia")
    return all_paragraphs


# ─────────────────────────────────────────────────────────────────────────────
# Q&A data (existing train.txt)
# ─────────────────────────────────────────────────────────────────────────────

def load_qa_examples() -> List[str]:
    """Load the existing hand-crafted Q&A training examples."""
    if not QA_FILE.exists():
        print(f"  Q&A file not found: {QA_FILE}")
        return []
    lines = []
    with open(QA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    print(f"  → {len(lines)} Q&A examples from {QA_FILE.name}")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Formatting raw text as training samples
# ─────────────────────────────────────────────────────────────────────────────

def chunk_paragraphs(paragraphs: List[str], chunk_size: int = 3) -> List[str]:
    """
    Group consecutive paragraphs into chunks for richer context.
    This gives the model longer contiguous text to learn from.
    """
    chunks = []
    for i in range(0, len(paragraphs), chunk_size):
        group = paragraphs[i: i + chunk_size]
        chunk = " ".join(group)
        if len(chunk) > 100:
            chunks.append(chunk)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def prepare(args):
    print("\n" + "=" * 60)
    print("  goktugGPT — Data Preparation Pipeline")
    print("=" * 60)

    all_lines: List[str] = []

    # 1. Q&A examples (always included first — they define the chat format)
    print("\n[1] Q&A examples (data/train.txt):")
    qa = load_qa_examples()
    # Repeat Q&A multiple times so the model really learns the format
    qa_repeats = 8  # Include Q&A 8× so chat format is well reinforced
    for _ in range(qa_repeats):
        all_lines.extend(qa)
    print(f"  Including {len(qa)} Q&A examples × {qa_repeats} repetitions = {len(qa)*qa_repeats}")

    # 2. WikiText
    print("\n[2] WikiText:")
    prefer_large = args.include_wikitext103
    wiki_paras = load_wikitext(prefer_large=prefer_large)
    if wiki_paras:
        wiki_chunks = chunk_paragraphs(wiki_paras, chunk_size=3)
        all_lines.extend(wiki_chunks)
        print(f"  Added {len(wiki_chunks)} WikiText chunks")

    # 3. Gutenberg books
    print("\n[3] Project Gutenberg books:")
    book_paras = load_gutenberg_books()
    if book_paras:
        book_chunks = chunk_paragraphs(book_paras, chunk_size=4)
        all_lines.extend(book_chunks)
        print(f"  Added {len(book_chunks)} book chunks")

    # 4. Wikipedia
    print("\n[4] Wikipedia articles:")
    wiki_article_paras = load_wikipedia()
    if wiki_article_paras:
        wiki_article_chunks = chunk_paragraphs(wiki_article_paras, chunk_size=3)
        all_lines.extend(wiki_article_chunks)
        print(f"  Added {len(wiki_article_chunks)} Wikipedia chunks")

    # ─── Shuffle (except keep Q&A well distributed) ─────────────────────────
    print("\n[5] Shuffling and writing...")
    # The Q&A lines are already at the front; shuffle the whole thing
    random.shuffle(all_lines)

    # ─── Stats ──────────────────────────────────────────────────────────────
    total_chars = sum(len(l) for l in all_lines)
    est_tokens = total_chars / 4.5  # rough chars-per-token for English
    total_lines = len(all_lines)

    print(f"\n{'='*60}")
    print("  Dataset Statistics")
    print(f"{'='*60}")
    print(f"  Total lines:     {total_lines:,}")
    print(f"  Total chars:     {total_chars/1_000_000:.2f}M")
    print(f"  Est. tokens:     ~{est_tokens/1_000_000:.1f}M")
    print(f"  Output file:     {OUT_FILE}")

    if args.stats:
        print("\n  (--stats mode: not writing output file)")
        return

    # ─── Write ──────────────────────────────────────────────────────────────
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("# goktugGPT combined training data\n")
        f.write("# Generated by data/prepare_data.py\n")
        f.write(f"# Lines: {total_lines:,}  |  Est tokens: ~{est_tokens/1e6:.1f}M\n")
        f.write("\n")
        for line in all_lines:
            f.write(line.strip() + "\n")

    out_mb = OUT_FILE.stat().st_size / 1_048_576
    print(f"  Written {out_mb:.1f} MB → {OUT_FILE}")

    # ─── Training guidance ──────────────────────────────────────────────────
    _print_guidance(est_tokens)


def _print_guidance(est_tokens: float):
    print(f"""
{'='*60}
  Next: Train the model
{'='*60}

  # Delete old tokenizer so it re-learns from the full dataset:
  del checkpoints\\tokenizer.json

  # Quick run (few hours on CPU):
  python train.py --config tiny --data data/train_big.txt --epochs 5

  # Longer, higher quality (overnight on CPU):
  python train.py --config tiny --data data/train_big.txt --epochs 15

  # With GPU (much faster):
  python train.py --config default --data data/train_big.txt --epochs 20

  # Chat:
  python chat.py
  python gui.py

  Estimated training time per epoch on CPU (TinyConfig):
    {est_tokens/1e6:.0f}M tokens / 256 block_size / batch_size 4 = ~{est_tokens/256/4:.0f} steps
    At ~7 steps/sec ≈ {est_tokens/256/4/7/3600:.1f} hours per epoch
""")


def parse_args():
    p = argparse.ArgumentParser(description="Prepare goktugGPT training data")
    p.add_argument("--include-wikitext103", action="store_true",
                   help="Prefer WikiText-103 over WikiText-2 if both are present")
    p.add_argument("--stats", action="store_true",
                   help="Print statistics only, do not write output file")
    p.add_argument("--chunk-size", type=int, default=3,
                   help="Number of paragraphs to group per training sample (default: 3)")
    return p.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
