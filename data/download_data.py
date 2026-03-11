"""
download_data.py — Download large-scale training data for goktugGPT

Sources:
  1. WikiText-2   (~2 MB  / ~2M tokens)   — always downloaded, quick sanity check
  2. WikiText-103 (~185 MB / ~103M tokens) — serious multi-hour training
  3. Project Gutenberg classics            — 20 public-domain books (~80 MB)
  4. Wikipedia API                         — 500+ curated articles (~50 MB)

After downloading, run:
  python data/prepare_data.py
  python train.py --config tiny --data data/train_big.txt --epochs 10

Usage:
  python data/download_data.py               # WikiText-2 + Gutenberg + Wikipedia
  python data/download_data.py --large       # WikiText-103 instead of WikiText-2
  python data/download_data.py --all         # Everything (WikiText-103 + books + wiki)
  python data/download_data.py --wikitext-only
  python data/download_data.py --books-only
  python data/download_data.py --wikipedia-only
"""

import argparse
import os
import re
import sys
import time
import zipfile
import urllib.request
import urllib.error
import json
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"

# ─────────────────────────────────────────────────────────────────────────────
# WikiText
# ─────────────────────────────────────────────────────────────────────────────

WIKITEXT_URLS = {
    "wikitext-2": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip",
    "wikitext-103": "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip",
}

# ─────────────────────────────────────────────────────────────────────────────
# Project Gutenberg — public domain books (UTF-8 plain text)
# IDs reference: https://www.gutenberg.org/
# ─────────────────────────────────────────────────────────────────────────────

GUTENBERG_BOOKS = {
    # English literature
    "pride_and_prejudice":       ("1342", "Pride and Prejudice — Jane Austen"),
    "frankenstein":              ("84",   "Frankenstein — Mary Shelley"),
    "sherlock_holmes":           ("1661", "Adventures of Sherlock Holmes — Conan Doyle"),
    "moby_dick":                 ("2701", "Moby Dick — Herman Melville"),
    "alice_in_wonderland":       ("11",   "Alice in Wonderland — Lewis Carroll"),
    "great_expectations":        ("1400", "Great Expectations — Charles Dickens"),
    "a_tale_of_two_cities":      ("98",   "A Tale of Two Cities — Charles Dickens"),
    "dracula":                   ("345",  "Dracula — Bram Stoker"),
    "dorian_gray":               ("174",  "The Picture of Dorian Gray — Oscar Wilde"),
    "huckleberry_finn":          ("76",   "Adventures of Huckleberry Finn — Twain"),
    "tom_sawyer":                ("74",   "The Adventures of Tom Sawyer — Twain"),
    "romeo_and_juliet":          ("1112", "Romeo and Juliet — Shakespeare"),
    "hamlet":                    ("1524", "Hamlet — Shakespeare"),
    "macbeth":                   ("1533", "Macbeth — Shakespeare"),
    "odyssey":                   ("1727", "The Odyssey — Homer"),
    "iliad":                     ("6130", "The Iliad — Homer"),
    "crime_and_punishment":      ("2554", "Crime and Punishment — Dostoevsky"),
    "war_and_peace":             ("2600", "War and Peace — Tolstoy"),
    "metamorphosis":             ("5200", "The Metamorphosis — Kafka"),
    "don_quixote":               ("996",  "Don Quixote — Cervantes"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Wikipedia — curated topic list (via Wikipedia REST API, no scraping needed)
# ─────────────────────────────────────────────────────────────────────────────

WIKIPEDIA_TOPICS = [
    # Science
    "Artificial intelligence", "Machine learning", "Deep learning",
    "Neural network", "Transformer (machine learning model)",
    "Natural language processing", "Computer vision", "Reinforcement learning",
    "Physics", "Quantum mechanics", "General relativity", "Special relativity",
    "Thermodynamics", "Classical mechanics", "Electromagnetism",
    "Chemistry", "Periodic table", "Chemical bond", "Organic chemistry",
    "Biology", "Evolution", "DNA", "Cell (biology)", "Photosynthesis",
    "Ecology", "Genetics", "Neuroscience", "Human brain",
    "Mathematics", "Calculus", "Linear algebra", "Statistics",
    "Number theory", "Geometry", "Topology", "Abstract algebra",
    "Astronomy", "Solar system", "Black hole", "Galaxy", "Big Bang",
    "Universe", "Star", "Exoplanet", "Dark matter", "Dark energy",
    # Technology
    "Computer", "Internet", "World Wide Web", "Operating system",
    "Programming language", "Python (programming language)",
    "Algorithm", "Data structure", "Database", "Cryptography",
    "Blockchain", "Cloud computing", "Cybersecurity", "Compiler",
    "Software engineering", "Open-source software", "Linux",
    "Processor (computing)", "Graphics processing unit", "Semiconductor",
    # History
    "World War I", "World War II", "Ancient Rome", "Ancient Greece",
    "Renaissance", "Industrial Revolution", "French Revolution",
    "History of science", "History of mathematics", "History of computing",
    "Ancient Egypt", "Byzantine Empire", "Ottoman Empire",
    "History of China", "Silk Road", "Age of Exploration",
    # Philosophy
    "Philosophy", "Ethics", "Epistemology", "Metaphysics",
    "Logic", "Philosophy of mind", "Philosophy of science",
    "Stoicism", "Existentialism", "Utilitarianism",
    "Socrates", "Plato", "Aristotle", "Immanuel Kant",
    # Geography & Society
    "Climate change", "Globalization", "Democracy", "Economics",
    "Psychology", "Sociology", "Linguistics", "Anthropology",
    "Renewable energy", "Urbanization", "Human rights",
    # Arts & Culture
    "Music theory", "Literature", "Philosophy of art",
    "Architecture", "Film", "Theatre",
    # Specific concepts
    "Consciousness", "Language", "Memory", "Emotion", "Intelligence",
    "Creativity", "Learning", "Communication", "Information theory",
    "Game theory", "Chaos theory", "Complexity theory",
    "Entropy", "Energy", "Time", "Space", "Matter",
    "Protein", "Virus", "Bacteria", "Immune system",
    "Ecosystem", "Climate", "Ocean", "Atmosphere",
    "Volcano", "Earthquake", "Plate tectonics",
    "Water cycle", "Carbon cycle", "Food web",
]

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        mb = downloaded / 1_048_576
        total_mb = total_size / 1_048_576
        print(f"\r  [{bar}] {pct:5.1f}%  {mb:.1f}/{total_mb:.1f} MB", end="", flush=True)
    else:
        mb = downloaded / 1_048_576
        print(f"\r  Downloaded {mb:.1f} MB...", end="", flush=True)


def _download_file(url: str, dest: Path, description: str) -> bool:
    """Download url → dest with a progress bar. Returns True on success."""
    if dest.exists():
        print(f"  Already exists: {dest.name}")
        return True
    print(f"  Downloading: {description}")
    print(f"  URL: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        headers = {"User-Agent": "goktugGPT-data-downloader/1.0 (educational project)"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            block = 65536
            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(block)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = min(100, downloaded * 100 / total)
                        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
                        print(
                            f"\r  [{bar}] {pct:5.1f}%  "
                            f"{downloaded/1e6:.1f}/{total/1e6:.1f} MB",
                            end="", flush=True,
                        )
                    else:
                        print(f"\r  {downloaded/1e6:.1f} MB downloaded...", end="", flush=True)
        print(f"\r  Done: {dest.name}" + " " * 40)
        return True
    except Exception as e:
        print(f"\r  FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def _unzip(zip_path: Path, out_dir: Path):
    print(f"  Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    zip_path.unlink()
    print(f"  Extracted to {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# WikiText downloader
# ─────────────────────────────────────────────────────────────────────────────

def download_wikitext(variant: str = "wikitext-2") -> bool:
    print(f"\n{'='*60}")
    print(f"  WikiText — {variant}")
    print(f"{'='*60}")
    url = WIKITEXT_URLS[variant]
    out_dir = RAW_DIR / variant
    if (out_dir / "wiki.train.raw").exists():
        print(f"  Already downloaded: {out_dir}")
        return True
    zip_path = RAW_DIR / f"{variant}.zip"
    ok = _download_file(url, zip_path, f"{variant} dataset")
    if not ok:
        return False
    # Extract
    print(f"  Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            filename = os.path.basename(member)
            if not filename:
                continue
            source = zf.open(member)
            dest = out_dir / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                f.write(source.read())
    zip_path.unlink()
    print(f"  WikiText extracted to {out_dir}")
    # Count tokens roughly
    train_file = out_dir / "wiki.train.raw"
    if train_file.exists():
        size_mb = train_file.stat().st_size / 1_048_576
        print(f"  Train file: {size_mb:.1f} MB (~{size_mb*150_000:.0f} tokens approx)")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Project Gutenberg downloader
# ─────────────────────────────────────────────────────────────────────────────

def _gutenberg_url(book_id: str) -> str:
    """Build direct download URL for a Project Gutenberg book."""
    return f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"


def download_gutenberg_books() -> int:
    print(f"\n{'='*60}")
    print(f"  Project Gutenberg — {len(GUTENBERG_BOOKS)} classic books")
    print(f"{'='*60}")
    out_dir = RAW_DIR / "books"
    out_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    for name, (book_id, description) in GUTENBERG_BOOKS.items():
        dest = out_dir / f"{name}.txt"
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"  [cached] {description}")
            success += 1
            continue
        url = _gutenberg_url(book_id)
        ok = _download_file(url, dest, description)
        if not ok:
            # Try alternative URL format
            alt_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
            ok = _download_file(alt_url, dest, f"{description} (alt)")
        if ok:
            success += 1
        time.sleep(0.5)  # Be polite to gutenberg.org
    print(f"\n  Downloaded {success}/{len(GUTENBERG_BOOKS)} books")
    return success


# ─────────────────────────────────────────────────────────────────────────────
# Wikipedia API downloader
# ─────────────────────────────────────────────────────────────────────────────

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {
    "User-Agent": "goktugGPT-data-downloader/1.0 (educational project; https://github.com/goktugGPT)"
}


def _fetch_wikipedia_article(title: str) -> str:
    """Fetch the plain-text extract of a Wikipedia article via the official API."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",        # plain text, no HTML/markup
        "exsectionformat": "plain",
        "titles": title,
        "format": "json",
        "redirects": "1",
    }
    query = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
    url = f"{WIKI_API}?{query}"
    try:
        import urllib.parse
        req = urllib.request.Request(url, headers=WIKI_HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "")
            if extract and len(extract) > 200:
                return extract
    except Exception:
        pass
    return ""


def download_wikipedia(topics: list = None) -> int:
    import urllib.parse
    if topics is None:
        topics = WIKIPEDIA_TOPICS
    print(f"\n{'='*60}")
    print(f"  Wikipedia API — {len(topics)} articles")
    print(f"{'='*60}")
    out_dir = RAW_DIR / "wikipedia"
    out_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    for i, title in enumerate(topics):
        safe_name = re.sub(r'[^\w\-]', '_', title).lower()
        dest = out_dir / f"{safe_name}.txt"
        if dest.exists() and dest.stat().st_size > 500:
            success += 1
            print(f"  [{i+1:3d}/{len(topics)}] [cached] {title}", flush=True)
            continue
        text = _fetch_wikipedia_article(title)
        if text:
            dest.write_text(text, encoding="utf-8")
            size_kb = len(text) / 1024
            print(f"  [{i+1:3d}/{len(topics)}] ✓ {title} ({size_kb:.0f} KB)", flush=True)
            success += 1
        else:
            print(f"  [{i+1:3d}/{len(topics)}] ✗ {title} (not found)", flush=True)
        time.sleep(0.3)  # Respect Wikipedia rate limits
    print(f"\n  Fetched {success}/{len(topics)} Wikipedia articles")
    return success


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Download training data for goktugGPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data/download_data.py                   # WikiText-2 + books + Wikipedia
  python data/download_data.py --large           # WikiText-103 (185 MB, ~103M tokens)
  python data/download_data.py --wikitext-only   # Only WikiText
  python data/download_data.py --books-only      # Only Gutenberg books
  python data/download_data.py --wikipedia-only  # Only Wikipedia articles
  python data/download_data.py --all             # WikiText-103 + books + Wikipedia

After downloading:
  python data/prepare_data.py
  python train.py --config tiny --data data/train_big.txt --epochs 10
        """,
    )
    p.add_argument("--large", action="store_true",
                   help="Download WikiText-103 (~185 MB) instead of WikiText-2 (~2 MB)")
    p.add_argument("--all", action="store_true",
                   help="Download everything: WikiText-103 + books + Wikipedia")
    p.add_argument("--wikitext-only", action="store_true")
    p.add_argument("--books-only", action="store_true")
    p.add_argument("--wikipedia-only", action="store_true")
    p.add_argument("--no-books", action="store_true", help="Skip Gutenberg books")
    p.add_argument("--no-wikipedia", action="store_true", help="Skip Wikipedia")
    return p.parse_args()


def main():
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  goktugGPT — Data Downloader")
    print("=" * 60)
    print(f"  Output directory: {RAW_DIR.resolve()}")

    use_large = args.large or args.all
    do_wikitext = True
    do_books = True
    do_wikipedia = True

    if args.wikitext_only:
        do_books = False
        do_wikipedia = False
    if args.books_only:
        do_wikitext = False
        do_wikipedia = False
    if args.wikipedia_only:
        do_wikitext = False
        do_books = False
    if args.no_books:
        do_books = False
    if args.no_wikipedia:
        do_wikipedia = False

    results = {}

    # --- WikiText ---
    if do_wikitext:
        variant = "wikitext-103" if use_large else "wikitext-2"
        results["wikitext"] = download_wikitext(variant)

    # --- Gutenberg books ---
    if do_books:
        results["books"] = download_gutenberg_books()

    # --- Wikipedia ---
    if do_wikipedia:
        results["wikipedia"] = download_wikipedia()

    # --- Summary ---
    print(f"\n{'='*60}")
    print("  Download Summary")
    print(f"{'='*60}")

    total_mb = 0
    for source_dir in RAW_DIR.iterdir():
        if source_dir.is_dir():
            size = sum(f.stat().st_size for f in source_dir.rglob("*") if f.is_file())
            size_mb = size / 1_048_576
            total_mb += size_mb
            files = sum(1 for f in source_dir.rglob("*") if f.is_file())
            print(f"  {source_dir.name:20s} {size_mb:7.1f} MB  ({files} files)")

    print(f"  {'TOTAL':20s} {total_mb:7.1f} MB")
    # Rough token estimate: ~150k tokens per MB of English text
    est_tokens = total_mb * 150_000
    print(f"  Estimated tokens:    ~{est_tokens/1_000_000:.0f}M")

    print(f"""
Next steps:
  1. python data/prepare_data.py
  2. python train.py --config tiny --data data/train_big.txt --epochs 10
""")


if __name__ == "__main__":
    main()
