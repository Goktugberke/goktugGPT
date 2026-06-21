"""
prepare_data.py — GoktugGPT Data Preparation (EN + TR, large scale)

Builds a clean conversational training set in the model's format:

    <user> {q} <assistant> {a} <eos>
    <user> {q} <assistant> <think> {reasoning} </think> {a} <eos>   (reasoning data)
    <user> q1 <assistant> a1 <user> q2 <assistant> a2 <eos>          (multi-turn)

Handles three source shapes automatically:
  1. instruction / input / output   (Alpaca, Dolly, merve, OpenOrca-tr, …)
  2. question / answer-with-#### CoT (GSM8K, gsm8k-tr → <think> blocks)
  3. ShareGPT "conversations" / chat "messages" (OpenHermes, UltraChat, …)

Turkish + English sources are pulled from HuggingFace. Huge sets are sampled
via streaming so you don't download tens of GB unless you ask for it.

Usage:
    python prepare_data.py                       # balanced EN+TR default mix
    python prepare_data.py --turkish-only        # only Turkish sources
    python prepare_data.py --sample-tr 500000 --sample-en 500000   # push it
    python prepare_data.py --max-length 400      # allow longer examples
    python prepare_data.py --everything          # pull full big sets (huge!)
"""

import argparse
import os
import random
import re
import sys


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def clean_text(text) -> str:
    text = "" if text is None else str(text)
    return re.sub(r"\s+", " ", text.strip())


def format_qa(user_text: str, assistant_text: str) -> str:
    return f"<user> {clean_text(user_text)} <assistant> {clean_text(assistant_text)} <eos>"


def format_cot(user_text: str, reasoning: str, answer: str) -> str:
    """Reasoning example with an explicit <think> block."""
    return (
        f"<user> {clean_text(user_text)} <assistant> "
        f"<think> {clean_text(reasoning)} </think> {clean_text(answer)} <eos>"
    )


def word_count(line: str) -> int:
    return len(line.split())


def _try_import_datasets():
    try:
        from datasets import load_dataset  # noqa: F401
        return load_dataset
    except ImportError:
        print("  [error] 'datasets' not installed. Run: pip install datasets")
        return None


# Candidate column names for auto-detection across heterogeneous datasets.
_INSTR_COLS = ["instruction", "talimat", "question-turkish", "question", "soru",
               "prompt", "input", "user", "text"]
_OUTPUT_COLS = ["output", "çıktı", "response-turkish", "response", "answer",
                "cevap", "completion", "target"]
_INPUT_COLS = ["input", "giriş", "context", "system_prompt-turkish"]


def _pick(row: dict, candidates: list):
    for c in candidates:
        if c in row and row[c] not in (None, ""):
            return c
    return None


# ---------------------------------------------------------------------------
# Generic loaders
# ---------------------------------------------------------------------------

def load_existing_chat(path: str) -> list:
    if not os.path.exists(path):
        print(f"  [skip] {path} not found.")
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "<user>" in line and "<assistant>" in line and "<eos>" in line:
                out.append(line)
    print(f"  Loaded {len(out):,} examples from {path}")
    return out


def load_hf_instruction(
    hf_id, *, config_name=None, split="train", max_length=300,
    sample_n=None, instr_col=None, output_col=None, input_col=None,
    label=None,
) -> list:
    """
    Instruction/output style loader with column auto-detection and optional
    streaming sample (for huge datasets).
    """
    load_dataset = _try_import_datasets()
    if load_dataset is None:
        return []
    label = label or hf_id
    streaming = bool(sample_n)
    print(f"  Loading {label}" + (f" (stream sample {sample_n:,})" if streaming else " (full)") + " ...")
    try:
        ds = load_dataset(hf_id, config_name, split=split, streaming=streaming)
        if streaming:
            ds = ds.shuffle(seed=42, buffer_size=10000).take(sample_n)
    except Exception as e:
        print(f"  [error] {label}: {e}")
        return []

    out, skipped = [], 0
    for row in ds:
        ic = instr_col or _pick(row, _INSTR_COLS)
        oc = output_col or _pick(row, _OUTPUT_COLS)
        if not ic or not oc:
            skipped += 1
            continue
        instr = clean_text(row.get(ic))
        ans = clean_text(row.get(oc))
        if not instr or not ans:
            skipped += 1
            continue
        ipc = input_col or _pick(row, _INPUT_COLS)
        if ipc and ipc != ic:
            extra = clean_text(row.get(ipc))
            if extra and extra.lower() not in instr.lower():
                instr = f"{instr}\n\n{extra}"
        line = format_qa(instr, ans)
        if word_count(line) > max_length:
            skipped += 1
            continue
        out.append(line)
    print(f"  -> {len(out):,} examples from {label}  ({skipped} skipped)")
    return out


def load_gsm8k_cot(hf_id, *, config_name="main", split="train",
                   max_length=400, sample_n=None, label=None) -> list:
    """
    GSM8K-style reasoning -> <think> block. Works for openai/gsm8k and
    malhajar/gsm8k-tr (columns: question, answer with `<<calc>>` and `#### final`).
    """
    load_dataset = _try_import_datasets()
    if load_dataset is None:
        return []
    label = label or hf_id
    print(f"  Loading {label} (reasoning/CoT) ...")
    try:
        ds = load_dataset(hf_id, config_name, split=split)
    except Exception as e:
        print(f"  [error] {label}: {e}")
        return []

    out, skipped = [], 0
    for row in ds:
        q = clean_text(row.get("question", ""))
        a = row.get("answer", "") or ""
        if not q or not a:
            skipped += 1
            continue
        if "####" in a:
            reasoning, final = a.split("####", 1)
        else:
            reasoning, final = a, a
        reasoning = clean_text(re.sub(r"<<.*?>>", "", reasoning))  # drop calculator annotations
        final = clean_text(final)
        if not reasoning or not final:
            skipped += 1
            continue
        line = format_cot(q, reasoning, final)
        if word_count(line) > max_length:
            skipped += 1
            continue
        out.append(line)
        if sample_n and len(out) >= sample_n:
            break
    print(f"  -> {len(out):,} reasoning examples from {label}  ({skipped} skipped)")
    return out


def load_hf_conversations(hf_id, *, config_name=None, split="train",
                          max_length=400, sample_n=None, label=None) -> list:
    """
    Multi-turn loader for ShareGPT-style ('conversations': [{from,value}]) or
    chat-style ('messages': [{role,content}]) datasets -> a single multi-turn line.
    """
    load_dataset = _try_import_datasets()
    if load_dataset is None:
        return []
    label = label or hf_id
    streaming = bool(sample_n)
    print(f"  Loading {label}" + (f" (stream sample {sample_n:,})" if streaming else " (full)") + " ...")
    try:
        ds = load_dataset(hf_id, config_name, split=split, streaming=streaming)
        if streaming:
            ds = ds.shuffle(seed=42, buffer_size=10000).take(sample_n)
    except Exception as e:
        print(f"  [error] {label}: {e}")
        return []

    def turns_of(row):
        conv = row.get("conversations") or row.get("messages") or row.get("conversation")
        if not isinstance(conv, list):
            return []
        parsed = []
        for m in conv:
            if not isinstance(m, dict):
                return []
            role = m.get("from") or m.get("role") or ""
            text = m.get("value") or m.get("content") or ""
            role = role.lower()
            if role in ("human", "user", "prompter"):
                parsed.append(("user", text))
            elif role in ("gpt", "assistant", "chatgpt", "model"):
                parsed.append(("assistant", text))
            # skip 'system' turns
        return parsed

    out, skipped = [], 0
    for row in ds:
        turns = turns_of(row)
        if len(turns) < 2:
            skipped += 1
            continue
        parts = []
        for role, text in turns:
            text = clean_text(text)
            if not text:
                continue
            parts.append(f"<{'user' if role == 'user' else 'assistant'}> {text}")
        if len(parts) < 2:
            skipped += 1
            continue
        line = " ".join(parts) + " <eos>"
        if word_count(line) > max_length:
            skipped += 1
            continue
        out.append(line)
    print(f"  -> {len(out):,} multi-turn examples from {label}  ({skipped} skipped)")
    return out


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------
# Each source: a callable producing a list of formatted lines, tagged by lang.

def build_sources(args):
    L = args.max_length
    tr_n = None if args.everything else args.sample_tr
    en_n = None if args.everything else args.sample_en
    sources = []  # (lang, name, callable)

    # ---------------- Turkish ----------------
    if not args.english_only:
        sources += [
            ("tr", "merve/turkish_instructions",
             lambda: load_hf_instruction("merve/turkish_instructions", max_length=L,
                                         instr_col="talimat", output_col="çıktı", input_col="giriş")),
            ("tr", "atasoglu/databricks-dolly-15k-tr",
             lambda: load_hf_instruction("atasoglu/databricks-dolly-15k-tr", max_length=L)),
            ("tr", "TFLai/Turkish-Alpaca",
             lambda: load_hf_instruction("TFLai/Turkish-Alpaca", max_length=L)),
            ("tr", "malhajar/gsm8k-tr (CoT)",
             lambda: load_gsm8k_cot("malhajar/gsm8k-tr", config_name="main", max_length=500)),
            ("tr", "malhajar/OpenOrca-tr",
             lambda: load_hf_instruction("malhajar/OpenOrca-tr", max_length=L, sample_n=tr_n,
                                         instr_col="question-turkish", output_col="response-turkish")),
            ("tr", "turkish-nlp-suite/InstrucTurca",
             lambda: load_hf_instruction("turkish-nlp-suite/InstrucTurca", max_length=L, sample_n=tr_n)),
        ]

    # ---------------- English ----------------
    if not args.turkish_only:
        sources += [
            ("en", "databricks/dolly-15k",
             lambda: load_hf_instruction("databricks/dolly-15k", max_length=L,
                                         instr_col="instruction", output_col="response", input_col="context")),
            ("en", "yahma/alpaca-cleaned",
             lambda: load_hf_instruction("yahma/alpaca-cleaned", max_length=L)),
            ("en", "openai/gsm8k (CoT)",
             lambda: load_gsm8k_cot("openai/gsm8k", config_name="main", max_length=500)),
            ("en", "teknium/OpenHermes-2.5",
             lambda: load_hf_conversations("teknium/OpenHermes-2.5", max_length=L, sample_n=en_n)),
            ("en", "HuggingFaceH4/ultrachat_200k",
             lambda: load_hf_conversations("HuggingFaceH4/ultrachat_200k", split="train_sft",
                                           max_length=L, sample_n=en_n)),
        ]
    return sources


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Prepare clean EN+TR training data for GoktugGPT")
    p.add_argument("--max-length", type=int, default=300, help="Skip examples longer than N words")
    p.add_argument("--val-fraction", type=float, default=0.02)
    p.add_argument("--sample-tr", type=int, default=200_000,
                   help="Max examples to stream from each huge Turkish set")
    p.add_argument("--sample-en", type=int, default=200_000,
                   help="Max examples to stream from each huge English set")
    p.add_argument("--turkish-only", action="store_true")
    p.add_argument("--english-only", action="store_true")
    p.add_argument("--everything", action="store_true",
                   help="Pull the FULL big datasets (tens of GB / millions of rows)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-prefix", type=str, default="data/train_clean")
    args = p.parse_args()

    random.seed(args.seed)
    os.makedirs("data", exist_ok=True)

    print("\n" + "=" * 64)
    print("  GoktugGPT — Data Preparation (EN + TR)")
    print("=" * 64)

    all_examples, by_lang = [], {"tr": 0, "en": 0, "seed": 0}

    # Existing synthetic/seed data (already formatted, keep <think> blocks)
    print("\n[seed] existing synthetic data")
    for path in ("data/train_chat.txt", "data/train_clean.txt"):
        seed = load_existing_chat(path) if path != args.out_prefix + ".txt" else []
        all_examples += seed
        by_lang["seed"] += len(seed)

    sources = build_sources(args)
    for i, (lang, name, fn) in enumerate(sources, 1):
        print(f"\n[{i}/{len(sources)}] {name}  [{lang}]")
        try:
            ex = fn()
        except Exception as e:
            print(f"  [error] {name}: {e}")
            ex = []
        all_examples += ex
        by_lang[lang] += len(ex)

    if not all_examples:
        print("\n[error] No examples collected. Aborting.")
        sys.exit(1)

    # De-duplicate + shuffle
    before = len(all_examples)
    all_examples = list(dict.fromkeys(all_examples))
    random.shuffle(all_examples)

    split_at = max(1, int(len(all_examples) * (1 - args.val_fraction)))
    train_examples = all_examples[:split_at]
    val_examples = all_examples[split_at:]

    train_path = args.out_prefix + ".txt"
    val_path = args.out_prefix.replace("train", "val") + ".txt"
    if val_path == train_path:
        val_path = args.out_prefix + "_val.txt"

    with open(train_path, "w", encoding="utf-8") as f:
        f.write(f"# GoktugGPT clean training data — TR:{by_lang['tr']:,} EN:{by_lang['en']:,} seed:{by_lang['seed']:,}\n")
        for line in train_examples:
            f.write(line + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        f.write(f"# GoktugGPT validation data — {len(val_examples):,} examples\n")
        for line in val_examples:
            f.write(line + "\n")

    print("\n" + "=" * 64)
    print("  Done!")
    print(f"  Turkish : {by_lang['tr']:,}")
    print(f"  English : {by_lang['en']:,}")
    print(f"  Seed    : {by_lang['seed']:,}")
    print(f"  Dedup   : {before:,} -> {len(all_examples):,}")
    print(f"  Train   : {len(train_examples):,} -> {train_path}")
    print(f"  Val     : {len(val_examples):,} -> {val_path}")
    print("=" * 64)
    print("\nTrain (XL on your RTX 5090):")
    print(f"  python train.py --config xl --data {train_path} --val-data {val_path}")
    print()


if __name__ == "__main__":
    main()
