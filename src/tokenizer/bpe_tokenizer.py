"""
BPE (Byte Pair Encoding) Tokenizer — built entirely from scratch.

How it works:
  1. Start with a character-level vocabulary where every word is split
     into individual characters + a special </w> end-of-word marker.
  2. Repeatedly find the most frequent adjacent pair of symbols and
     merge them into a single new symbol.
  3. After `vocab_size - base` merges the vocabulary is fixed.
  4. Encoding new text: apply the learned merge rules in order.
  5. Decoding: reverse the process — rejoin symbols, strip </w>.
"""

import json
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple


class BPETokenizer:
    # End-of-word marker appended to the last char of every word during training
    EOW = "</w>"

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size

        # Bidirectional maps
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}

        # Ordered list of BPE merge rules: (a, b) -> a+b
        self.merges: List[Tuple[str, str]] = []

        # Set for O(1) merge lookup during encoding
        self._merge_set: Dict[Tuple[str, str], int] = {}

        self.special_tokens: List[str] = []
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _word_to_symbols(self, word: str) -> Tuple[str, ...]:
        """Split word into character symbols with EOW on last char."""
        chars = list(word)
        chars[-1] = chars[-1] + self.EOW
        return tuple(chars)

    def _get_pair_freqs(
        self, word_freqs: Dict[Tuple[str, ...], int]
    ) -> Counter:
        pairs: Counter = Counter()
        for symbols, freq in word_freqs.items():
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _apply_merge(
        self,
        pair: Tuple[str, str],
        word_freqs: Dict[Tuple[str, ...], int],
    ) -> Dict[Tuple[str, ...], int]:
        """Merge every occurrence of `pair` in the word vocabulary."""
        a, b = pair
        merged = a + b
        new_freqs: Dict[Tuple[str, ...], int] = {}
        for symbols, freq in word_freqs.items():
            new_syms: List[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_syms.append(merged)
                    i += 2
                else:
                    new_syms.append(symbols[i])
                    i += 1
            new_freqs[tuple(new_syms)] = freq
        return new_freqs

    def train(self, texts: List[str], special_tokens: Optional[List[str]] = None):
        """
        Train BPE on a list of strings.

        Args:
            texts:          Training corpus (list of lines / sentences).
            special_tokens: Tokens that must be kept verbatim (e.g. <pad>).
        """
        if special_tokens is None:
            special_tokens = []
        self.special_tokens = list(special_tokens)

        print("Building initial vocabulary...")

        # --- Step 1: build word-frequency table (skip special tokens) ---
        special_re = (
            re.compile(
                r"(" + "|".join(re.escape(s) for s in special_tokens) + r")"
            )
            if special_tokens
            else None
        )

        raw_word_freqs: Counter = Counter()
        for text in texts:
            # Split out special tokens so they are not corrupted
            parts = special_re.split(text) if special_re else [text]
            for part in parts:
                if part in special_tokens:
                    continue
                for word in re.findall(r"\S+", part):
                    raw_word_freqs[word] += 1

        # --- Step 2: symbolise every word ---
        word_freqs: Dict[Tuple[str, ...], int] = {}
        char_vocab: set = set()
        for word, freq in raw_word_freqs.items():
            syms = self._word_to_symbols(word)
            word_freqs[syms] = word_freqs.get(syms, 0) + freq
            char_vocab.update(syms)

        # --- Step 3: build initial token list ---
        # Order: special tokens first, then sorted chars
        token_list: List[str] = list(special_tokens) + sorted(char_vocab)
        self.token2id = {tok: i for i, tok in enumerate(token_list)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}

        n_merges = self.vocab_size - len(token_list)
        print(
            f"Initial vocab: {len(token_list)} tokens. "
            f"Target: {self.vocab_size} (+{n_merges} merges)."
        )

        # --- Step 4: BPE merge loop ---
        self.merges = []
        self._merge_set = {}
        for step in range(n_merges):
            pairs = self._get_pair_freqs(word_freqs)
            if not pairs:
                break
            best_pair, best_freq = pairs.most_common(1)[0]
            if best_freq < 2:
                break

            word_freqs = self._apply_merge(best_pair, word_freqs)
            merged = best_pair[0] + best_pair[1]

            self.merges.append(best_pair)
            self._merge_set[best_pair] = step

            if merged not in self.token2id:
                idx = len(self.token2id)
                self.token2id[merged] = idx
                self.id2token[idx] = merged

            if (step + 1) % 500 == 0:
                print(f"  Step {step + 1}/{n_merges}  vocab={len(self.token2id)}")

        self.vocab_size = len(self.token2id)
        self._trained = True
        print(f"Training complete. Final vocab size: {self.vocab_size}")

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode_word(self, word: str) -> List[int]:
        """Apply BPE merges to a single word and return token IDs."""
        syms = list(self._word_to_symbols(word))

        # Greedily apply merges in their training order
        while len(syms) > 1:
            # Find the earliest merge among all adjacent pairs
            best_idx = None
            best_pos = -1
            for i in range(len(syms) - 1):
                pair = (syms[i], syms[i + 1])
                rank = self._merge_set.get(pair)
                if rank is not None:
                    if best_idx is None or rank < best_idx:
                        best_idx = rank
                        best_pos = i
            if best_idx is None:
                break
            merged = syms[best_pos] + syms[best_pos + 1]
            syms = syms[:best_pos] + [merged] + syms[best_pos + 2:]

        ids: List[int] = []
        unk_id = self.token2id.get(self.special_tokens[3] if len(self.special_tokens) > 3 else "<unk>", 0)
        for sym in syms:
            ids.append(self.token2id.get(sym, unk_id))
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode a string into a list of token IDs."""
        ids: List[int] = []

        if add_bos and self.special_tokens:
            bos = self.special_tokens[1] if len(self.special_tokens) > 1 else "<bos>"
            ids.append(self.token2id.get(bos, 0))

        # Split on special tokens so they survive verbatim
        if self.special_tokens:
            pattern = re.compile(
                r"(" + "|".join(re.escape(s) for s in self.special_tokens) + r")"
            )
            parts = pattern.split(text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                if part in self.token2id:
                    ids.append(self.token2id[part])
            else:
                # Encode each whitespace-delimited word
                words = re.findall(r"\S+", part)
                for word in words:
                    ids.extend(self._encode_word(word))
                # Preserve single spaces as a space before next token
                # (handled implicitly by </w> decoding)

        if add_eos and self.special_tokens:
            eos = self.special_tokens[2] if len(self.special_tokens) > 2 else "<eos>"
            ids.append(self.token2id.get(eos, 0))

        return ids

    def encode_batch(self, texts: List[str], **kwargs) -> List[List[int]]:
        return [self.encode(t, **kwargs) for t in texts]

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Convert token IDs back to a human-readable string."""
        tokens: List[str] = []
        for i in ids:
            tok = self.id2token.get(i, "")
            if skip_special_tokens and tok in self.special_tokens:
                continue
            tokens.append(tok)

        # Join all tokens; EOW marks a word boundary → replace with space
        text = "".join(tokens)
        text = text.replace(self.EOW, " ")
        text = re.sub(r" +", " ", text).strip()
        return text

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        data = {
            "vocab_size": self.vocab_size,
            "token2id": self.token2id,
            "merges": [list(pair) for pair in self.merges],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved → {path}")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.token2id = data["token2id"]
        tok.id2token = {int(k) if isinstance(k, str) else k: v
                        for k, v in {v: k for k, v in tok.token2id.items()}.items()}
        tok.merges = [tuple(p) for p in data["merges"]]
        tok._merge_set = {tuple(p): i for i, p in enumerate(tok.merges)}
        tok.special_tokens = data.get("special_tokens", [])
        tok._trained = True
        return tok

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.token2id)

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={len(self)}, trained={self._trained})"

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.token2id)

    def token_to_id(self, token: str) -> int:
        return self.token2id.get(token, self.token2id.get("<unk>", 0))

    def id_to_token(self, idx: int) -> str:
        return self.id2token.get(idx, "<unk>")
