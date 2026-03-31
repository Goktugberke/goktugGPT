"""
Dataset and data-loading utilities for goktugGPT.

Training data format (data/train.txt):
  • One conversation example per line (or multi-line blocks separated by blank lines)
  • Each example follows the chat template:
      <user> ... <assistant> <think> ... </think> ... <eos>
  • Lines starting with '#' are comments and are skipped

The dataset tokenises everything into a flat stream of token IDs.
During training we use a "sliding window" approach: the input is
tokens[0..T-1] and the target is tokens[1..T], so the model predicts
the next token at every position.

Padding: sequences shorter than block_size are padded to block_size.
         Padded positions receive target index -1 so they are ignored
         by the cross-entropy loss.
"""

import os
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from ..tokenizer import BPETokenizer


class ConversationDataset(Dataset):
    """
    Token-level language modelling dataset.

    Each sample is a (input_ids, target_ids) pair of length block_size.
    input_ids[i]  → model input at position i
    target_ids[i] → expected output at position i  (= input_ids[i+1])

    Pad positions in target are set to -1 (ignored by CrossEntropyLoss).
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: BPETokenizer,
        block_size: int = 512,
        split: str = "train",
        val_fraction: float = 0.05,
    ):
        self.block_size = block_size
        self.tokenizer = tokenizer

        raw_text = self._load_text(file_path)
        all_ids = tokenizer.encode(raw_text)

        # Split into train / val
        split_at = max(1, int(len(all_ids) * (1 - val_fraction)))
        if split == "train":
            self.data = all_ids[:split_at]
        else:
            self.data = all_ids[split_at:]

        # Build non-overlapping chunks of length block_size + 1
        # (the extra token is used as the final target label)
        self.chunks: List[List[int]] = []
        for i in range(0, len(self.data) - block_size, block_size):
            self.chunks.append(self.data[i: i + block_size + 1])

        # If last chunk is too short, pad it
        remainder = self.data[len(self.chunks) * block_size:]
        if len(remainder) > 1:
            pad_id = tokenizer.token_to_id("<pad>")
            chunk = remainder + [pad_id] * (block_size + 1 - len(remainder))
            self.chunks.append(chunk)

        print(
            f"Dataset ({split}): {len(self.data)} tokens, "
            f"{len(self.chunks)} samples of block_size={block_size}"
        )

    def _load_text(self, path: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Training data not found: {path}")
        lines = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    lines.append(line)
        return " ".join(lines)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.chunks[idx]
        pad_id = self.tokenizer.token_to_id("<pad>")

        x = chunk[:-1]  # input:  positions 0..T-1
        y = chunk[1:]   # target: positions 1..T

        # Replace pad token positions in targets with -1
        y = [-1 if t == pad_id else t for t in y]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate a list of (x, y) tensors into (B, T) batches.
    Pads sequences to the longest sequence in the batch if needed.
    """
    xs, ys = zip(*batch)
    # All same length (block_size) so just stack
    return torch.stack(xs), torch.stack(ys)


def build_dataloaders(
    file_path: str,
    tokenizer: BPETokenizer,
    block_size: int = 512,
    batch_size: int = 8,
    val_fraction: float = 0.05,
    num_workers: int = 0,
    val_file_path: str = None,
) -> Tuple[DataLoader, DataLoader]:
    """Convenience function to create train and validation DataLoaders.

    Args:
        val_file_path: Optional path to a dedicated validation file. When
                       provided the validation set is loaded from this file
                       instead of being split from file_path.  This avoids
                       the model seeing validation examples during training.
    """
    train_ds = ConversationDataset(
        file_path, tokenizer, block_size,
        split="train", val_fraction=val_fraction if val_file_path is None else 0.0,
    )

    if val_file_path is not None:
        # Use a dedicated validation file — split="train" loads the whole file
        val_ds = ConversationDataset(val_file_path, tokenizer, block_size, split="train", val_fraction=0.0)
        print(f"Validation data loaded from separate file: {val_file_path}")
    else:
        val_ds = ConversationDataset(file_path, tokenizer, block_size, "val", val_fraction)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    return train_dl, val_dl
