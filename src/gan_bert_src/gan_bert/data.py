from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def label_str2int(label_names: list[str]) -> dict[str, int]:
    return {label: i for i, label in enumerate(label_names)}


class TextDataset(Dataset):
    """Tokenizes text examples on-the-fly."""

    def __init__(
        self,
        texts: list[str],
        label_masks: np.ndarray,
        labels: Optional[list[int]] = None,
        max_seq_length: int = 160,
        model_name: str = "allenai/scibert_scivocab_uncased",
    ):
        self.texts = texts
        self.labels = labels
        self.label_masks = label_masks
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label_masks": torch.tensor(bool(self.label_masks[idx])),
            "texts": self.texts[idx],
        }
        # Include token_type_ids only if the tokenizer/model uses them
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def make_train_dataloader(
    labeled_examples: pd.DataFrame,
    unlabeled_examples: Optional[pd.DataFrame],
    max_seq_length: int,
    model_name: str,
    col_text: str = "text",
    label_col: str = "intent_int",
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    texts = labeled_examples[col_text].astype(str).tolist()
    labels = labeled_examples[label_col].astype(int).tolist()

    train_label_masks = np.ones(len(labeled_examples), dtype=bool)

    if unlabeled_examples is not None and unlabeled_examples[col_text].notnull().all():
        texts = texts + unlabeled_examples[col_text].astype(str).tolist()
        labels = labels + unlabeled_examples[label_col].astype(int).tolist()

        tmp_masks = np.zeros(len(unlabeled_examples), dtype=bool)
        train_label_masks = np.concatenate([train_label_masks, tmp_masks])

    dataset = TextDataset(
        texts=texts,
        labels=labels,
        label_masks=train_label_masks,
        max_seq_length=max_seq_length,
        model_name=model_name,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def make_eval_dataloader(
    examples: pd.DataFrame,
    max_seq_length: int,
    model_name: str,
    col_text: str = "text",
    label_col: str = "intent_int",
    batch_size: int = 32,
    shuffle: bool = False,
) -> DataLoader:
    texts = examples[col_text].astype(str).tolist()
    labels = examples[label_col].astype(int).tolist()

    label_masks = np.ones(len(examples), dtype=bool)
    dataset = TextDataset(
        texts=texts,
        labels=labels,
        label_masks=label_masks,
        max_seq_length=max_seq_length,
        model_name=model_name,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
