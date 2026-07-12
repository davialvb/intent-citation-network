from __future__ import annotations

import datetime
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the torch device (cuda if available and preferred, else cpu)."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def format_time(elapsed_seconds: float) -> str:
    """Format seconds as hh:mm:ss."""
    elapsed_rounded = int(round(elapsed_seconds))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_transformer_representation(
    model_outputs: Any, attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Pick a single vector representation per input for GAN-BERT.

    Prefers pooler_output when available (BERT-like models: SciBERT, SPECTER2).
    Otherwise gathers each row's last real (non-padded) token from
    last_hidden_state, using attention_mask. This is required for models like
    XLNet, which have no pooler and whose tokenizer left-pads (so the true
    <cls> token sits at the last real-token position, not index 0).
    """
    # HuggingFace BaseModelOutputWithPoolingAndCrossAttentions supports both attribute and tuple access.
    pooler = getattr(model_outputs, "pooler_output", None)
    if pooler is not None:
        return pooler
    last_hidden = getattr(model_outputs, "last_hidden_state", None)
    if last_hidden is None:
        # Fallback to tuple indexing (last_hidden_state is usually 0)
        last_hidden = model_outputs[0]

    if attention_mask is None:
        # No mask available: assume right-padding and take position 0 (CLS).
        return last_hidden[:, 0, :]

    # Index of the last position where attention_mask == 1, per row, regardless
    # of whether padding is on the left (XLNet) or right (BERT-like).
    seq_len = attention_mask.size(1)
    flipped = attention_mask.flip(dims=[1])
    last_idx = seq_len - 1 - flipped.argmax(dim=1)
    batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
    return last_hidden[batch_idx, last_idx, :]


def freeze_transformer_layers(transformer: Any, num_trainable_layers: int) -> Dict[str, int]:
    """Freeze all transformer parameters except the last `num_trainable_layers`
    encoder blocks (+ pooler, if present).

    Supports BERT-like models (`transformer.encoder.layer`, e.g. SciBERT,
    SPECTER2) and XLNet (`transformer.layer`). If `num_trainable_layers` is
    >= the total number of encoder blocks, this is a full fine-tune: nothing
    is frozen, including embeddings (matches the reference paper's setup).
    """
    if hasattr(transformer, "encoder") and hasattr(transformer.encoder, "layer"):
        layers = transformer.encoder.layer
    elif hasattr(transformer, "layer"):
        layers = transformer.layer
    else:
        raise ValueError(f"Don't know how to locate encoder layers on {type(transformer).__name__}")

    if num_trainable_layers >= len(layers):
        for p in transformer.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in transformer.parameters())
        return {"trainable_params": trainable, "frozen_params": 0}

    for p in transformer.parameters():
        p.requires_grad = False

    for layer in layers[-num_trainable_layers:]:
        for p in layer.parameters():
            p.requires_grad = True

    pooler = getattr(transformer, "pooler", None)
    if pooler is not None:
        for p in pooler.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in transformer.parameters() if not p.requires_grad)
    return {"trainable_params": trainable, "frozen_params": frozen}


@dataclass(frozen=True)
class SavePaths:
    root: Path
    discriminator_dir: Path
    generator_dir: Path
    transformer_dir: Path
    config_path: Path

    @staticmethod
    def for_run(root: str | Path) -> "SavePaths":
        root = ensure_dir(root)
        return SavePaths(
            root=root,
            discriminator_dir=ensure_dir(root / "discriminator"),
            generator_dir=ensure_dir(root / "generator"),
            transformer_dir=ensure_dir(root / "transformer"),
            config_path=root / "config.json",
        )
