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


def get_transformer_representation(model_outputs: Any) -> torch.Tensor:
    """Pick a single vector representation per input for GAN-BERT.

    Prefers pooler_output when available, else uses CLS token from last_hidden_state.
    """
    # HuggingFace BaseModelOutputWithPoolingAndCrossAttentions supports both attribute and tuple access.
    pooler = getattr(model_outputs, "pooler_output", None)
    if pooler is not None:
        return pooler
    last_hidden = getattr(model_outputs, "last_hidden_state", None)
    if last_hidden is None:
        # Fallback to tuple indexing (last_hidden_state is usually 0)
        last_hidden = model_outputs[0]
    # CLS token representation
    return last_hidden[:, 0, :]


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
