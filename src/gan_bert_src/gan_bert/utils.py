from __future__ import annotations

import datetime
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


# cuBLAS reads this when it initializes its workspace, which happens on the first
# GPU matmul. Setting it at import time (before any CUDA work) is what makes
# torch.use_deterministic_algorithms() usable for the cuBLAS-backed ops; setting
# it later has no effect for the current process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed every RNG the training loop draws from.

    With ``deterministic=True`` this also pins the nondeterministic GPU kernels,
    so that two runs of the same configuration on the same hardware produce
    identical results. Without it, cuDNN algorithm selection and atomic-add
    reduction order vary between runs and a fixed seed does NOT pin a
    trajectory -- an effect measured at ~0.02 macro-F1 on this project, large
    enough to swamp the differences some ablations report.

    ``warn_only=True`` keeps the run alive if an op has no deterministic
    implementation (it warns instead of raising); check the warnings if exact
    reproducibility matters for a given configuration.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        # The fused attention kernels accumulate their backward pass with
        # atomics, so their reduction order -- and therefore the gradient --
        # varies between runs. Restrict scaled-dot-product attention to the
        # math backend, which is deterministic. This is the one remaining
        # source of run-to-run drift once cuDNN and cuBLAS are pinned.
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)


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


def best_checkpoint(directory: str | Path, prefix: str) -> Optional[Path]:
    """Highest-validation-F1 checkpoint written by the training CLIs, or None.

    Checkpoints are named ``<prefix>epoch<N>_f1_<score>.pth``; the score in the
    filename is the validation macro-F1 that triggered the save, so the maximum
    over that field is the best-validation checkpoint.
    """
    directory = Path(directory)
    candidates = list(directory.glob(f"{prefix}*.pth"))
    if not candidates:
        return None

    def f1_of(path: Path) -> float:
        m = re.search(r"_f1_([0-9.]+)\.pth$", path.name)
        return float(m.group(1)) if m else -1.0

    return max(candidates, key=f1_of)


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
