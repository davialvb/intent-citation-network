#!/usr/bin/env python
"""Re-evaluates finished runs on the test set using their best-validation-F1
checkpoint, for runs trained before cli_train.py / cli_train_baseline.py were
changed to do that at the end of training.

Those earlier revisions evaluated the test set with the *final-epoch* weights
still in memory and never reloaded the best checkpoint, which flatters an arm
whose last epoch happens to be its best and penalizes an arm that overfits
within the epoch budget. This script recomputes the headline metrics under the
same model-selection rule for every run whose checkpoints still exist.

For each run directory it writes:
  - test_metrics_best_checkpoint.json  (new, best-validation-F1 weights)
  - test_metrics_final_epoch.json      (a copy of the pre-existing numbers,
                                        only if that file is not already there)
leaving the original test_metrics.json untouched so nothing already cited is
silently rewritten.

Usage:
    uv run python scripts/backfill_best_checkpoint_metrics.py results/verify/*/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoConfig, AutoModel

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "gan_bert_src"))

from gan_bert.data import label_str2int, make_eval_dataloader  # noqa: E402
from gan_bert.models import Discriminator, SimpleClassifier  # noqa: E402
from gan_bert.train_eval import evaluate as gan_evaluate  # noqa: E402
from gan_bert.utils import best_checkpoint, get_device, save_json  # noqa: E402


def _dataset_dir(run_meta: dict) -> Path:
    return REPO / "data" / str(run_meta["dataset_name"]) / "gan_bert"


def evaluate_run(run_dir: Path, device: torch.device) -> dict | None:
    cfg = json.loads((run_dir / "config.json").read_text())
    meta = json.loads((run_dir / "run_meta.json").read_text())

    if cfg.get("use_section_feature"):
        # These runs feed [transformer_rep ; section_embedding] to the head, so
        # re-instantiating it here would need the section column and embedding
        # too. The section variant was not adopted, so it is out of scope.
        print(f"  skip {run_dir}: section-feature run (not re-evaluated here)")
        return None

    t_ckpt = best_checkpoint(run_dir / "transformer", "transformer_")
    if t_ckpt is None:
        print(f"  skip {run_dir}: no transformer checkpoint retained")
        return None

    # A GAN run saves discriminator_*.pth; the no-GAN baseline saves classifier_*.pth.
    d_ckpt = best_checkpoint(run_dir / "discriminator", "discriminator_")
    c_ckpt = best_checkpoint(run_dir / "discriminator", "classifier_")
    if d_ckpt is None and c_ckpt is None:
        print(f"  skip {run_dir}: no head checkpoint retained")
        return None

    test_csv = _dataset_dir(meta) / "test.csv"
    test = pd.read_csv(test_csv)
    # run_meta labels for a GAN run carry a trailing "unknown" sentinel that is
    # not one of the classifier's output classes.
    labels = list(meta["labels"])
    real_labels = labels[:-1] if labels[-1] == "unknown" else labels
    test["intent_int"] = test["intent"].map(label_str2int(labels))

    loader = make_eval_dataloader(
        examples=test,
        max_seq_length=cfg["max_seq_length"],
        model_name=cfg["model_name"],
        col_text="text",
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    transformer = AutoModel.from_pretrained(cfg["model_name"])
    transformer.load_state_dict(torch.load(t_ckpt, map_location="cpu"))
    hidden_size = int(AutoConfig.from_pretrained(cfg["model_name"]).hidden_size)
    transformer.to(device)

    if d_ckpt is not None:
        head = Discriminator(
            input_size=hidden_size,
            hidden_sizes=[cfg["hidden_size"]] * cfg["num_hidden_layers_d"],
            num_labels=len(real_labels),
            dropout_rate=cfg["out_dropout_rate"],
            noise_stddev=cfg["discriminator_noise_stddev"],
        )
        head.load_state_dict(torch.load(d_ckpt, map_location="cpu"))
        head.to(device)
        m = gan_evaluate(loader, transformer, head, device=device, verbose=False)
        ckpt_name = d_ckpt.name
    else:
        head = SimpleClassifier(
            input_size=hidden_size, num_labels=len(real_labels), dropout_rate=cfg.get("dropout", 0.1)
        )
        head.load_state_dict(torch.load(c_ckpt, map_location="cpu"))
        head.to(device)
        # SimpleClassifier returns bare logits; reuse the GAN evaluator by
        # wrapping it so it exposes the (features, logits, probs) triple and one
        # extra trailing column, which that evaluator slices off.
        class _Wrap(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                logits = self.inner(x)
                pad = torch.full((logits.size(0), 1), -1e9, device=logits.device, dtype=logits.dtype)
                padded = torch.cat([logits, pad], dim=1)
                return x, padded, torch.softmax(padded, dim=-1)

        m = gan_evaluate(loader, transformer, _Wrap(head).to(device), device=device, verbose=False)
        ckpt_name = c_ckpt.name

    return {
        "selection": "best_val_f1_checkpoint",
        "epoch": meta.get("best_epoch"),
        "checkpoints": [t_ckpt.name, ckpt_name],
        "accuracy": m["accuracy"],
        "f1_macro": m["f1_macro"],
        "avg_loss": m["avg_loss"],
        "confusion_matrix": m["confusion_matrix"].tolist(),
        "classification_report": m["classification_report"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--no_cuda", action="store_true")
    args = p.parse_args()
    device = get_device(prefer_cuda=not args.no_cuda)

    for run_dir in args.run_dirs:
        run_dir = run_dir.resolve()
        if not (run_dir / "run_meta.json").exists():
            continue
        print(f"[{run_dir.relative_to(REPO)}]")
        out = evaluate_run(run_dir, device)
        if out is None:
            continue

        old_p = run_dir / "test_metrics.json"
        final_p = run_dir / "test_metrics_final_epoch.json"
        if old_p.exists() and not final_p.exists():
            old = json.loads(old_p.read_text())
            if old.get("selection") is None:
                old["selection"] = "final_epoch"
                save_json(old, final_p)

        save_json(out, run_dir / "test_metrics_best_checkpoint.json")
        prev = json.loads(old_p.read_text())["f1_macro"] if old_p.exists() else float("nan")
        print(f"  final-epoch F1={prev:.4f} -> best-checkpoint F1={out['f1_macro']:.4f} (epoch {out['epoch']})")


if __name__ == "__main__":
    main()
