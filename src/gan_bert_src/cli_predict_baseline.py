#!/usr/bin/env python
"""Run inference with a trained no-GAN baseline model (transformer + SimpleClassifier;
see cli_train_baseline.py). Used to build a Kaggle-submission-format CSV for
the 3C Shared Task test set.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from gan_bert.models import SimpleClassifier
from gan_bert.utils import get_device, get_transformer_representation


def _best_checkpoint(run_dir: Path, subdir: str, prefix: str) -> Path:
    candidates = [c for c in glob.glob(str(run_dir / subdir / "*.pth")) if Path(c).name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(f"No '{prefix}*' checkpoints found in {run_dir / subdir}")

    def f1_of(path: str) -> float:
        m = re.search(r"f1_([0-9.]+)\.pth$", path)
        return float(m.group(1)) if m else -1.0

    return Path(max(candidates, key=f1_of))


def main():
    p = argparse.ArgumentParser(description="Predict with a trained no-GAN baseline model.")
    p.add_argument("--run_dir", required=True, help="cli_train_baseline.py output_dir.")
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--text_col", default="text")
    p.add_argument("--id_col", default=None, help="If set, copied through unchanged to the output.")
    p.add_argument("--pred_col", default="predicted_intent")
    p.add_argument("--labels", nargs="+", required=True, help="Real class labels, same order used for training.")
    p.add_argument("--no_cuda", action="store_true")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    device = get_device(prefer_cuda=not args.no_cuda)

    transformer_ckpt = _best_checkpoint(run_dir, "transformer", prefix="transformer_")
    classifier_ckpt = _best_checkpoint(run_dir, "discriminator", prefix="classifier_")
    print(f"Loading transformer checkpoint: {transformer_ckpt.name}")
    print(f"Loading classifier checkpoint: {classifier_ckpt.name}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model_config = AutoConfig.from_pretrained(cfg["model_name"])
    transformer = AutoModel.from_pretrained(cfg["model_name"])
    transformer.load_state_dict(torch.load(transformer_ckpt, map_location="cpu"))

    hidden_size = int(model_config.hidden_size)
    classifier = SimpleClassifier(input_size=hidden_size, num_labels=cfg["num_labels"], dropout_rate=0.1)
    classifier.load_state_dict(torch.load(classifier_ckpt, map_location="cpu"))

    transformer.to(device).eval()
    classifier.to(device).eval()

    df = pd.read_csv(args.input_csv)
    texts = df[args.text_col].astype(str).tolist()

    preds = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoding = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=cfg["max_seq_length"],
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attn_mask = encoding["attention_mask"].to(device)

            outputs = transformer(input_ids, attention_mask=attn_mask)
            rep = get_transformer_representation(outputs, attention_mask=attn_mask)
            logits = classifier(rep)
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())

    out_cols = {}
    if args.id_col:
        out_cols[args.id_col] = df[args.id_col]
    out_cols[args.pred_col] = [args.labels[p] for p in preds]
    pd.DataFrame(out_cols).to_csv(args.output_csv, index=False)
    print(f"Wrote predictions for {len(df)} rows to {args.output_csv}")
    print(pd.Series([args.labels[p] for p in preds]).value_counts())


if __name__ == "__main__":
    main()
