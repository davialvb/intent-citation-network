#!/usr/bin/env python
"""Run inference with a trained GAN-BERT model (transformer + discriminator;
the generator is only needed during adversarial training, per the README).

Loads the best-F1 transformer/discriminator checkpoints from a cli_train.py
output_dir and predicts intent labels for a CSV of citation texts.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from gan_bert.models import Discriminator
from gan_bert.utils import get_device, get_transformer_representation

# Must match src/utils/gan_bert_preprocessing.py's SECTION_VOCAB.
SECTION_VOCAB = [
    "introduction",
    "background",
    "related_work",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "unknown",
]
_SECTION_ALIASES = {"experiments": "results", "experiment": "results", "method": "methods", "methodology": "methods"}


def normalize_section_clean(raw: object) -> str:
    if raw is None or (isinstance(raw, float)) or not str(raw).strip():
        return "unknown"
    s = str(raw).strip().lower()
    s = _SECTION_ALIASES.get(s, s)
    return s if s in SECTION_VOCAB else "unknown"


def _best_checkpoint(run_dir: Path, subdir: str, prefix: str) -> Path:
    candidates = [c for c in glob.glob(str(run_dir / subdir / "*.pth")) if Path(c).name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(f"No '{prefix}*' checkpoints found in {run_dir / subdir}")

    def f1_of(path: str) -> float:
        m = re.search(r"f1_([0-9.]+)\.pth$", path)
        return float(m.group(1)) if m else -1.0

    return Path(max(candidates, key=f1_of))


def main():
    p = argparse.ArgumentParser(description="Predict citation intent with a trained GAN-BERT model.")
    p.add_argument("--run_dir", required=True, help="cli_train.py output_dir (has config.json, transformer/, discriminator/).")
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--text_col", default="citation_text")
    p.add_argument("--pred_col", default="predicted_intent")
    p.add_argument("--labels", nargs="+", required=True, help="Real class labels, in the same order used for training (no trailing 'unknown').")
    p.add_argument("--section_col", default=None, help="Column with the citation's paper section (e.g. 'chapter'). Required if the model was trained with --use_section_feature.")
    p.add_argument("--no_cuda", action="store_true")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    device = get_device(prefer_cuda=not args.no_cuda)
    use_section_feature = bool(cfg.get("use_section_feature"))

    transformer_ckpt = _best_checkpoint(run_dir, "transformer", prefix="transformer_")
    discriminator_ckpt = _best_checkpoint(run_dir, "discriminator", prefix="discriminator_")
    print(f"Loading transformer checkpoint: {transformer_ckpt.name}")
    print(f"Loading discriminator checkpoint: {discriminator_ckpt.name}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model_config = AutoConfig.from_pretrained(cfg["model_name"])
    transformer = AutoModel.from_pretrained(cfg["model_name"])
    transformer.load_state_dict(torch.load(transformer_ckpt, map_location="cpu"))

    hidden_size = int(model_config.hidden_size)
    combined_size = hidden_size

    section_embed = None
    if use_section_feature:
        if not args.section_col:
            raise SystemExit("This model was trained with --use_section_feature; pass --section_col.")
        section_embed_dim = cfg["section_embed_dim"]
        section_embed = nn.Embedding(len(SECTION_VOCAB), section_embed_dim)
        section_embed_ckpt = _best_checkpoint(run_dir, "discriminator", prefix="section_embed_")
        print(f"Loading section embedding checkpoint: {section_embed_ckpt.name}")
        section_embed.load_state_dict(torch.load(section_embed_ckpt, map_location="cpu"))
        section_embed.to(device).eval()
        combined_size = hidden_size + section_embed_dim

    hidden_levels_d = [hidden_size for _ in range(cfg["num_hidden_layers_d"])]
    discriminator = Discriminator(
        input_size=combined_size,
        hidden_sizes=hidden_levels_d,
        num_labels=cfg["num_labels"],
        dropout_rate=cfg["out_dropout_rate"],
        noise_stddev=cfg["discriminator_noise_stddev"],
    )
    discriminator.load_state_dict(torch.load(discriminator_ckpt, map_location="cpu"))

    transformer.to(device).eval()
    discriminator.to(device).eval()

    df = pd.read_csv(args.input_csv)
    texts = df[args.text_col].astype(str).tolist()
    section_ids = None
    if use_section_feature:
        section_map = {s: i for i, s in enumerate(SECTION_VOCAB)}
        section_ids = df[args.section_col].map(normalize_section_clean).map(section_map).tolist()

    preds = []
    confidences = []
    batch_size = 16
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
            if section_embed is not None:
                batch_section_ids = torch.tensor(section_ids[i : i + batch_size], dtype=torch.long, device=device)
                rep = torch.cat([rep, section_embed(batch_section_ids)], dim=-1)
            _, logits, probs = discriminator(rep)
            filtered_logits = logits[:, 0 : cfg["num_labels"]]
            filtered_probs = torch.softmax(filtered_logits, dim=-1)

            batch_preds = torch.argmax(filtered_logits, dim=1).cpu().tolist()
            batch_conf = torch.max(filtered_probs, dim=1).values.cpu().tolist()
            preds.extend(batch_preds)
            confidences.extend(batch_conf)

    df[args.pred_col] = [args.labels[p] for p in preds]
    df[f"{args.pred_col}_confidence"] = [round(c, 4) for c in confidences]
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote predictions for {len(df)} rows to {args.output_csv}")
    print(df[args.pred_col].value_counts())


if __name__ == "__main__":
    main()
