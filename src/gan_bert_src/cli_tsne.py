#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from transformers import AutoModel

from gan_bert.data import TextDataset
from gan_bert.utils import get_device, get_transformer_representation, set_seed

try:
    import plotly.express as px
except ImportError:
    px = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="t-SNE visualization for transformer representations.")
    p.add_argument("--csv", required=True, help="CSV with text + (optional) label column.")
    p.add_argument("--text_col", default="text")
    p.add_argument("--label_col", default="label")
    p.add_argument("--labels", nargs="*", default=None, help="Optional label list for consistent ordering.")

    # Model loading
    p.add_argument("--model_name", default=None, help="Base model name (used if no checkpoint provided).")
    p.add_argument("--transformer_checkpoint", default=None, help="Path to a saved transformer .pth file.")
    p.add_argument("--model_dir", default=None, help="Training output dir from cli_train (auto-picks best transformer checkpoint).")

    # t-SNE settings
    p.add_argument("--n_components", type=int, choices=[2, 3], default=2)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--learning_rate", type=str, default="auto")
    p.add_argument("--n_iter", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)

    # Sampling / output
    p.add_argument("--max_points", type=int, default=1000, help="Subsample to at most this many points.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--output", required=True, help="Output file: .html (plotly) or .csv (embeddings).")

    return p.parse_args()


def _pick_best_transformer_checkpoint(model_dir: Path) -> Path:
    tdir = model_dir / "transformer"
    if not tdir.exists():
        raise SystemExit(f"No transformer/ dir found under: {model_dir}")

    ckpts = list(tdir.glob("transformer_epoch*_f1_*.pth"))
    if not ckpts:
        # fallback: any .pth
        ckpts = list(tdir.glob("*.pth"))
    if not ckpts:
        raise SystemExit(f"No transformer checkpoints found under: {tdir}")

    # Prefer the highest f1 in the filename.
    def score(p: Path) -> float:
        m = re.search(r"_f1_([0-9.]+)\.pth$", p.name)
        return float(m.group(1)) if m else -1.0

    ckpts.sort(key=score, reverse=True)
    return ckpts[0]


@torch.no_grad()
def _compute_representations(
    df: pd.DataFrame,
    model_name: str,
    checkpoint: Optional[Path],
    max_seq_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    ds = TextDataset(
        df=df,
        label_list=["__dummy__"],
        unknown_label="__dummy__",
        model_name=model_name,
        max_seq_length=max_seq_length,
        text_col="__text__",
        label_col="__label__",
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    transformer = AutoModel.from_pretrained(model_name)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        transformer.load_state_dict(state, strict=False)

    transformer.to(device)
    transformer.eval()

    reps: List[torch.Tensor] = []
    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        outputs = transformer(input_ids, attention_mask=attn_mask)
        rep = get_transformer_representation(outputs)
        reps.append(rep.detach().cpu())

    return torch.cat(reps, dim=0).numpy()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if args.text_col not in df.columns:
        raise SystemExit(f"Missing text column '{args.text_col}' in CSV.")
    # Prepare dummy columns to satisfy TextDataset API without label mapping headaches
    work = pd.DataFrame({
        "__text__": df[args.text_col].astype(str),
        "__label__": "__dummy__",
    })

    # Optional labels for hover/color
    labels = None
    if args.label_col in df.columns:
        labels = df[args.label_col].astype(str).copy()
        if args.labels:
            # keep only known labels first
            known = set(args.labels)
            labels = labels.where(labels.isin(known), other="__other__")

    if args.model_dir:
        ckpt = _pick_best_transformer_checkpoint(Path(args.model_dir))
        if args.model_name is None:
            # fall back to common default; if you want the exact model, pass --model_name
            args.model_name = "allenai/scibert_scivocab_uncased"
        checkpoint = ckpt
    else:
        checkpoint = Path(args.transformer_checkpoint) if args.transformer_checkpoint else None
        if args.model_name is None:
            raise SystemExit("Provide --model_name (and optionally --transformer_checkpoint) or use --model_dir.")

    device = get_device()

    # Subsample
    if len(work) > args.max_points:
        work = work.sample(n=args.max_points, random_state=args.seed).reset_index(drop=True)
        if labels is not None:
            labels = labels.loc[work.index].reset_index(drop=True)

    reps = _compute_representations(
        df=work,
        model_name=args.model_name,
        checkpoint=checkpoint,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        device=device,
    )

    tsne = TSNE(
        n_components=args.n_components,
        random_state=args.seed,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        init="pca",
    )
    emb = tsne.fit_transform(reps)

    # Save embeddings CSV always (useful for other plotting tools)
    cols = ["x", "y"] if args.n_components == 2 else ["x", "y", "z"]
    out_df = pd.DataFrame(emb, columns=cols)
    out_df["text"] = work["__text__"].values
    if labels is not None:
        out_df["label"] = labels.values

    if out_path.suffix.lower() == ".csv":
        out_df.to_csv(out_path, index=False)
        print(f"Wrote: {out_path}")
        return

    if out_path.suffix.lower() == ".html":
        if px is None:
            raise SystemExit("Plotly is not installed. Install with: pip install plotly")
        if args.n_components == 2:
            fig = px.scatter(out_df, x="x", y="y", color="label" if labels is not None else None, hover_data=["text"])
        else:
            fig = px.scatter_3d(out_df, x="x", y="y", z="z", color="label" if labels is not None else None, hover_data=["text"])
        fig.update_layout(title="t-SNE of transformer representations")
        fig.write_html(str(out_path), include_plotlyjs="cdn")
        print(f"Wrote: {out_path}")
        return

    raise SystemExit("Output must end with .html or .csv")


if __name__ == "__main__":
    main()
