#!/usr/bin/env python
"""Combine results/baseline, results/gan_bert (vanilla), and
results/gan_bert_improved into one comparison table for the thesis."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"

MODEL_LABELS = {"specter2": "SPECTER2", "scibert": "SciBERT", "xlnet": "XLNet"}
DATASET_LABELS = {"scicite": "SciCite", "acl-arc": "ACL-ARC", "3C": "3C"}
VARIANTS = {
    "baseline": "No-GAN baseline",
    "gan_bert": "GAN-BERT (vanilla)",
    "gan_bert_improved": "GAN-BERT (improved)",
}


def collect(variant_dir: str) -> list[dict]:
    root = RESULTS_ROOT / variant_dir
    rows = []
    if not root.exists():
        return rows
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name == "logs":
            continue
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.endswith("_seed123"):
                continue
            metrics_path = model_dir / "test_metrics.json"
            if not metrics_path.exists():
                continue
            metrics = json.loads(metrics_path.read_text())
            rows.append(
                {
                    "dataset": DATASET_LABELS.get(dataset_dir.name, dataset_dir.name),
                    "model": MODEL_LABELS.get(model_dir.name, model_dir.name),
                    "variant": VARIANTS[variant_dir],
                    "accuracy": round(metrics["accuracy"], 4),
                    "f1_macro": round(metrics["f1_macro"], 4),
                }
            )
    return rows


def main() -> None:
    rows = collect("baseline") + collect("gan_bert") + collect("gan_bert_improved")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_ROOT / "comparison.csv", index=False)

    pivot_f1 = df.pivot_table(index=["dataset", "model"], columns="variant", values="f1_macro")
    variant_order = [v for v in VARIANTS.values() if v in pivot_f1.columns]
    pivot_f1 = pivot_f1[variant_order]
    pivot_acc = df.pivot_table(index=["dataset", "model"], columns="variant", values="accuracy")[variant_order]

    lines = [
        "# Baseline vs. GAN-BERT (vanilla) vs. GAN-BERT (improved)\n",
        "## F1-macro\n",
        pivot_f1.to_markdown(),
        "\n## Accuracy\n",
        pivot_acc.to_markdown(),
    ]
    (RESULTS_ROOT / "comparison.md").write_text("\n".join(lines) + "\n")
    print(pivot_f1.to_string())


if __name__ == "__main__":
    main()
