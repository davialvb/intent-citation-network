#!/usr/bin/env python
"""Combine every experiment variant (partial vs. full fine-tuning; no-GAN
baseline vs. vanilla GAN-BERT vs. improved GAN-BERT; in-domain vs.
cross-dataset unlabeled data) into one comparison table for the thesis."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"

MODEL_LABELS = {"specter2": "SPECTER2", "scibert": "SciBERT", "xlnet": "XLNet"}
DATASET_LABELS = {"scicite": "SciCite", "acl-arc": "ACL-ARC", "3C": "3C"}

# variant_dir -> (display name, ordering key)
VARIANTS = {
    "baseline": "No-GAN baseline (partial)",
    "baseline_full": "No-GAN baseline (full)",
    "gan_bert": "GAN-BERT vanilla (partial)",
    "gan_bert_full": "GAN-BERT vanilla (full)",
    "gan_bert_improved": "GAN-BERT improved (partial)",
    "gan_bert_improved_full": "GAN-BERT improved (full)",
    "gan_bert_improved_cross": "GAN-BERT improved (partial, cross-dataset unlabeled)",
    "gan_bert_full_section": "GAN-BERT vanilla (full, + section feature)",
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
    rows = []
    for variant_dir in VARIANTS:
        rows.extend(collect(variant_dir))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_ROOT / "comparison.csv", index=False)

    variant_order = [v for v in VARIANTS.values() if v in df["variant"].unique()]
    pivot_f1 = df.pivot_table(index=["dataset", "model"], columns="variant", values="f1_macro")[variant_order]
    pivot_acc = df.pivot_table(index=["dataset", "model"], columns="variant", values="accuracy")[variant_order]

    best_f1_row = df.loc[df["f1_macro"].idxmax()]

    lines = [
        "# GAN-BERT citation-intent experiments: full comparison\n",
        "Dimensions varied: (1) partial fine-tuning (last 2 transformer layers + pooler) "
        "vs. full fine-tuning; (2) no-GAN baseline vs. vanilla GAN-BERT vs. improved "
        "GAN-BERT (label smoothing + Pi-model consistency regularization, and a "
        "corrected learning rate for SciCite); (3) in-domain vs. cross-dataset "
        "unlabeled data for the GAN discriminator's real/fake stream; (4) with vs. "
        "without a learned paper-section embedding (Introduction/Methods/Results/etc., "
        "normalized from each dataset's raw section metadata) concatenated to the "
        "representation before the discriminator/generator. The section feature did "
        "not yield a net improvement (mean F1-macro across all 9 dataset/model "
        "combinations: 0.6504 without vs. 0.6329 with), so it was not adopted for the "
        "final/best model.\n",
        f"**Best overall result:** {best_f1_row['model']} / {best_f1_row['dataset']} / "
        f"{best_f1_row['variant']} -- F1-macro={best_f1_row['f1_macro']:.4f}, "
        f"accuracy={best_f1_row['accuracy']:.4f}\n",
        "## F1-macro\n",
        pivot_f1.to_markdown(),
        "\n## Accuracy\n",
        pivot_acc.to_markdown(),
    ]
    (RESULTS_ROOT / "comparison.md").write_text("\n".join(lines) + "\n")
    print(pivot_f1.to_string())


if __name__ == "__main__":
    main()
