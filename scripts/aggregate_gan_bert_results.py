#!/usr/bin/env python
"""Aggregate per-run test_metrics.json files from results/gan_bert/<dataset>/<model>/
into a single summary.csv and summary.md table."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results" / "gan_bert"

MODEL_LABELS = {"specter2": "SPECTER2", "scibert": "SciBERT", "xlnet": "XLNet"}
DATASET_LABELS = {"scicite": "SciCite", "acl-arc": "ACL-ARC", "3C": "3C"}


def main() -> None:
    rows = []
    for dataset_dir in sorted(RESULTS_ROOT.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name == "logs":
            continue
        for model_dir in sorted(dataset_dir.iterdir()):
            metrics_path = model_dir / "test_metrics.json"
            meta_path = model_dir / "run_meta.json"
            if not metrics_path.exists():
                rows.append(
                    {
                        "dataset": DATASET_LABELS.get(dataset_dir.name, dataset_dir.name),
                        "model": MODEL_LABELS.get(model_dir.name, model_dir.name),
                        "accuracy": None,
                        "f1_macro": None,
                        "status": "MISSING (check logs)",
                    }
                )
                continue
            metrics = json.loads(metrics_path.read_text())
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            rows.append(
                {
                    "dataset": DATASET_LABELS.get(dataset_dir.name, dataset_dir.name),
                    "model": MODEL_LABELS.get(model_dir.name, model_dir.name),
                    "accuracy": round(metrics["accuracy"], 4),
                    "f1_macro": round(metrics["f1_macro"], 4),
                    "best_epoch": meta.get("best_epoch"),
                    "wall_clock_seconds": round(meta.get("wall_clock_seconds", 0), 1),
                    "status": "ok",
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_ROOT / "summary.csv", index=False)

    pivot_acc = df.pivot(index="dataset", columns="model", values="accuracy")
    pivot_f1 = df.pivot(index="dataset", columns="model", values="f1_macro")

    lines = ["# GAN-BERT experiment results\n", "## Accuracy\n", pivot_acc.to_markdown(), "\n## F1-macro\n", pivot_f1.to_markdown(), "\n## Full detail\n", df.to_markdown(index=False)]
    (RESULTS_ROOT / "summary.md").write_text("\n".join(lines) + "\n")

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
