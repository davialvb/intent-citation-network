#!/usr/bin/env python
"""No-GAN baseline: same frozen/partially-fine-tuned transformer backbone as
cli_train.py, but with a plain linear classification head trained with
ordinary supervised cross-entropy on the labeled data only (no generator, no
discriminator adversarial game, no unlabeled stream). Used to isolate whether
the GAN-BERT semi-supervised training actually helps over plain fine-tuning.
"""
from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import AutoConfig, AutoModel, get_constant_schedule_with_warmup

from gan_bert.data import label_str2int, make_eval_dataloader, make_train_dataloader
from gan_bert.models import SimpleClassifier
from gan_bert.utils import (
    SavePaths,
    best_checkpoint,
    freeze_transformer_layers,
    get_device,
    get_transformer_representation,
    save_json,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train a plain (no-GAN) classifier baseline.")
    p.add_argument("--labeled_csv", required=True)
    p.add_argument("--val_csv", required=True)
    p.add_argument("--test_csv", required=False, default=None)
    p.add_argument("--text_col", default="text")
    p.add_argument("--label_col", default="intent")
    p.add_argument("--labels", nargs="+", required=True, help="Real class labels (no trailing 'unknown').")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_seq_length", type=int, default=160)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing on the supervised cross-entropy. Lets the no-GAN arm be compared "
        "against the 'improved' SS-cGAN's label-smoothing ingredient in isolation.",
    )
    p.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="AdamW weight decay (0.01 is torch's default, i.e. what the original runs used).",
    )
    p.add_argument("--num_trainable_layers", type=int, default=2)
    p.add_argument("--warmup_proportion", type=float, default=0.1)
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--nondeterministic",
        action="store_true",
        help="Allow nondeterministic GPU kernels (faster, but a fixed seed no longer pins the run).",
    )
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()


def _git_commit_hash() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
        ).strip()
    except Exception:
        return None


@torch.no_grad()
def evaluate(dataloader, transformer, classifier, device):
    transformer.eval()
    classifier.eval()
    nll_loss = nn.CrossEntropyLoss()

    all_preds, all_labels, all_texts = [], [], []
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = transformer(input_ids, attention_mask=attn_mask)
        rep = get_transformer_representation(outputs, attention_mask=attn_mask)
        logits = classifier(rep)

        loss = nll_loss(logits, labels)
        total_loss += float(loss.detach().cpu())

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_texts.extend(list(batch["texts"]))

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    acc = float(np.mean(y_pred == y_true))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=1.0, output_dict=True)

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "avg_loss": total_loss / max(1, len(dataloader)),
        "confusion_matrix": cm,
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "texts": all_texts,
    }


def main():
    args = parse_args()
    device = get_device(prefer_cuda=not args.no_cuda)
    set_seed(args.seed, deterministic=not args.nondeterministic)

    labeled = pd.read_csv(args.labeled_csv)
    val = pd.read_csv(args.val_csv)
    test = pd.read_csv(args.test_csv) if args.test_csv else None

    label_map = label_str2int(args.labels)
    labeled["intent_int"] = labeled[args.label_col].map(label_map)
    val["intent_int"] = val[args.label_col].map(label_map)
    if test is not None:
        test["intent_int"] = test[args.label_col].map(label_map)

    train_loader = make_train_dataloader(
        labeled_examples=labeled,
        unlabeled_examples=None,
        max_seq_length=args.max_seq_length,
        model_name=args.model_name,
        col_text=args.text_col,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = make_eval_dataloader(
        examples=val,
        max_seq_length=args.max_seq_length,
        model_name=args.model_name,
        col_text=args.text_col,
        batch_size=args.batch_size,
        shuffle=False,
    )

    transformer = AutoModel.from_pretrained(args.model_name)
    model_config = AutoConfig.from_pretrained(args.model_name)
    hidden_size = int(model_config.hidden_size)

    freeze_stats = freeze_transformer_layers(transformer, args.num_trainable_layers)
    print(
        f"Transformer params -- trainable: {freeze_stats['trainable_params']:,} "
        f"frozen: {freeze_stats['frozen_params']:,}"
    )

    classifier = SimpleClassifier(input_size=hidden_size, num_labels=len(args.labels), dropout_rate=args.dropout)

    transformer.to(device)
    classifier.to(device)

    trainable_vars = [p for p in transformer.parameters() if p.requires_grad] + list(classifier.parameters())
    optimizer = torch.optim.AdamW(trainable_vars, lr=args.lr, weight_decay=args.weight_decay)

    num_train_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(num_train_steps * args.warmup_proportion)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)

    paths = SavePaths.for_run(args.output_dir)
    save_json(
        {
            "model_name": args.model_name,
            "num_labels": len(args.labels),
            "labels": args.labels,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_seq_length": args.max_seq_length,
            "num_trainable_layers": args.num_trainable_layers,
            "label_smoothing": args.label_smoothing,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
        },
        paths.config_path,
    )

    run_start = time.time()
    run_meta = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "labels": args.labels,
        "num_trainable_layers": args.num_trainable_layers,
        "trainable_transformer_params": freeze_stats["trainable_params"],
        "frozen_transformer_params": freeze_stats["frozen_params"],
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "git_commit": _git_commit_hash(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "deterministic": not args.nondeterministic,
        "n_train_labeled": len(labeled),
        "n_val": len(val),
        "n_test": len(test) if test is not None else 0,
    }

    # Training loss may be smoothed; the validation loss in evaluate() deliberately
    # stays unsmoothed so val-loss trajectories remain comparable across arms.
    nll_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    best_f1 = -1.0
    best_epoch = -1
    history: List[dict] = []

    for epoch in range(1, args.epochs + 1):
        transformer.train()
        classifier.train()
        t0 = time.time()
        tr_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = transformer(input_ids, attention_mask=attn_mask)
            rep = get_transformer_representation(outputs, attention_mask=attn_mask)
            logits = classifier(rep)
            loss = nll_loss(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            tr_loss += float(loss.detach().cpu())

        avg_train_loss = tr_loss / max(1, len(train_loader))
        val_metrics = evaluate(val_loader, transformer, classifier, device)
        epoch_time = time.time() - t0

        print(
            f"Epoch {epoch}/{args.epochs} train_loss={avg_train_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1_macro']:.4f} "
            f"({epoch_time:.1f}s)"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_avg_loss": val_metrics["avg_loss"],
                "epoch_time_seconds": epoch_time,
            }
        )

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            torch.save(classifier.state_dict(), paths.discriminator_dir / f"classifier_epoch{epoch}_f1_{best_f1:.4f}.pth")
            torch.save(transformer.state_dict(), paths.transformer_dir / f"transformer_epoch{epoch}_f1_{best_f1:.4f}.pth")
            save_json(
                {
                    "epoch": best_epoch,
                    "f1_macro": val_metrics["f1_macro"],
                    "accuracy": val_metrics["accuracy"],
                    "confusion_matrix": val_metrics["confusion_matrix"].tolist(),
                    "classification_report": val_metrics["classification_report"],
                },
                paths.root / "best_val_metrics.json",
            )

    save_json(history, paths.root / "history.json")

    fig, ax = plt.subplots(figsize=(6, 4))
    epochs_x = [h["epoch"] for h in history]
    ax.plot(epochs_x, [h["train_loss"] for h in history], label="train loss")
    ax.plot(epochs_x, [h["val_avg_loss"] for h in history], label="val loss")
    ax2 = ax.twinx()
    ax2.plot(epochs_x, [h["val_f1_macro"] for h in history], label="val F1-macro", color="green")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax2.set_ylabel("F1-macro")
    fig.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(paths.root / "training_curve.png", dpi=150)
    plt.close(fig)

    run_meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    run_meta["wall_clock_seconds"] = time.time() - run_start
    run_meta["best_epoch"] = best_epoch
    run_meta["best_val_f1_macro"] = best_f1

    if test is not None:
        test_loader = make_eval_dataloader(
            examples=test,
            max_seq_length=args.max_seq_length,
            model_name=args.model_name,
            col_text=args.text_col,
            batch_size=args.batch_size,
            shuffle=False,
        )
        # See cli_train.py: the final-epoch numbers are kept for comparability
        # with earlier revisions, but the headline metrics use the
        # best-validation-F1 checkpoint so that every arm is selected the same
        # way regardless of where in the epoch budget it peaks.
        final_metrics = evaluate(test_loader, transformer, classifier, device)
        print(
            f"TEST (final epoch) accuracy={final_metrics['accuracy']:.4f} "
            f"f1_macro={final_metrics['f1_macro']:.4f}"
        )
        save_json(
            {
                "selection": "final_epoch",
                "epoch": args.epochs,
                "accuracy": final_metrics["accuracy"],
                "f1_macro": final_metrics["f1_macro"],
                "avg_loss": final_metrics["avg_loss"],
                "confusion_matrix": final_metrics["confusion_matrix"].tolist(),
                "classification_report": final_metrics["classification_report"],
            },
            paths.root / "test_metrics_final_epoch.json",
        )

        t_ckpt = best_checkpoint(paths.transformer_dir, "transformer_")
        c_ckpt = best_checkpoint(paths.discriminator_dir, "classifier_")
        if t_ckpt is not None and c_ckpt is not None:
            print(f"Reloading best-validation checkpoints: {t_ckpt.name} / {c_ckpt.name}")
            transformer.load_state_dict(torch.load(t_ckpt, map_location=device))
            classifier.load_state_dict(torch.load(c_ckpt, map_location=device))
            test_metrics = evaluate(test_loader, transformer, classifier, device)
            selection, sel_epoch = "best_val_f1_checkpoint", best_epoch
        else:
            print("WARNING: no checkpoints found; reporting final-epoch weights as the headline metrics.")
            test_metrics = final_metrics
            selection, sel_epoch = "final_epoch", args.epochs

        print(
            f"TEST (best checkpoint, epoch {sel_epoch}) accuracy={test_metrics['accuracy']:.4f} "
            f"f1_macro={test_metrics['f1_macro']:.4f}"
        )
        save_json(
            {
                "selection": selection,
                "epoch": sel_epoch,
                "accuracy": test_metrics["accuracy"],
                "f1_macro": test_metrics["f1_macro"],
                "avg_loss": test_metrics["avg_loss"],
                "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
                "classification_report": test_metrics["classification_report"],
            },
            paths.root / "test_metrics.json",
        )
        pd.DataFrame(
            {"text": test_metrics["texts"], "y_true": test_metrics["y_true"], "y_pred": test_metrics["y_pred"]}
        ).to_csv(paths.root / "predictions_test.csv", index=False)

    save_json(run_meta, paths.root / "run_meta.json")


if __name__ == "__main__":
    main()
