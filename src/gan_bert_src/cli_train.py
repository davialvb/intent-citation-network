#!/usr/bin/env python
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
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, get_constant_schedule_with_warmup

from gan_bert.config import GanBertConfig
from gan_bert.data import label_str2int, make_eval_dataloader, make_train_dataloader
from gan_bert.models import ConditionalGenerator, Discriminator
from gan_bert.train_eval import evaluate, predict, train_one_epoch
from gan_bert.utils import SavePaths, freeze_transformer_layers, get_device, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train GAN-BERT (refactored).")
    p.add_argument("--labeled_csv", required=True, help="CSV with labeled data (needs columns: text, intent).")
    p.add_argument("--unlabeled_csv", required=False, default=None, help="CSV with unlabeled data (text column).")
    p.add_argument("--val_csv", required=False, default=None, help="CSV used for validation.")
    p.add_argument("--test_csv", required=False, default=None, help="Optional CSV used for final test reporting.")
    p.add_argument("--text_col", default="text")
    p.add_argument("--label_col", default="intent")
    p.add_argument("--unknown_label", default="unknown", help="Label name for unlabeled rows.")
    p.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Label list in order; include the unknown label last. num_labels passed to "
        "the discriminator's classification head EXCLUDES this trailing unknown label.",
    )
    p.add_argument("--output_dir", required=True, help="Where to save weights/config.")
    p.add_argument("--epochs", type=int, default=None, help="Override num_train_epochs.")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--max_seq_length", type=int, default=None)
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--lr_g", type=float, default=None)
    p.add_argument("--lr_d", type=float, default=None)
    p.add_argument("--noise_size", type=int, default=None)
    p.add_argument("--hidden_layers_g", type=int, default=None)
    p.add_argument("--hidden_layers_d", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--epsilon", type=float, default=None)
    p.add_argument(
        "--num_trainable_layers",
        type=int,
        default=None,
        help="Only the last N transformer encoder blocks (+ pooler, if present) are fine-tuned; "
        "everything else (embeddings + earlier blocks) is frozen.",
    )
    p.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing applied to the discriminator's supervised loss (0 = vanilla GAN-BERT).",
    )
    p.add_argument(
        "--consistency_weight",
        type=float,
        default=0.0,
        help="Weight for Pi-model consistency regularization on the unlabeled stream "
        "(two stochastic discriminator forward passes); 0 = vanilla GAN-BERT.",
    )
    p.add_argument("--dataset_name", type=str, default=None, help="Free-text label recorded in run_meta.json.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()


def _git_commit_hash() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
        ).strip()
    except Exception:
        return None


def _plot_training_curve(history: List[dict], out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, [h["avg_gen_loss"] for h in history], label="generator loss")
    axes[0].plot(epochs, [h["avg_dis_loss"] for h in history], label="discriminator loss")
    axes[0].plot(epochs, [h["val_avg_loss"] for h in history], label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].set_title("Losses")

    axes[1].plot(epochs, [h["val_accuracy"] for h in history], label="val accuracy")
    axes[1].plot(epochs, [h["val_f1_macro"] for h in history], label="val F1-macro")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].legend()
    axes[1].set_title("Validation metrics")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_confusion_matrix(cm, labels: List[str], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 2, 1.1 * len(labels) + 2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    device = get_device(prefer_cuda=not args.no_cuda)
    set_seed(args.seed)

    cfg = GanBertConfig(
        seed=args.seed,
        model_name=args.model_name or GanBertConfig.model_name,
        # `args.labels` includes a trailing "unknown" label used only to mark
        # unlabeled rows (whose supervised loss is masked out); the
        # discriminator's classification head must NOT include it as a class.
        num_labels=len(args.labels) - 1,
    )
    # overrides
    if args.epochs is not None:
        cfg.num_train_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.max_seq_length is not None:
        cfg.max_seq_length = args.max_seq_length
    if args.lr_g is not None:
        cfg.learning_rate_generator = args.lr_g
    if args.lr_d is not None:
        cfg.learning_rate_discriminator = args.lr_d
    if args.noise_size is not None:
        cfg.noise_size = args.noise_size
    if args.hidden_layers_g is not None:
        cfg.num_hidden_layers_g = args.hidden_layers_g
    if args.hidden_layers_d is not None:
        cfg.num_hidden_layers_d = args.hidden_layers_d
    if args.dropout is not None:
        cfg.out_dropout_rate = args.dropout
    if args.epsilon is not None:
        cfg.epsilon = args.epsilon
    if args.num_trainable_layers is not None:
        cfg.num_trainable_layers = args.num_trainable_layers

    labeled = pd.read_csv(args.labeled_csv)
    unlabeled = pd.read_csv(args.unlabeled_csv) if args.unlabeled_csv else None
    val = pd.read_csv(args.val_csv) if args.val_csv else None
    test = pd.read_csv(args.test_csv) if args.test_csv else None

    # Ensure unlabeled has unknown label
    label_map = label_str2int(args.labels)
    real_labels = args.labels[:-1]  # excludes the trailing "unknown" entry
    labeled["intent_int"] = labeled[args.label_col].map(label_map)

    if unlabeled is not None:
        unlabeled = unlabeled.copy()
        unlabeled[args.label_col] = args.unknown_label
        unlabeled["intent_int"] = unlabeled[args.label_col].map(label_map)

    if val is None:
        raise SystemExit("Please provide --val_csv (this refactor expects a validation split).")
    val["intent_int"] = val[args.label_col].map(label_map)

    if test is not None:
        test["intent_int"] = test[args.label_col].map(label_map)

    train_loader = make_train_dataloader(
        labeled_examples=labeled,
        unlabeled_examples=unlabeled,
        max_seq_length=cfg.max_seq_length,
        model_name=cfg.model_name,
        col_text=args.text_col,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = make_eval_dataloader(
        examples=val,
        max_seq_length=cfg.max_seq_length,
        model_name=cfg.model_name,
        col_text=args.text_col,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    # models
    transformer = AutoModel.from_pretrained(cfg.model_name)
    model_config = AutoConfig.from_pretrained(cfg.model_name)
    cfg.hidden_size = int(model_config.hidden_size)

    freeze_stats = freeze_transformer_layers(transformer, cfg.num_trainable_layers)
    print(
        f"Transformer params -- trainable: {freeze_stats['trainable_params']:,} "
        f"frozen: {freeze_stats['frozen_params']:,}"
    )

    hidden_levels_g = [cfg.hidden_size for _ in range(cfg.num_hidden_layers_g)]
    hidden_levels_d = [cfg.hidden_size for _ in range(cfg.num_hidden_layers_d)]

    generator = ConditionalGenerator(
        noise_size=cfg.noise_size,
        output_size=cfg.hidden_size,
        hidden_sizes=hidden_levels_g,
        dropout_rate=cfg.out_dropout_rate,
    )
    discriminator = Discriminator(
        input_size=cfg.hidden_size,
        hidden_sizes=hidden_levels_d,
        num_labels=cfg.num_labels,
        dropout_rate=cfg.out_dropout_rate,
        noise_stddev=cfg.discriminator_noise_stddev,
    )

    transformer.to(device)
    generator.to(device)
    discriminator.to(device)

    # optimizers -- only trainable transformer params (frozen layers excluded)
    transformer_trainable_vars = [p for p in transformer.parameters() if p.requires_grad]
    d_vars = transformer_trainable_vars + list(discriminator.parameters())
    g_vars = list(generator.parameters())

    dis_opt = torch.optim.AdamW(d_vars, lr=cfg.learning_rate_discriminator)
    gen_opt = torch.optim.AdamW(g_vars, lr=cfg.learning_rate_generator)

    scheduler_d = scheduler_g = None
    if cfg.apply_scheduler:
        num_train_steps = int(len(train_loader) * cfg.num_train_epochs)
        num_warmup_steps = int(num_train_steps * cfg.warmup_proportion)
        scheduler_d = get_constant_schedule_with_warmup(dis_opt, num_warmup_steps=num_warmup_steps)
        scheduler_g = get_constant_schedule_with_warmup(gen_opt, num_warmup_steps=num_warmup_steps)

    # saving
    paths = SavePaths.for_run(args.output_dir)
    save_json(cfg.to_dict(), paths.config_path)

    run_start = time.time()
    run_meta = {
        "model_name": cfg.model_name,
        "dataset_name": args.dataset_name,
        "labels": args.labels,
        "num_labels": cfg.num_labels,
        "num_trainable_layers": cfg.num_trainable_layers,
        "trainable_transformer_params": freeze_stats["trainable_params"],
        "frozen_transformer_params": freeze_stats["frozen_params"],
        "label_smoothing": args.label_smoothing,
        "consistency_weight": args.consistency_weight,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "git_commit": _git_commit_hash(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_train_labeled": len(labeled),
        "n_train_unlabeled": len(unlabeled) if unlabeled is not None else 0,
        "n_val": len(val),
        "n_test": len(test) if test is not None else 0,
    }

    best_f1 = -1.0
    best_epoch = -1
    history: List[dict] = []

    for epoch in range(1, cfg.num_train_epochs + 1):
        print(f"\n======== Epoch {epoch} / {cfg.num_train_epochs} ========")
        stats = train_one_epoch(
            train_loader,
            transformer,
            generator,
            discriminator,
            gen_opt,
            dis_opt,
            noise_size=cfg.noise_size,
            num_labels=cfg.num_labels,
            epsilon=cfg.epsilon,
            device=device,
            print_each_n_step=cfg.print_each_n_step,
            apply_scheduler=cfg.apply_scheduler,
            scheduler_d=scheduler_d,
            scheduler_g=scheduler_g,
            verbose=True,
            label_smoothing=args.label_smoothing,
            consistency_weight=args.consistency_weight,
        )
        print(f"  Avg generator loss: {stats['avg_gen_loss']:.4f}")
        print(f"  Avg discriminator loss: {stats['avg_dis_loss']:.4f}")
        print(f"  Epoch time: {stats['epoch_time']}")

        print("\nRunning validation...")
        val_metrics = evaluate(val_loader, transformer, discriminator, device=device, verbose=True)

        history.append(
            {
                "epoch": epoch,
                "avg_gen_loss": stats["avg_gen_loss"],
                "avg_dis_loss": stats["avg_dis_loss"],
                "epoch_time": stats["epoch_time"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_avg_loss": val_metrics["avg_loss"],
            }
        )

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            torch.save(generator.state_dict(), paths.generator_dir / f"generator_epoch{epoch}_f1_{best_f1:.4f}.pth")
            torch.save(discriminator.state_dict(), paths.discriminator_dir / f"discriminator_epoch{epoch}_f1_{best_f1:.4f}.pth")
            torch.save(transformer.state_dict(), paths.transformer_dir / f"transformer_epoch{epoch}_f1_{best_f1:.4f}.pth")
            print(f"Saved best model (epoch={epoch}, f1={best_f1:.4f})")

            save_json(
                {
                    "epoch": best_epoch,
                    "f1_macro": val_metrics["f1_macro"],
                    "accuracy": val_metrics["accuracy"],
                    "avg_loss": val_metrics["avg_loss"],
                    "confusion_matrix": val_metrics["confusion_matrix"].tolist(),
                    "classification_report": val_metrics["classification_report"],
                },
                paths.root / "best_val_metrics.json",
            )

    print(f"\nBest validation F1-macro: {best_f1:.4f} at epoch {best_epoch}")

    save_json(history, paths.root / "history.json")
    _plot_training_curve(history, paths.root / "training_curve.png")

    run_meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    run_meta["wall_clock_seconds"] = time.time() - run_start
    run_meta["best_epoch"] = best_epoch
    run_meta["best_val_f1_macro"] = best_f1

    if test is not None:
        print("\nRunning test...")
        test_loader = make_eval_dataloader(
            examples=test,
            max_seq_length=cfg.max_seq_length,
            model_name=cfg.model_name,
            col_text=args.text_col,
            batch_size=cfg.batch_size,
            shuffle=False,
        )
        test_metrics = evaluate(test_loader, transformer, discriminator, device=device, verbose=True)
        save_json(
            {
                "accuracy": test_metrics["accuracy"],
                "f1_macro": test_metrics["f1_macro"],
                "avg_loss": test_metrics["avg_loss"],
                "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
                "classification_report": test_metrics["classification_report"],
            },
            paths.root / "test_metrics.json",
        )
        _plot_confusion_matrix(
            test_metrics["confusion_matrix"], real_labels, paths.root / "confusion_matrix.png", "Test confusion matrix"
        )

        preds, trues, texts = predict(test_loader, transformer, discriminator, device=device)
        pd.DataFrame({"text": texts, "y_true": trues, "y_pred": preds}).to_csv(
            paths.root / "predictions_test.csv", index=False
        )

    save_json(run_meta, paths.root / "run_meta.json")


if __name__ == "__main__":
    main()
