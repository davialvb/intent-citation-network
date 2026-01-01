#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, get_constant_schedule_with_warmup

from gan_bert.config import GanBertConfig
from gan_bert.data import label_str2int, make_eval_dataloader, make_train_dataloader
from gan_bert.models import ConditionalGenerator, Discriminator
from gan_bert.train_eval import evaluate, train_one_epoch
from gan_bert.utils import SavePaths, get_device, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train GAN-BERT (refactored).")
    p.add_argument("--labeled_csv", required=True, help="CSV with labeled data (needs columns: text, intent).")
    p.add_argument("--unlabeled_csv", required=False, default=None, help="CSV with unlabeled data (text column).")
    p.add_argument("--val_csv", required=False, default=None, help="CSV used for validation.")
    p.add_argument("--test_csv", required=False, default=None, help="Optional CSV used for final test reporting.")
    p.add_argument("--text_col", default="text")
    p.add_argument("--label_col", default="intent")
    p.add_argument("--unknown_label", default="unknown", help="Label name for unlabeled rows.")
    p.add_argument("--labels", nargs="+", required=True, help="Label list in order; include the unknown label last.")
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(prefer_cuda=not args.no_cuda)
    set_seed(args.seed)

    cfg = GanBertConfig(
        seed=args.seed,
        model_name=args.model_name or GanBertConfig.model_name,
        num_labels=len(args.labels),
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

    labeled = pd.read_csv(args.labeled_csv)
    unlabeled = pd.read_csv(args.unlabeled_csv) if args.unlabeled_csv else None
    val = pd.read_csv(args.val_csv) if args.val_csv else None
    test = pd.read_csv(args.test_csv) if args.test_csv else None

    # Ensure unlabeled has unknown label
    label_map = label_str2int(args.labels)
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

    # optimizers
    transformer_vars = list(transformer.parameters())
    d_vars = transformer_vars + list(discriminator.parameters())
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

    best_f1 = -1.0
    best_epoch = -1

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
        )
        print(f"  Avg generator loss: {stats['avg_gen_loss']:.4f}")
        print(f"  Avg discriminator loss: {stats['avg_dis_loss']:.4f}")
        print(f"  Epoch time: {stats['epoch_time']}")

        print("\nRunning validation...")
        val_metrics = evaluate(val_loader, transformer, discriminator, device=device, verbose=True)

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            torch.save(generator.state_dict(), paths.generator_dir / f"generator_epoch{epoch}_f1_{best_f1:.4f}.pth")
            torch.save(discriminator.state_dict(), paths.discriminator_dir / f"discriminator_epoch{epoch}_f1_{best_f1:.4f}.pth")
            torch.save(transformer.state_dict(), paths.transformer_dir / f"transformer_epoch{epoch}_f1_{best_f1:.4f}.pth")
            print(f"Saved best model (epoch={epoch}, f1={best_f1:.4f})")

    print(f"\nBest validation F1-macro: {best_f1:.4f} at epoch {best_epoch}")
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
        _ = evaluate(test_loader, transformer, discriminator, device=device, verbose=True)


if __name__ == "__main__":
    main()
