#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, get_constant_schedule_with_warmup

try:
    import optuna
except ImportError as e:
    raise SystemExit(
        "Optuna is not installed. Install with: pip install optuna
"
        "Or install all extras: pip install -r requirements.txt"
    ) from e

from gan_bert.config import GanBertConfig
from gan_bert.data import make_eval_dataloader, make_train_dataloader
from gan_bert.models import ConditionalGenerator, Discriminator
from gan_bert.train_eval import evaluate, train_one_epoch
from gan_bert.utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna hyperparameter search for GAN-BERT training.")
    p.add_argument("--labeled_csv", required=True)
    p.add_argument("--unlabeled_csv", required=True)
    p.add_argument("--val_csv", required=True)
    p.add_argument("--text_col", default="text")
    p.add_argument("--label_col", default="label")
    p.add_argument("--unknown_label", default="unknown")
    p.add_argument("--labels", nargs="+", required=True, help="Label list in order; include the unknown label last.")

    p.add_argument("--output_dir", required=True, help="Where to save Optuna study + best params.")
    p.add_argument("--study_name", default="gan_bert_optuna")
    p.add_argument("--storage", default=None, help="Optuna storage URL (e.g. sqlite:///study.db). Optional.")
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)

    # Fixed-ish training defaults (you can override)
    p.add_argument("--model_name", type=str, default="allenai/scibert_scivocab_uncased")
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=6, help="Epochs per trial (keep small for tuning).")
    p.add_argument("--warmup_proportion", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)

    # Search ranges (change if you want)
    p.add_argument("--lr_g_min", type=float, default=1e-7)
    p.add_argument("--lr_g_max", type=float, default=5e-4)
    p.add_argument("--lr_d_min", type=float, default=1e-7)
    p.add_argument("--lr_d_max", type=float, default=5e-4)
    p.add_argument("--lr_t_min", type=float, default=2e-5)
    p.add_argument("--lr_t_max", type=float, default=5e-5)
    p.add_argument("--dropout_min", type=float, default=0.1)
    p.add_argument("--dropout_max", type=float, default=0.4)
    p.add_argument("--noise_min", type=int, default=100)
    p.add_argument("--noise_max", type=int, default=768)
    p.add_argument("--hidden_layers_g_min", type=int, default=1)
    p.add_argument("--hidden_layers_g_max", type=int, default=2)
    p.add_argument("--hidden_layers_d_min", type=int, default=1)
    p.add_argument("--hidden_layers_d_max", type=int, default=2)

    # speed knobs
    p.add_argument("--limit_train_rows", type=int, default=None, help="Optional: cap labeled+unlabeled rows per trial.")
    p.add_argument("--limit_val_rows", type=int, default=None, help="Optional: cap validation rows per trial.")
    p.add_argument("--disable_progress", action="store_true")

    return p.parse_args()


def _maybe_limit(df: pd.DataFrame, n: Optional[int]) -> pd.DataFrame:
    if n is None:
        return df
    return df.head(int(n)).copy()


def make_objective(args: argparse.Namespace):
    labels: List[str] = args.labels
    if labels[-1] != args.unknown_label:
        raise SystemExit("--labels must include --unknown_label as the last label (unlabeled class).")

    labeled = _maybe_limit(pd.read_csv(args.labeled_csv), args.limit_train_rows)
    unlabeled = _maybe_limit(pd.read_csv(args.unlabeled_csv), args.limit_train_rows)
    val_df = _maybe_limit(pd.read_csv(args.val_csv), args.limit_val_rows)

    device = get_device()
    set_seed(args.seed)

    def objective(trial: "optuna.Trial") -> float:
        # Sample hyperparameters
        lr_g = trial.suggest_float("lr_g", args.lr_g_min, args.lr_g_max, log=True)
        lr_d = trial.suggest_float("lr_d", args.lr_d_min, args.lr_d_max, log=True)
        lr_t = trial.suggest_float("lr_t", args.lr_t_min, args.lr_t_max, log=True)
        dropout = trial.suggest_float("dropout", args.dropout_min, args.dropout_max)
        noise_size = trial.suggest_int("noise_size", args.noise_min, args.noise_max)
        hidden_layers_g = trial.suggest_int("hidden_layers_g", args.hidden_layers_g_min, args.hidden_layers_g_max)
        hidden_layers_d = trial.suggest_int("hidden_layers_d", args.hidden_layers_d_min, args.hidden_layers_d_max)

        # Build config
        cfg = GanBertConfig(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            learning_rate_generator=lr_g,
            learning_rate_discriminator=lr_d,
            learning_rate_transformer=lr_t,
            warmup_proportion=args.warmup_proportion,
            weight_decay=args.weight_decay,
            noise_size=noise_size,
            num_hidden_layers_g=hidden_layers_g,
            num_hidden_layers_d=hidden_layers_d,
            out_dropout_rate=dropout,
            label_list=labels,
            unknown_label=args.unknown_label,
            disable_progress=args.disable_progress,
        )

        # Data
        train_dl = make_train_dataloader(
            labeled_df=labeled,
            unlabeled_df=unlabeled,
            label_list=labels,
            unknown_label=args.unknown_label,
            model_name=cfg.model_name,
            max_seq_length=cfg.max_seq_length,
            batch_size=cfg.batch_size,
            text_col=args.text_col,
            label_col=args.label_col,
        )
        val_dl = make_eval_dataloader(
            df=val_df,
            label_list=labels,
            unknown_label=args.unknown_label,
            model_name=cfg.model_name,
            max_seq_length=cfg.max_seq_length,
            batch_size=cfg.batch_size,
            text_col=args.text_col,
            label_col=args.label_col,
        )

        # Models
        transformer = AutoModel.from_pretrained(cfg.model_name)
        hidden_size = transformer.config.hidden_size

        generator = ConditionalGenerator(
            noise_size=cfg.noise_size,
            output_size=hidden_size,
            hidden_sizes=[hidden_size] * cfg.num_hidden_layers_g,
            dropout_rate=cfg.out_dropout_rate,
        )
        discriminator = Discriminator(
            input_size=hidden_size,
            hidden_sizes=[hidden_size] * cfg.num_hidden_layers_d,
            num_labels=cfg.num_labels_without_unknown,
            dropout_rate=cfg.out_dropout_rate,
        )

        transformer.to(device)
        generator.to(device)
        discriminator.to(device)

        # Optimizers + schedules
        d_vars = list(transformer.parameters()) + list(discriminator.parameters())
        g_vars = list(generator.parameters())
        t_vars = list(transformer.parameters())

        dis_opt = torch.optim.AdamW(d_vars, lr=cfg.learning_rate_discriminator, weight_decay=cfg.weight_decay)
        gen_opt = torch.optim.AdamW(g_vars, lr=cfg.learning_rate_generator, weight_decay=cfg.weight_decay)
        tr_opt = torch.optim.AdamW(t_vars, lr=cfg.learning_rate_transformer, weight_decay=cfg.weight_decay)

        num_train_steps = max(1, len(train_dl) * cfg.num_train_epochs)
        num_warmup_steps = int(cfg.warmup_proportion * num_train_steps)

        dis_sched = get_constant_schedule_with_warmup(dis_opt, num_warmup_steps=num_warmup_steps)
        gen_sched = get_constant_schedule_with_warmup(gen_opt, num_warmup_steps=num_warmup_steps)
        tr_sched = get_constant_schedule_with_warmup(tr_opt, num_warmup_steps=num_warmup_steps)

        best_f1 = -1.0
        for epoch in range(1, cfg.num_train_epochs + 1):
            train_one_epoch(
                train_dl,
                transformer,
                generator,
                discriminator,
                gen_opt,
                dis_opt,
                tr_opt,
                gen_sched,
                dis_sched,
                tr_sched,
                device=device,
                cfg=cfg,
                verbose=False,
            )
            metrics = evaluate(val_dl, transformer, discriminator, device=device, verbose=False)
            best_f1 = max(best_f1, float(metrics["f1_macro"]))

            # Let Optuna prune unpromising trials.
            trial.report(best_f1, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_f1

    return objective


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    objective = make_objective(args)

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=args.n_trials)

    best = {
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "best_trial_number": int(study.best_trial.number),
        "n_trials": int(len(study.trials)),
    }

    (out / "best_params.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    (out / "study_summary.txt").write_text(study.trials_dataframe().to_string(index=False), encoding="utf-8")
    print("Done.")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
