from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .losses import discriminator_loss, generator_loss, get_cgan_input
from .utils import format_time, get_transformer_representation


@torch.no_grad()
def predict(dataloader, transformer, discriminator, device: torch.device):
    transformer.eval()
    discriminator.eval()

    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_texts: List[str] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        texts = batch["texts"]

        outputs = transformer(input_ids, attention_mask=attn_mask)
        rep = get_transformer_representation(outputs)

        _, logits, _ = discriminator(rep)
        filtered_logits = logits[:, 0:-1]

        preds = torch.argmax(filtered_logits, dim=1)

        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_texts.extend(list(texts))

    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy(), all_texts


@torch.no_grad()
def evaluate(dataloader, transformer, discriminator, device: torch.device, verbose: bool = True):
    transformer.eval()
    discriminator.eval()

    nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = transformer(input_ids, attention_mask=attn_mask)
        rep = get_transformer_representation(outputs)

        _, logits, _ = discriminator(rep)
        filtered_logits = logits[:, 0:-1]

        loss = nll_loss(filtered_logits, labels)
        total_loss += float(loss.detach().cpu())

        preds = torch.argmax(filtered_logits, dim=1)
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    acc = float(np.mean(y_pred == y_true))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    cm = confusion_matrix(y_true, y_pred)

    if verbose:
        print(f"  F1-macro: {f1_macro:.4f}")
        print(f"  Accuracy: {acc:.4f}")
        print(classification_report(y_true, y_pred, zero_division=1.0))

    return {
        "f1_macro": f1_macro,
        "accuracy": acc,
        "avg_loss": total_loss / max(1, len(dataloader)),
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_one_epoch(
    dataloader,
    transformer,
    generator,
    discriminator,
    gen_optimizer,
    dis_optimizer,
    *,
    noise_size: int,
    num_labels: int,
    epsilon: float,
    device: torch.device,
    print_each_n_step: int = 50,
    apply_scheduler: bool = False,
    scheduler_d=None,
    scheduler_g=None,
    verbose: bool = True,
):
    transformer.train()
    generator.train()
    discriminator.train()

    tr_g_loss = 0.0
    tr_d_loss = 0.0

    t0 = time.time()

    for step, batch in enumerate(dataloader):
        if step % print_each_n_step == 0 and step != 0 and verbose:
            print(f"  Batch {step:>5,} of {len(dataloader):>5,}. Elapsed: {format_time(time.time() - t0)}")

        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        label_mask = batch["label_masks"].to(device)

        real_batch_size = input_ids.size(0)

        outputs = transformer(input_ids, attention_mask=attn_mask)
        real_rep = get_transformer_representation(outputs)

        noise, cond_labels = get_cgan_input(real_batch_size, noise_size, labels, device=device)
        fake_rep = generator(noise, cond_labels)

        disc_input = torch.cat([real_rep, fake_rep], dim=0)
        features, logits, probs = discriminator(disc_input)

        real_features, fake_features = torch.split(features, real_batch_size)
        real_logits, _fake_logits = torch.split(logits, real_batch_size)
        real_probs, fake_probs = torch.split(probs, real_batch_size)

        g_loss = generator_loss(fake_probs, fake_features, real_features, epsilon)
        d_loss = discriminator_loss(
            real_logits,
            real_probs,
            fake_probs,
            labels=labels,
            label_mask=label_mask,
            num_labels=num_labels,
            epsilon=epsilon,
            device=device,
        )

        gen_optimizer.zero_grad(set_to_none=True)
        dis_optimizer.zero_grad(set_to_none=True)

        g_loss.backward(retain_graph=True)
        d_loss.backward()

        gen_optimizer.step()
        dis_optimizer.step()

        if apply_scheduler:
            if scheduler_d is not None:
                scheduler_d.step()
            if scheduler_g is not None:
                scheduler_g.step()

        tr_g_loss += float(g_loss.detach().cpu())
        tr_d_loss += float(d_loss.detach().cpu())

    avg_g = tr_g_loss / max(1, len(dataloader))
    avg_d = tr_d_loss / max(1, len(dataloader))

    return {
        "avg_gen_loss": avg_g,
        "avg_dis_loss": avg_d,
        "epoch_time": format_time(time.time() - t0),
    }
