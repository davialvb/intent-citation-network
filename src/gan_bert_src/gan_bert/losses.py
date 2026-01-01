from __future__ import annotations

import torch
import torch.nn.functional as F


def get_cgan_input(batch_size: int, noise_size: int, labels: torch.Tensor, device: torch.device):
    """Noise + conditional labels for ConditionalGenerator."""
    noise = torch.zeros(batch_size, noise_size, device=device).uniform_(0, 1)
    # Keep labels numeric; ConditionalGenerator concatenates them.
    fake_labels = labels.detach().view(-1, 1).to(device)
    return noise, fake_labels


def generator_loss(d_fake_probs: torch.Tensor, d_fake_features: torch.Tensor, d_real_features: torch.Tensor, epsilon: float):
    g_loss_d = -1.0 * torch.mean(torch.log(1 - d_fake_probs[:, -1] + epsilon))
    g_feat_reg = torch.mean((torch.mean(d_real_features, dim=0) - torch.mean(d_fake_features, dim=0)) ** 2)
    return g_loss_d + g_feat_reg


def discriminator_loss(
    d_real_logits: torch.Tensor,
    d_real_probs: torch.Tensor,
    d_fake_probs: torch.Tensor,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
    num_labels: int,
    epsilon: float,
    device: torch.device,
):
    # Supervised loss (ignore unlabeled by masking)
    logits = d_real_logits[:, 0:-1]  # ignore fake/real column
    log_probs = F.log_softmax(logits, dim=-1)

    label2one_hot = torch.nn.functional.one_hot(labels, num_labels).to(device)
    per_example_loss = -torch.sum(label2one_hot * log_probs, dim=-1)

    mask = label_mask.to(device).bool()
    per_example_loss = torch.masked_select(per_example_loss, mask)
    labeled_example_count = per_example_loss.float().numel()

    if labeled_example_count == 0:
        d_l_supervised = torch.tensor(0.0, device=device)
    else:
        d_l_supervised = torch.sum(per_example_loss) / labeled_example_count

    d_l_unsup_real = -1.0 * torch.mean(torch.log(1 - d_real_probs[:, -1] + epsilon))
    d_l_unsup_fake = -1.0 * torch.mean(torch.log(d_fake_probs[:, -1] + epsilon))
    return d_l_supervised + d_l_unsup_real + d_l_unsup_fake
