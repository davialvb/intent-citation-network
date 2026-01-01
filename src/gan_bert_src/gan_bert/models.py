from __future__ import annotations

import torch
import torch.nn as nn


class ConditionalGenerator(nn.Module):
    def __init__(
        self,
        noise_size: int = 100,
        condition_size: int = 1,
        output_size: int = 512,
        hidden_sizes: list[int] | None = None,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512]

        layers: list[nn.Module] = []
        input_size = noise_size + condition_size
        sizes = [input_size] + list(hidden_sizes)

        for i in range(len(sizes) - 1):
            layers.extend(
                [
                    nn.Linear(sizes[i], sizes[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(dropout_rate),
                    nn.LayerNorm(sizes[i + 1]),
                ]
            )

        layers.append(nn.Linear(sizes[-1], output_size))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # condition is expected to be shape [B, 1] (integer labels) or already embedded; here we keep it numeric.
        if condition.dim() == 1:
            condition = condition.view(-1, 1)
        input_tensor = torch.cat((noise, condition), dim=1)
        return self.layers(input_tensor)


class Generator(nn.Module):
    def __init__(
        self,
        noise_size: int = 100,
        output_size: int = 512,
        hidden_sizes: list[int] | None = None,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512]

        layers: list[nn.Module] = []
        sizes = [noise_size] + list(hidden_sizes)
        for i in range(len(sizes) - 1):
            layers.extend(
                [
                    nn.Linear(sizes[i], sizes[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(dropout_rate),
                ]
            )

        layers.append(nn.Linear(sizes[-1], output_size))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.layers(noise)


class GaussianNoise(nn.Module):
    """Gaussian noise layer used by the discriminator."""

    def __init__(self, stddev: float = 0.1):
        super().__init__()
        self.stddev = stddev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return x + torch.randn_like(x) * self.stddev
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        hidden_sizes: list[int] | None = None,
        num_labels: int = 2,
        dropout_rate: float = 0.1,
        noise_stddev: float = 0.1,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [512]

        self.input_dropout = nn.Dropout(p=dropout_rate)
        self.gaussian_noise = GaussianNoise(stddev=noise_stddev)

        layers: list[nn.Module] = []
        sizes = [input_size] + list(hidden_sizes)
        for i in range(len(sizes) - 1):
            layers.extend(
                [
                    nn.Linear(sizes[i], sizes[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(dropout_rate),
                ]
            )

        self.layers = nn.Sequential(*layers)
        self.logit = nn.Linear(sizes[-1], num_labels + 1)  # +1 for "fake/real" logit
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_rep: torch.Tensor):
        input_rep = self.gaussian_noise(input_rep)
        input_rep = self.input_dropout(input_rep)
        last_rep = self.layers(input_rep)
        logits = self.logit(last_rep)
        probs = self.softmax(logits)
        return last_rep, logits, probs
