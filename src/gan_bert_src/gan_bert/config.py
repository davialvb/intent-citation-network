from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class GanBertConfig:
    # data
    max_seq_length: int = 160
    batch_size: int = 32
    model_name: str = "allenai/scibert_scivocab_uncased"

    # model sizes
    num_labels: int = 4
    hidden_size: int = 768

    # generator/discriminator
    noise_size: int = 768
    num_hidden_layers_g: int = 1
    num_hidden_layers_d: int = 3
    out_dropout_rate: float = 0.2
    discriminator_noise_stddev: float = 0.1

    # optimization
    learning_rate_generator: float = 2e-7
    learning_rate_discriminator: float = 2e-7
    epsilon: float = 2e-7

    # training
    num_train_epochs: int = 25
    print_each_n_step: int = 10

    # scheduler
    apply_scheduler: bool = True
    warmup_proportion: float = 0.1

    # misc
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GanBertConfig":
        return GanBertConfig(**d)
