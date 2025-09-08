from dataclasses import dataclass

import torch


@dataclass
class ForwardResult:
    output: torch.Tensor
    representation: torch.Tensor
