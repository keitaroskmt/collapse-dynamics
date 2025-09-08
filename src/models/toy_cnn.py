import torch
from torch import Tensor, nn

from src.models.utils import ForwardResult


class CNNModel(nn.Module):
    """A simple five layer CNN model."""

    def __init__(
        self,
        input_size: int = 28,
        input_channels: int = 1,
        num_classes: int = 10,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        if activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError
        num_channel_scale = 1 if input_channels == 1 else 2

        self.conv1 = nn.Conv2d(
            input_channels,
            16 * num_channel_scale,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            16 * num_channel_scale,
            32 * num_channel_scale,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear1 = nn.Linear(32 * num_channel_scale * (input_size // 2) ** 2, 128)
        self.linear2 = nn.Linear(128, num_classes)

    @property
    def last_layer(self) -> nn.Module:
        return self.linear2

    def forward(
        self,
        x: Tensor,
        *,
        return_repr: bool = False,
    ) -> Tensor | ForwardResult:
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.linear1(x))
        out = self.linear2(x)
        if return_repr:
            return ForwardResult(output=out, representation=x)
        return out
