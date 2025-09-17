from abc import ABC, abstractmethod

from torch import Tensor, nn


class EmbeddingLayer(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass


class ImageEmbedding(EmbeddingLayer):
    def __init__(
        self,
        in_channels: int = 1,
        patch_size: int = 4,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)  # (B, hidden_size, H / patch_size, W / patch_size)
        x = x.flatten(2)  # (B, hidden_size, num_patches)
        return x.transpose(1, 2)  # (B, num_patches, hidden_size)
