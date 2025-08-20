# The implementation of the autoencoder model for MI estimation.
# The following code is based on https://arxiv.org/abs/2305.08013

import torch
from torch import Tensor, nn


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_size: int = 28,
        input_channels: int = 1,
        latent_dim: int = 4,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError
        self.tanh = torch.nn.Tanh()
        self.dropout = nn.Dropout(p=0.1)

        num_channel_scale = 1 if input_channels == 1 else 2

        self.conv1 = nn.Conv2d(
            input_channels,
            8 * num_channel_scale,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            8 * num_channel_scale,
            16 * num_channel_scale,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            16 * num_channel_scale,
            32 * num_channel_scale,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear1 = nn.Linear(32 * num_channel_scale * (input_size // 8) ** 2, 128)
        self.linear2 = nn.Linear(128, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(x)
        x = self.activation(self.pool(self.conv1(x)))
        x = self.dropout(x)
        x = self.activation(self.pool(self.conv2(x)))
        x = self.dropout(x)
        x = self.activation(self.pool(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.linear1(x))
        return self.tanh(self.linear2(x))


class ConvDecoder(nn.Module):
    def __init__(
        self,
        input_size: int = 28,
        input_channels: int = 1,
        latent_dim: int = 4,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError
        num_channel_scale = 1 if input_channels == 1 else 2

        self.conv1 = nn.Conv2d(
            32 * num_channel_scale,
            16 * num_channel_scale,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            16 * num_channel_scale,
            8 * num_channel_scale,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            8 * num_channel_scale,
            input_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.upsample = nn.Upsample(scale_factor=2)

        self.linear1 = nn.Linear(latent_dim, 128)
        self.linear2 = nn.Linear(128, 32 * num_channel_scale * (input_size // 4) ** 2)

        self.num_channel_scale = num_channel_scale
        self.input_size = input_size

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = torch.reshape(
            x,
            (
                -1,
                32 * self.num_channel_scale,
                self.input_size // 4,
                self.input_size // 4,
            ),
        )
        x = self.activation(self.upsample(self.conv1(x)))
        x = self.activation(self.upsample(self.conv2(x)))
        return self.conv3(x)


class Autoencoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        sigma: float = 0.1,
    ) -> None:
        super().__init__()
        # Encoder and decoder.
        self.encoder = encoder
        self.decoder = decoder

        # Noise.
        self.agn = AdditiveGaussianNoise(sigma=sigma, enabled_on_inference=False)

    def forward(self, x: torch.tensor) -> torch.tensor:
        latent = self.encoder(x)
        latent = self.agn(latent)
        return self.decoder(latent)

    def encode(self, x: torch.tensor) -> torch.tensor:
        return self.encoder(x)

    def decode(self, x: torch.tensor) -> torch.tensor:
        return self.decoder(x)


class AdditiveGaussianNoise(nn.Module):
    """Additive Gaussian noise layer."""

    def __init__(
        self,
        sigma: float = 0.1,
        *,
        relative_scale: bool = True,
        enabled_on_inference: bool = False,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.relative_scale = relative_scale
        self.enabled_on_inference = enabled_on_inference

    def forward(self, x: torch.tensor) -> torch.tensor:
        if (self.training or self.enabled_on_inference) and self.sigma > 0:
            scale = self.sigma * x if self.relative_scale else self.sigma
            noise = torch.randn_like(x) * scale
            return x + noise
        return x


if __name__ == "__main__":
    # Example usage
    encoder = ConvEncoder(input_size=28, input_channels=1, latent_dim=4)
    decoder = ConvDecoder(input_size=28, input_channels=1, latent_dim=4)
    autoencoder = Autoencoder(encoder, decoder)

    # Test the autoencoder with a random tensor
    test_input = torch.randn(1, 1, 28, 28)
    output = autoencoder(test_input)
    print(test_input.shape, output.shape)  # noqa: T201
