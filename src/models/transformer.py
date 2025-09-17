import torch
from torch import Tensor, nn

from src.models.embedding import EmbeddingLayer
from src.models.utils import ForwardResult


class OneLayerTransformer(nn.Module):
    max_len: int = 512

    def __init__(
        self,
        embedding: EmbeddingLayer,
        hidden_size: int = 128,
        n_head: int = 4,
        output_size: int = 10,
    ) -> None:
        super().__init__()

        self.embedding = embedding

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_len + 1, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_head,
            dim_feedforward=2 * hidden_size,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Linear(hidden_size, output_size)

    def reset_init(self, init_scale: float = 1.0) -> None:
        for name, m in self.named_modules():
            if name.endswith(("linear1", "linear2")):
                m.weight.data = m.weight.data * init_scale
                m.bias.data = m.bias.data * init_scale

    @property
    def last_layer(self) -> nn.Module:
        return self.head

    def forward(
        self,
        x: Tensor,
        *,
        return_repr: bool = False,
    ) -> Tensor | ForwardResult:
        x = self.embedding(x)  # (B, num_patches, hidden_size)

        if x.size(1) > self.max_len:
            msg = f"Sequence length {x.size(1)} exceeds maximum length {self.max_len}"
            raise ValueError(msg)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, hidden_size)
        z = torch.cat([cls_tokens, x], dim=1)  # (B, 1 + num_patches, hidden_size)
        z = z + self.pos_embedding[:, : z.size(1), :]
        z = self.encoder(z)
        out = self.head(z[:, 0, :])
        if return_repr:
            return ForwardResult(output=out, representation=z[:, 0, :])
        return out
