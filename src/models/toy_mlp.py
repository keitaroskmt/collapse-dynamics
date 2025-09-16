from torch import Tensor, nn

from src.models.utils import ForwardResult


# Model
class MLPModel(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        depth: int,
        *,
        last_layer_act: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        if activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError

        self.net = nn.ModuleList()
        self.net.append(nn.Flatten())
        self.net.extend([nn.Linear(input_size, hidden_size), self.activation])
        for i in range(depth - 2):
            if not last_layer_act and i == depth - 3:
                self.net.append(nn.Linear(hidden_size, hidden_size))
            else:
                self.net.extend([nn.Linear(hidden_size, hidden_size), self.activation])
        self.net.append(nn.Linear(hidden_size, output_size))

    @property
    def last_layer(self) -> nn.Module:
        return self.net[-1]

    def forward(
        self,
        x: Tensor,
        *,
        return_repr: bool = False,
    ) -> Tensor | ForwardResult:
        if x.dim() > 2:  # noqa: PLR2004
            x = x.flatten(1)
        for layer in self.net[:-1]:
            x = layer(x)
        out = self.net[-1](x)
        if return_repr:
            return ForwardResult(output=out, representation=x)
        return out
