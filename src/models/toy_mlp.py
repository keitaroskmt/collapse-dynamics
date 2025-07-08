from torch import Tensor, nn


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

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.net:
            x = layer(x)
        return x
