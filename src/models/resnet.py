import torch
from torch import Tensor, nn

from src.models.utils import ForwardResult

# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return nn.functional.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = nn.functional.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return nn.functional.relu(out)


class ResNet(nn.Module):
    """ResNet model."""

    def __init__(
        self,
        model_name: str = "resnet18",
        input_channels: int = 3,
        output_size: int = 10,
    ) -> None:
        super().__init__()
        block, dim_out, num_blocks = get_model_settings(model_name)

        self.init_in_planes = 64
        self.in_planes = self.init_in_planes

        self.conv1 = nn.Conv2d(
            input_channels,
            self.in_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(
            block,
            self.init_in_planes,
            num_blocks[0],
            stride=1,
        )
        self.layer2 = self._make_layer(
            block,
            self.init_in_planes * 2,
            num_blocks[1],
            stride=2,
        )
        self.layer3 = self._make_layer(
            block,
            self.init_in_planes * 4,
            num_blocks[2],
            stride=2,
        )
        self.layer4 = self._make_layer(
            block,
            self.init_in_planes * 8,
            num_blocks[3],
            stride=2,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(dim_out, output_size)

    def _make_layer(
        self,
        block: type[BasicBlock | Bottleneck],
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride_ in strides:
            layers.append(block(self.in_planes, planes, stride_))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    @property
    def last_layer(self) -> nn.Module:
        return self.linear

    def forward(
        self,
        x: Tensor,
        *,
        return_repr: bool = False,
    ) -> Tensor | ForwardResult:
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        out = self.linear(x)
        if return_repr:
            return ForwardResult(output=out, representation=x)
        return out


def get_model_settings(
    model_name: str,
) -> tuple[type[BasicBlock | Bottleneck], int, list[int]]:
    if model_name == "resnet18":
        block = BasicBlock
        dim_out = 512
        num_blocks = [2, 2, 2, 2]
    elif model_name == "resnet34":
        block = BasicBlock
        dim_out = 512
        num_blocks = [3, 4, 6, 3]
    elif model_name == "resnet50":
        block = Bottleneck
        dim_out = 2048
        num_blocks = [3, 4, 6, 3]
    elif model_name == "resnet101":
        block = Bottleneck
        dim_out = 2048
        num_blocks = [3, 4, 23, 3]
    elif model_name == "resnet152":
        block = Bottleneck
        dim_out = 2048
        num_blocks = [3, 8, 36, 3]
    else:
        msg = f"ResNet model name {model_name} is invalid"
        raise ValueError(msg)
    return block, dim_out, num_blocks
