import logging
import random
from collections.abc import Generator
from itertools import islice
from pathlib import Path

import hydra
import numpy as np
import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

import wandb
from src.information_plane.nhsic import calc_nhsic_plane
from src.neural_collapse import NeuralCollapseValues, calc_neural_collapse_values


# Dataset
def get_dataloader(
    cfg: DictConfig,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if cfg.dataset.name != "mnist":
        msg = "Only MNIST dataset is currently supported."
        raise NotImplementedError(msg)

    full_train_dataset = torchvision.datasets.MNIST(
        root="~/pytorch_datasets",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    test_dataset = torchvision.datasets.MNIST(
        root="~/pytorch_datasets",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    train_dataset = torch.utils.data.Subset(
        full_train_dataset,
        range(cfg.train_size),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


# Model
class MLPModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        depth: int,
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
        for _ in range(depth - 2):
            self.net.extend([nn.Linear(hidden_size, hidden_size), self.activation])
        self.net.append(nn.Linear(hidden_size, output_size))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.net:
            x = layer(x)
        return x


# Helper functions
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002


def infinite_loader(loader: torch.utils.data.DataLoader) -> Generator:
    while True:
        yield from loader


def calc_accuracy_and_loss(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Calculate accuracy and loss of the model on the given data loader."""
    model.eval()
    correct = 0
    loss_total = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)  # noqa: PLW2901
            pred = model(data)
            correct += (pred.argmax(dim=1) == target).sum().item()
            one_hots = torch.eye(pred.size(1), pred.size(1)).to(device)
            loss_total += F.mse_loss(pred, one_hots[target], reduction="sum").item()
            total += target.size(0)
    model.train()
    return correct / total, loss_total / total


@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:  # noqa: C901, PLR0915
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        job_type=cfg.wandb.job_type,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        save_code=True,
    )
    logger.info("wandb run url: %s", run.get_url())

    set_seed(cfg.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Dataset
    train_loader, test_loader = get_dataloader(cfg)

    # Model
    model = MLPModel(
        input_size=784,
        hidden_size=cfg.hidden_size,
        output_size=cfg.dataset.num_classes,
        depth=cfg.depth,
    )
    penultimate_layer = model.net[-2]  # Second to last layer

    with torch.no_grad():
        for param in model.parameters():
            param.data = param.data * cfg.init_scale

    # Optimizer
    if cfg.optimizer.name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.learning_rate,
            weight_decay=cfg.optimizer.weight_decay,
        )
    elif cfg.optimizer.name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.learning_rate,
            weight_decay=cfg.optimizer.weight_decay,
        )
    else:
        msg = f"Optimizer {cfg.optimizer.name} is not implemented."
        raise NotImplementedError(msg)

    log_interval = cfg.num_steps // 150
    one_hots = torch.eye(cfg.dataset.num_classes, cfg.dataset.num_classes).to(device)
    for time_step, (data, target) in enumerate(
        islice(infinite_loader(train_loader), cfg.num_steps),
    ):
        # Logging, following https://github.com/KindXiaoming/Omnigrok
        if (
            (time_step < 30)  # noqa: PLR2004
            or (time_step < 150 and time_step % 10 == 0)  # noqa: PLR2004
            or time_step % log_interval == 0
        ):
            train_acc, train_loss = calc_accuracy_and_loss(model, train_loader, device)
            test_acc, test_loss = calc_accuracy_and_loss(model, test_loader, device)

            with torch.no_grad():
                weight_norm = sum(
                    torch.norm(param, p=2) for param in model.parameters()
                ).item()
                last_layer_norm = torch.norm(model.net[-1].weight, p=2).item()

                neural_collapse_values = calc_neural_collapse_values(
                    model,
                    penultimate_layer,
                    train_loader,
                    cfg.dataset.num_classes,
                    device,
                )
                if cfg.calc_nhsic:
                    nhsic_zx, nhsic_zy = calc_nhsic_plane(
                        model,
                        penultimate_layer,
                        test_loader,
                        cfg.dataset.num_classes,
                        10,
                        device,
                    )
                    logger.info(
                        "nHSIC(z; x) = %.4f, nHSIC(z; y) = %.4f",
                        nhsic_zx,
                        nhsic_zy,
                    )
            wandb_log = {
                "train_accuracy": train_acc,
                "train_loss": train_loss,
                "test_accuracy": test_acc,
                "test_loss": test_loss,
                "weight_norm": weight_norm,
                "last_layer_norm": last_layer_norm,
                "nc1_score": neural_collapse_values.nc1_score,
                "nc2_score": neural_collapse_values.nc2_score,
                "within_class_variance": neural_collapse_values.within_class_variance,
                "between_class_variance": neural_collapse_values.between_class_variance,
                "time_step": time_step,
            }
            if cfg.calc_nhsic:
                wandb_log.update(
                    {
                        "nhsic_zx": nhsic_zx,
                        "nhsic_zy": nhsic_zy,
                    },
                )
            wandb.log(wandb_log)
            logger.info(
                "Step %d: train_acc=%.4f, train_loss=%.4f, "
                "test_acc=%.4f, test_loss=%.4f, "
                "weight_norm=%.4f, last_layer_norm=%.4f, "
                "nc1_score=%.4f, nc2_score=%.4f, "
                "within_class_variance=%.4f, "
                "between_class_variance=%.4f, ",
                time_step,
                train_acc,
                train_loss,
                test_acc,
                test_loss,
                weight_norm,
                last_layer_norm,
                neural_collapse_values.nc1_score,
                neural_collapse_values.nc2_score,
                neural_collapse_values.within_class_variance,
                neural_collapse_values.between_class_variance,
            )
            for param in optimizer.param_groups:
                logger.info(
                    "Optimizer param group: lr=%.6f, weight_decay=%.6f",
                    param["lr"],
                    param["weight_decay"],
                )

        # Train
        model.train()
        data, target = data.to(device), target.to(device)  # noqa: PLW2901
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, one_hots[target], reduction="sum")
        loss.backward()
        optimizer.step()

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info("output_dir: %s", output_dir)
    run.config["output_dir"] = str(output_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
