import logging
import random
from collections.abc import Generator
from itertools import islice
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn import functional as F  # noqa: N812

import wandb
from src.dataset import get_dataloader
from src.information_plane.mi_estimation import (
    MIEstimationConfig,
    estimate_mi_zx,
    estimate_mi_zx_cond_y,
)
from src.information_plane.mi_estimation_compression import (
    MIEstimationCompressionConfig,
)
from src.information_plane.mi_estimation_compression import (
    estimate_mi_zx as estimate_mi_zx_compression,
)
from src.information_plane.mi_estimation_compression import (
    estimate_mi_zy as estimate_mi_zy_compression,
)
from src.information_plane.nhsic import calc_nhsic_plane
from src.models.toy_cnn import CNNModel
from src.models.toy_mlp import MLPModel
from src.neural_collapse import calc_neural_collapse_values


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
    train_loader, test_loader = get_dataloader(
        dataset_name=cfg.dataset.name,
        batch_size=cfg.batch_size,
        train_size=cfg.train_size,
    )

    # Model
    if cfg.model.name == "toy_mlp":
        model = MLPModel(
            input_size=784,
            hidden_size=cfg.model.hidden_size,
            output_size=cfg.dataset.num_classes,
            depth=cfg.model.depth,
            last_layer_act=cfg.model.last_layer_act,
        )
    elif cfg.model.name == "toy_cnn":
        model = CNNModel(
            input_size=cfg.dataset.size,
            input_channels=cfg.dataset.num_channels,
            num_classes=cfg.dataset.num_classes,
        )

    with torch.no_grad():
        if cfg.model.name == "toy_mlp":
            for param in model.parameters():
                param.data = param.data * cfg.init_scale
        elif cfg.model.name == "toy_cnn":
            if cfg.model.init_method == "linear":
                for name, param in model.named_parameters():
                    if name.startswith("linear"):
                        param.data = param.data * cfg.init_scale
            elif cfg.model.init_method == "zero_last_layer":
                for param in model.parameters():
                    param.data = param.data * cfg.init_scale
                model.last_layer.weight.data.zero_()
                if model.last_layer.bias is not None:
                    model.last_layer.bias.data.zero_()
            else:
                msg = f"Initialization method {cfg.model.init_method} is not supported."
                raise NotImplementedError(msg)

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
                last_layer_norm = torch.norm(model.last_layer.weight, p=2).item()

            neural_collapse_values = calc_neural_collapse_values(
                model,
                train_loader,
                cfg.dataset.num_classes,
                device,
            )
            if cfg.calc_nhsic:
                nhsic_zx, nhsic_zy = calc_nhsic_plane(
                    model,
                    test_loader,
                    cfg.dataset.num_classes,
                    10,
                    device,
                )
                logger.info(
                    "nHSIC(z; x): %.4f, nHSIC(z; y): %.4f",
                    nhsic_zx,
                    nhsic_zy,
                )
            if cfg.calc_mi_estimation:
                mi_zx_estimation = estimate_mi_zx(
                    model,
                    test_loader,
                    device,
                    MIEstimationConfig(mode="mc"),
                )
                mi_zx_cond_y_estimation = estimate_mi_zx_cond_y(
                    model,
                    test_loader,
                    cfg.dataset.num_classes,
                    device,
                    MIEstimationConfig(mode="mc"),
                )
                mi_zy_estimation = mi_zx_estimation - mi_zx_cond_y_estimation
                logger.info(
                    "hat{I}(Z; X): %.4f, hat{I}(Z; Y): %.4f, hat{I}(Z; X | Y): %.4f",
                    mi_zx_estimation,
                    mi_zy_estimation,
                    mi_zx_cond_y_estimation,
                )
                mi_zx_jensen_estimation = estimate_mi_zx(
                    model,
                    test_loader,
                    device,
                    MIEstimationConfig(mode="jensen"),
                )
                mi_zx_cond_y_jensen_estimation = estimate_mi_zx_cond_y(
                    model,
                    test_loader,
                    cfg.dataset.num_classes,
                    device,
                    MIEstimationConfig(mode="jensen"),
                )
                mi_zy_jensen_estimation = (
                    mi_zx_jensen_estimation - mi_zx_cond_y_jensen_estimation
                )
                logger.info(
                    "check{I}(Z; X): %.4f, check{I}(Z; Y): %.4f, check{I}(Z; X | Y): %.4f",
                    mi_zx_jensen_estimation,
                    mi_zy_jensen_estimation,
                    mi_zx_cond_y_jensen_estimation,
                )
            if cfg.calc_mi_estimation_compression:
                mi_zx_compression = estimate_mi_zx_compression(
                    model,
                    test_loader,
                    device,
                    cfg,
                    MIEstimationCompressionConfig(),
                )
                mi_zy_compression = estimate_mi_zy_compression(
                    model,
                    test_loader,
                    device,
                    MIEstimationCompressionConfig(),
                )
                logger.info(
                    "I(Z; X) estimated by comp: %.4f, I(Z; Y) estimated by comp: %.4f",
                    mi_zx_compression,
                    mi_zy_compression,
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
            if cfg.calc_mi_estimation:
                wandb_log.update(
                    {
                        "mi_zx_estimation": mi_zx_estimation,
                        "mi_zy_estimation": mi_zy_estimation,
                        "mi_zx_cond_y_estimation": mi_zx_cond_y_estimation,
                        "mi_zx_jensen_estimation": mi_zx_jensen_estimation,
                        "mi_zy_jensen_estimation": mi_zy_jensen_estimation,
                        "mi_zx_cond_y_jensen_estimation": mi_zx_cond_y_jensen_estimation,
                    },
                )
            if cfg.calc_mi_estimation_compression:
                wandb_log.update(
                    {
                        "mi_zx_compression": mi_zx_compression,
                        "mi_zy_compression": mi_zy_compression,
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
            # Save model
            if cfg.save_model:
                output_dir = (
                    Path(__file__).parent.resolve()
                    / "saved_models"
                    / cfg.model.name
                    / cfg.dataset.name
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Model output directory: %s", output_dir)
                torch.save(
                    model.state_dict(),
                    output_dir / f"model_step_{time_step}.pt",
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
