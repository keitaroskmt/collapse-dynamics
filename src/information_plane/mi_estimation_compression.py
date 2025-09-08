# Mutual information estimation based on the compression with a trained auto-encoder.
# The technique and code are referred from ICLR 2024 paper:
# "Information Bottleneck Analysis of Deep Neural Networks via Lossy Compression".
# https://openreview.net/forum?id=huGECz8dPp

import torch
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from torch import nn

from src.information_plane.compression.kl_estimator import MIEstimator
from src.information_plane.compression.train_autoencoder import (
    MIEstimationCompressionConfig,
    prepare_input_autoencoder,
)
from src.representation import RepresentationHook


def estimate_mi_zx(  # noqa: PLR0913
    model: nn.Module,
    target_layer: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: DictConfig,
    mi_config: MIEstimationCompressionConfig = None,
) -> float:
    if mi_config is None:
        mi_config = MIEstimationCompressionConfig()

    representation_hook = RepresentationHook(target_layer)
    model.eval()

    input_ae = prepare_input_autoencoder(cfg=cfg, mi_config=mi_config, device=device)
    input_ae.eval()
    x_compressed = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)  # noqa: PLW2901
            x_compressed.append(input_ae.encode(data))
    x_compressed = torch.cat(x_compressed, dim=0)

    features = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)  # noqa: PLW2901
            model(data)
            features.append(representation_hook.get_representation())
    features = torch.cat(features, dim=0)
    z_compressed = PCA(n_components=mi_config.latent_dim).fit_transform(
        features.numpy(),
    )

    mi_zx_estimator = MIEstimator(
        entropy_estimator_params={
            "method": "KL",
            "functional_params": {"k_neighbors": 5},
        },
        verbose=False,
    )
    return mi_zx_estimator.fit_estimate(z_compressed, x_compressed)


def estimate_mi_zy(
    model: nn.Module,
    target_layer: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    mi_config: MIEstimationCompressionConfig = None,
) -> float:
    if mi_config is None:
        mi_config = MIEstimationCompressionConfig()

    representation_hook = RepresentationHook(target_layer)
    model.eval()

    targets = loader.dataset.targets
    features = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)  # noqa: PLW2901
            model(data)
            features.append(representation_hook.get_representation())
    features = torch.cat(features, dim=0)
    z_compressed = PCA(n_components=mi_config.latent_dim).fit_transform(
        features.numpy(),
    )

    mi_zx_estimator = MIEstimator(
        y_is_discrete=True,
        entropy_estimator_params={
            "method": "KL",
            "functional_params": {"k_neighbors": 5},
        },
        verbose=False,
    )
    return mi_zx_estimator.fit_estimate(z_compressed, targets)
