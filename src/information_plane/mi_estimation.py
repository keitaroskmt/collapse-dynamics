# Mutual information estimation used in the ICML 2023 paper "How does information bottleneck help deep learning?".
# Code: https://github.com/xu-ji/information-bottleneck/tree/main

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytest
import torch
from torch import Tensor, nn


@dataclass
class MIEstimationConfig:
    """Configuration for mutual information estimation."""

    eval_size: int = 2000  # sz1
    eval_size_cond_y: int = 1000
    ref_size: int = 400  # sz2
    batch_size: int = 100
    std: float = 0.1
    mode: Literal["mc", "jensen"] = "mc"


def estimate_mi_zx(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    mi_config: MIEstimationConfig = None,
) -> float:
    """Estimate the mutual information I(Z; X) between the target layer output Z and the input X."""
    if mi_config is None:
        mi_config = MIEstimationConfig()

    model.eval()

    features = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)  # noqa: PLW2901
            forward_result = model(data, return_repr=True)
            features.append(forward_result.representation)
    features = torch.cat(features, dim=0)

    num_samples, _ = features.size()
    rng = np.random.default_rng()
    eval_index = torch.tensor(
        rng.choice(num_samples, mi_config.eval_size, replace=True),
    )
    ref_index = torch.tensor(
        rng.choice(num_samples, mi_config.ref_size, replace=True),
    )
    eval_features = features[eval_index]
    ref_features = features[ref_index]
    feature_dim = eval_features.size(1)

    log_prob_numerator = (
        torch.distributions.normal.Normal(0, mi_config.std)
        .log_prob(torch.zeros(feature_dim))
        .sum()
    )
    dist = torch.distributions.normal.Normal(ref_features, mi_config.std)
    log_prob_denominator = compute_log_likelihood(eval_features, dist, mi_config)
    return (log_prob_numerator - log_prob_denominator).mean().item()


def estimate_mi_zx_cond_y(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    num_classes: int,
    device: torch.device,
    mi_config: MIEstimationConfig = None,
) -> float:
    """Estimate the mutual information I(Z; X | Y) between the target layer representations Z and the input X conditioned on the labels Y.

    NOTE: We currently expect the balanced classes, i.e., p_Y is a uniform distribution.
    """
    if mi_config is None:
        mi_config = MIEstimationConfig()
    model.eval()

    features = [[] for _ in range(num_classes)]
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)  # noqa: PLW2901
            forward_result = model(data, return_repr=True)
            for c in range(num_classes):
                class_mask = target == c
                features[c].append(forward_result.representation[class_mask])
    features = [torch.cat(x, dim=0) for x in features]

    rng = np.random.default_rng()
    estimations = []
    for c in range(num_classes):
        num_samples, _ = features[c].size()
        eval_index = torch.tensor(
            rng.choice(num_samples, mi_config.eval_size, replace=True),
        )
        ref_index = torch.tensor(
            rng.choice(num_samples, mi_config.ref_size, replace=True),
        )
        eval_features = features[c][eval_index]
        ref_features = features[c][ref_index]
        feature_dim = eval_features.size(1)

        log_prob_numerator = (
            torch.distributions.normal.Normal(0, mi_config.std)
            .log_prob(torch.zeros(feature_dim))
            .sum()
        )
        dist = torch.distributions.normal.Normal(ref_features, mi_config.std)
        log_prob_denominator = compute_log_likelihood(
            eval_features,
            dist,
            mi_config,
        )
        estimations.append((log_prob_numerator - log_prob_denominator).mean().item())
    return np.mean(estimations).item()


def compute_log_likelihood(
    eval_features: Tensor,
    dist: torch.distributions.distribution.Distribution,
    mi_config: MIEstimationConfig,
) -> Tensor:
    if dist.mean.size(0) != mi_config.ref_size:
        msg = (
            f"Expected distribution mean shape ({mi_config.ref_size}, dimension), "
            f"but got the first size {dist.mean.size(0)} "
        )
        raise ValueError(msg)

    log_likelihood_list = []
    for eval_features_batch in torch.split(eval_features, mi_config.batch_size):
        eval_features_expanded = eval_features_batch.unsqueeze(1).expand(
            -1,
            mi_config.ref_size,
            -1,
        )
        log_prob = dist.log_prob(eval_features_expanded).sum(
            dim=2,
        )  # Shape: (batch_size, ref_size)

        if mi_config.mode == "mc":
            log_likelihood = -np.log(mi_config.eval_size) + torch.logsumexp(
                log_prob,
                dim=1,
            )
        elif mi_config.mode == "jensen":
            log_likelihood = log_prob.mean(dim=1)
        else:
            pytest.fail("Unreachable")
        log_likelihood_list.append(log_likelihood)
    return torch.cat(log_likelihood_list, dim=0)  # Shape: (eval_size,)
