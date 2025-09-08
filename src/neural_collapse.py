from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class NeuralCollapseValues:
    """Dataclass to store values of neural collapse experiments."""

    nc1_score: float
    nc2_score: float
    within_class_variance: float
    between_class_variance: float


def calc_neural_collapse_values(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    num_classes: int,
    device: torch.device,
) -> NeuralCollapseValues:
    """Calculate the neural collapse values for the output of the penultimate layer."""
    model.eval()

    class_means = {}
    num_samples = [0 for _ in range(num_classes)]
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)  # noqa: PLW2901
            forward_result = model(data, return_repr=True)

            for c in range(num_classes):
                class_mask = target == c
                class_reps = forward_result.representation[class_mask]
                if c not in class_means:
                    class_means[c] = class_reps.sum(dim=0)
                    num_samples[c] += class_mask.sum().item()
                else:
                    class_means[c] += class_reps.sum(dim=0)
                    num_samples[c] += class_mask.sum().item()
    class_means = {c: cm / num_samples[c] for c, cm in class_means.items()}

    within_class_variance = 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)  # noqa: PLW2901
            forward_result = model(data, return_repr=True)

            for c in range(num_classes):
                class_mask = target == c
                class_reps = forward_result.representation[class_mask]
                if class_reps.size(0) > 0:
                    within_class_variance += (
                        torch.norm(class_reps - class_means[c], p=2) ** 2
                    )
    within_class_variance /= sum(num_samples)

    global_mean = torch.stack(
        [rep * sz for rep, sz in zip(class_means.values(), num_samples, strict=True)],
        dim=0,
    ).sum(dim=0) / sum(num_samples)
    between_class_variance = 0.0
    for c in range(num_classes):
        between_class_variance += torch.norm(class_means[c] - global_mean, p=2) ** 2 * (
            num_samples[c] / sum(num_samples)
        )
    nc1_score = within_class_variance / (between_class_variance + 1e-8)

    if num_classes > global_mean.size(0):
        msg = (
            "Currently, number of classes are assumed to be less than or equal to the "
            "dimension of the representation."
        )
        raise NotImplementedError(msg)
    class_means_matrix = torch.stack(
        [class_means[c] for c in range(num_classes)],
        dim=0,
    )
    nc2_score = (
        torch.linalg.cond(class_means_matrix @ class_means_matrix.T).sqrt().item()
    )

    return NeuralCollapseValues(
        nc1_score=nc1_score,
        nc2_score=nc2_score,
        within_class_variance=within_class_variance.item(),
        between_class_variance=between_class_variance.item(),
    )
