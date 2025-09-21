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
    scale_means: float


def calc_neural_collapse_values(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    num_classes: int,
    device: torch.device,
) -> NeuralCollapseValues:
    """Calculate the neural collapse values for the output of the penultimate layer."""
    model.eval()

    sum_class_reps = None
    sum_class_norm = torch.zeros(num_classes, device=device)
    sum_norm = 0.0
    num_samples = torch.zeros(num_classes, device=device)
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)  # noqa: PLW2901
            forward_result = model(data, return_repr=True)
            reps = forward_result.representation

            if sum_class_reps is None:
                sum_class_reps = torch.zeros((num_classes, reps.size(1)), device=device)

            sum_class_reps.index_add_(dim=0, index=target, source=reps)
            sum_class_norm.index_add_(
                dim=0,
                index=target,
                source=reps.pow(2).sum(dim=1),
            )
            num_samples += torch.bincount(target, minlength=num_classes)
            sum_norm += reps.norm(p=2, dim=1).sum().item()

    class_means = sum_class_reps / num_samples.unsqueeze(1)
    global_mean = sum_class_reps.sum(dim=0) / num_samples.sum()

    within_class_variance_sum = (
        sum_class_norm - num_samples * class_means.pow(2).sum(dim=1)
    ).sum()
    within_class_variance = (within_class_variance_sum / num_samples.sum()).item()

    class_ratio = num_samples / num_samples.sum()
    between_class_variance_sum = (
        (class_means - global_mean.unsqueeze(0)).pow(2).sum(dim=1)
    )
    between_class_variance = (between_class_variance_sum * class_ratio).sum().item()
    scale_means = sum_norm / sum(num_samples)

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
        within_class_variance=within_class_variance,
        between_class_variance=between_class_variance,
        scale_means=scale_means,
    )


if __name__ == "__main__":
    import time

    import torchvision

    image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
            torchvision.transforms.Lambda(lambda x: x.view(-1)),
        ],
    )
    train_dataset = torchvision.datasets.MNIST(
        root="~/pytorch_datasets",
        train=True,
        transform=image_transform,
        download=True,
    )
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)
    num_classes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Original calculation
    def f_original() -> tuple[float, float]:
        class_means = {}
        num_samples = [0 for _ in range(num_classes)]
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)  # noqa: PLW2901

                for c in range(num_classes):
                    class_mask = target == c
                    class_reps = data[class_mask]
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

                for c in range(num_classes):
                    class_mask = target == c
                    class_reps = data[class_mask]
                    if class_reps.size(0) > 0:
                        within_class_variance += (
                            torch.norm(class_reps - class_means[c], p=2) ** 2
                        )
        within_class_variance /= sum(num_samples)

        global_mean = torch.stack(
            [
                rep * sz
                for rep, sz in zip(class_means.values(), num_samples, strict=True)
            ],
            dim=0,
        ).sum(dim=0) / sum(num_samples)
        between_class_variance = 0.0
        for c in range(num_classes):
            between_class_variance += torch.norm(
                class_means[c] - global_mean,
                p=2,
            ) ** 2 * (num_samples[c] / sum(num_samples))
        return within_class_variance.item(), between_class_variance.item()

    def f_new() -> tuple[float, float]:
        sum_class_reps = None
        sum_class_norm = torch.zeros(num_classes, device=device)
        sum_norm = 0.0
        num_samples = torch.zeros(num_classes, device=device)
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)  # noqa: PLW2901

                if sum_class_reps is None:
                    sum_class_reps = torch.zeros(
                        (num_classes, data.size(1)),
                        device=device,
                    )

                sum_class_reps.index_add_(dim=0, index=target, source=data)
                sum_class_norm.index_add_(
                    dim=0,
                    index=target,
                    source=data.pow(2).sum(dim=1),
                )
                num_samples += torch.bincount(target, minlength=num_classes)
                sum_norm += data.norm(p=2, dim=1).sum().item()

        class_means = sum_class_reps / num_samples.unsqueeze(1)
        global_mean = sum_class_reps.sum(dim=0) / num_samples.sum()

        within_class_variance_sum = (
            sum_class_norm - num_samples * class_means.pow(2).sum(dim=1)
        ).sum()
        within_class_variance = (within_class_variance_sum / num_samples.sum()).item()

        class_ratio = num_samples / num_samples.sum()
        between_class_variance_sum = (
            (class_means - global_mean.unsqueeze(0)).pow(2).sum(dim=1)
        )
        between_class_variance = (between_class_variance_sum * class_ratio).sum().item()
        return within_class_variance, between_class_variance

    t0 = time.time()
    wcv1, bcv1 = f_original()
    t1 = time.time()
    wcv2, bcv2 = f_new()
    t2 = time.time()

    print(f"Original: {wcv1=}, {bcv1=}, {t1 - t0:.4f} sec")
    print(f"New:      {wcv2=}, {bcv2=}, {t2 - t1:.4f} sec")
    print(f"Diff:     {wcv2 - wcv1=}, {bcv2 - bcv1=}")
