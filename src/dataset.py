from typing import Any

import torchvision
from torch.utils.data import DataLoader, Dataset, Subset


def get_dataset(
    dataset_name: str,
    train_size: int | None = None,
) -> tuple[Dataset, Dataset]:
    if dataset_name != "mnist":
        msg = "Only MNIST dataset is currently supported."
        raise NotImplementedError(msg)

    if dataset_name == "mnist":
        image_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
            ],
        )
        train_dataset = torchvision.datasets.MNIST(
            root="~/pytorch_datasets",
            train=True,
            transform=image_transform,
            download=True,
        )
        test_dataset = torchvision.datasets.MNIST(
            root="~/pytorch_datasets",
            train=False,
            transform=image_transform,
            download=True,
        )
    else:
        msg = f"Dataset {dataset_name} is not implemented."
        raise NotImplementedError(msg)

    if train_size is not None:
        train_dataset = Subset(train_dataset, range(train_size))

    return train_dataset, test_dataset


def get_dataloader(
    dataset_name: str,
    batch_size: int,
    train_size: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_dataset, test_dataset = get_dataset(
        dataset_name=dataset_name,
        train_size=train_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


class AutoencoderDataset(Dataset):
    """Construct dataset for autoencoder training from another dataset."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset: Dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        x = self.dataset[index]
        return (x, x)
