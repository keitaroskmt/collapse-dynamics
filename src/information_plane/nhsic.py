import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class NormalizedHSIC:
    """Class for computing the normalized HSIC between two sets of representations.

    Currently, Gaussian and linear kernels are supported.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.epsilon = 1e-5

    def calc_score(
        self,
        x: Tensor,
        y: Tensor,
        x_kernel: str = "gaussian",
        y_kernel: str = "gaussian",
    ) -> Tensor:
        batch_size = x.size(0)
        k_x = self._kernel_matrix(x, kernel=x_kernel)
        k_y = self._kernel_matrix(y, kernel=y_kernel)

        m_i = torch.eye(batch_size).to(self.device)
        k_x_inv = torch.inverse(k_x + self.epsilon * batch_size * m_i)
        k_y_inv = torch.inverse(k_y + self.epsilon * batch_size * m_i)
        r_x = torch.matmul(k_x, k_x_inv)
        r_y = torch.matmul(k_y, k_y_inv)
        return torch.sum(r_x * r_y.T)

    def _kernel_matrix(self, x: Tensor, kernel: str = "gaussian") -> Tensor:
        if kernel == "gaussian":
            return self._gaussian_kernel(x)
        if kernel == "linear":
            return self._linear_kernel(x)
        msg = f"Unknown kernel type: {kernel}"
        raise ValueError(msg)

    def _gaussian_kernel(self, x: Tensor, sigma: float = 5.0) -> Tensor:
        x = x.view(x.size(0), -1)
        dist = torch.norm(x[:, None, :] - x[None, :, :], dim=2)
        gram = torch.exp(-(dist**2) / (2 * sigma * sigma * x.size(1)))
        centering = (
            torch.eye(x.size(0)) - torch.ones(x.size(0), x.size(0)) / x.size(0)
        ).to(self.device)
        return torch.matmul(gram, centering)

    def _linear_kernel(self, x: Tensor) -> Tensor:
        return torch.matmul(x, x.T)


def calc_nhsic_plane(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    num_classes: int,
    num_iter: int,
    device: torch.device,
) -> tuple[float, float]:
    """Calculate the normalized HSIC for the representations for the given DataLoader.

    Returns:
        [nHSIC(z; x), nHSIC(z; y)], where z is the output of the penultimate layer,
        x is the input data, and y is the one-hot encoded target labels. This estimate
        is averaged over `num_iter` times the number of batches in the loader.

    """
    nhsic = NormalizedHSIC(device)

    count = 0
    nhsic_zx = 0.0
    nhsic_zy = 0.0
    for _ in range(num_iter):
        model.eval()
        for data, target in loader:
            data, target = data.to(device), target.to(device)  # noqa: PLW2901
            with torch.no_grad():
                forward_result = model(data, return_repr=True)
            nhsic_zx += nhsic.calc_score(
                forward_result.representation,
                data.view(data.size(0), -1),
            ).item()
            nhsic_zy += nhsic.calc_score(
                forward_result.representation,
                F.one_hot(target, num_classes=num_classes).float(),
                y_kernel="linear",
            ).item()
            count += 1
    nhsic_zx /= count
    nhsic_zy /= count
    return nhsic_zx, nhsic_zy
