from torch import nn


class RepresentationHook:
    """Class to register a forward hook on a layer and collect its outputs."""

    def __init__(self, layer: nn.Module) -> None:
        self.layer = layer
        self.representation = None
        self.handle = self.layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):  # noqa: A002, ANN001, ANN201, ARG002
        self.representation = output.view(output.shape[0], -1).detach()

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def get_representation(self) -> nn.Module:
        """Return the collected representation."""
        if self.representation is None:
            msg = "No representation collected yet."
            raise ValueError(msg)
        return self.representation

    def __del__(self) -> None:
        self.remove()
