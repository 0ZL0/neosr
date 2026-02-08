from typing import Any

import torch
from torch import Tensor, nn

from neosr.utils.registry import LOSS_REGISTRY

try:
    import pyiqa
except ModuleNotFoundError:
    pyiqa = None


@LOSS_REGISTRY.register()
class pyiqa_loss(nn.Module):
    """Adapter to use any pyiqa metric as training loss."""

    def __init__(
        self,
        metric_name: str,
        loss_weight: float = 1.0,
        loss_reduction: str = "mean",
        clamp_input: bool = True,
        device: str | None = None,
        **metric_kwargs: Any,
    ) -> None:
        super().__init__()
        if pyiqa is None:
            msg = (
                "pyiqa is not installed. Install it with `uv pip install pyiqa` "
                "before using pyiqa_loss."
            )
            raise ImportError(msg)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.metric_name = metric_name
        self.clamp_input = clamp_input
        self.loss_weight = loss_weight
        self.loss = pyiqa.create_metric(
            metric_name,
            as_loss=True,
            loss_reduction=loss_reduction,
            device=device,
            **metric_kwargs,
        )
        self.metric_mode = getattr(self.loss, "metric_mode", None)

    def forward(self, pred: Tensor, target: Tensor | None = None) -> Tensor:
        if self.clamp_input:
            pred = torch.clamp(pred, 0.0, 1.0)
            if target is not None:
                target = torch.clamp(target, 0.0, 1.0)

        if self.metric_mode == "FR":
            if target is None:
                msg = (
                    f"pyiqa metric '{self.metric_name}' is FR and requires a "
                    "reference target tensor."
                )
                raise ValueError(msg)
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred)

        if isinstance(loss, tuple):
            loss = loss[0]
        if not isinstance(loss, Tensor):
            loss = torch.tensor(loss, device=pred.device, dtype=pred.dtype)
        if loss.ndim > 0:
            loss = loss.mean()
        return loss * self.loss_weight
