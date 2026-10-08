import torch
from torch import Tensor, nn

from neosr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class ncc_loss(nn.Module):
    """Normalized Cross-Correlation loss.

    Args:
    ----
        loss_weight (float): weight for the loss. Default: 1.0
    """

    def __init__(self, loss_weight: float = 1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def _cc(self, net_output: Tensor, gt: Tensor):
        if net_output.ndim != 4 or net_output.shape != gt.shape:
            msg = "NCC expects prediction and GT with the same NCHW shape."
            raise ValueError(msg)

        # Accumulate low-precision inputs in FP32 while preserving FP64.
        dtype = torch.promote_types(net_output.dtype, gt.dtype)
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        x = net_output.to(dtype=dtype).flatten(2)
        y = gt.to(dtype=dtype).flatten(2)
        x = x - x.mean(dim=-1, keepdim=True)
        y = y - y.mean(dim=-1, keepdim=True)

        # Floor spatial mean variances before sqrt for finite backward at zero.
        # For [0, 1] images, 1e-8 corresponds to a standard deviation of 1e-4.
        sx = x.square().mean(dim=-1).clamp_min(1e-8).sqrt()
        sy = y.square().mean(dim=-1).clamp_min(1e-8).sqrt()
        cc = (x * y).mean(dim=-1) / sx / sy
        return cc.clamp(-1, 1).mean()

    def forward(self, net_output: Tensor, gt: Tensor):
        cc_value = self._cc(net_output, gt)
        return (1 - ((cc_value + 1) * 0.5)) * self.loss_weight
