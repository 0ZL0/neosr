"""Enhanced Deep Residual Network (EDSR).

Adapted from the official EDSR-PyTorch implementation:
https://github.com/sanghyun-son/EDSR-PyTorch
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from neosr.archs.arch_util import net_opt
from neosr.utils.registry import ARCH_REGISTRY

upscale, _ = net_opt()


def _conv(in_channels: int, out_channels: int, kernel_size: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)


class MeanShift(nn.Conv2d):
    """Fixed RGB mean shift used by the official EDSR checkpoints."""

    def __init__(
        self,
        rgb_range: float,
        rgb_mean: tuple[float, float, float] = (0.4488, 0.4371, 0.4040),
        rgb_std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        sign: int = -1,
    ) -> None:
        super().__init__(3, 3, kernel_size=1)
        std = torch.tensor(rgb_std)
        with torch.no_grad():
            self.weight.copy_(torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1))
            self.bias.copy_(sign * rgb_range * torch.tensor(rgb_mean) / std)
        self.requires_grad_(requires_grad=False)


class ResBlock(nn.Module):
    def __init__(self, num_feat: int, res_scale: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            _conv(num_feat, num_feat, 3),
            nn.ReLU(inplace=True),
            _conv(num_feat, num_feat, 3),
        )
        self.res_scale = res_scale

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x).mul(self.res_scale) + x


class Upsampler(nn.Sequential):
    def __init__(self, scale: int, num_feat: int) -> None:
        modules: list[nn.Module] = []
        if scale > 0 and scale & (scale - 1) == 0:
            for _ in range(int(math.log2(scale))):
                modules.extend([_conv(num_feat, 4 * num_feat, 3), nn.PixelShuffle(2)])
        elif scale == 3:
            modules.extend([_conv(num_feat, 9 * num_feat, 3), nn.PixelShuffle(3)])
        else:
            msg = f"EDSR supports powers of two and scale 3, got {scale}."
            raise ValueError(msg)
        super().__init__(*modules)


@ARCH_REGISTRY.register()
class edsr(nn.Module):
    """Official EDSR network, weight-compatible with EDSR-PyTorch.

    The defaults select the full EDSR model (32 residual blocks and 256 features).
    Set ``num_block=16`` and ``num_feat=64`` for the official baseline model.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 256,
        num_block: int = 32,
        upscale: int = upscale,
        res_scale: float = 0.1,
        img_range: float = 255.0,
        scale: int | None = None,
    ) -> None:
        super().__init__()
        if scale is not None:
            upscale = scale
        if num_in_ch != 3 or num_out_ch != 3:
            msg = "Official EDSR mean-shift layers require three input/output channels."
            raise ValueError(msg)
        if num_feat <= 0 or num_block <= 0:
            msg = "num_feat and num_block must be positive."
            raise ValueError(msg)
        if not math.isfinite(res_scale) or res_scale <= 0:
            msg = "res_scale must be a finite positive value."
            raise ValueError(msg)
        if not math.isfinite(img_range) or img_range <= 0:
            msg = "img_range must be a finite positive value."
            raise ValueError(msg)

        self.sub_mean = MeanShift(img_range)
        self.add_mean = MeanShift(img_range, sign=1)
        self.head = nn.Sequential(_conv(num_in_ch, num_feat, 3))
        self.body = nn.Sequential(
            *(ResBlock(num_feat, res_scale) for _ in range(num_block)),
            _conv(num_feat, num_feat, 3),
        )
        self.tail = nn.Sequential(
            Upsampler(upscale, num_feat), _conv(num_feat, num_out_ch, 3)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.head(self.sub_mean(x))
        x = self.tail(self.body(x) + x)
        return self.add_mean(x)
