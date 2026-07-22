"""Residual Local Feature Network (RLFN).

Adapted from the official ByteDance implementation:
https://github.com/bytedance/RLFN
"""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F

from neosr.archs.arch_util import net_opt
from neosr.utils.registry import ARCH_REGISTRY

upscale, _ = net_opt()


def _conv(in_channels: int, out_channels: int, kernel_size: int) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2
    )


class ESA(nn.Module):
    def __init__(self, esa_channels: int, num_feat: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, esa_channels, 1)
        self.conv_f = nn.Conv2d(esa_channels, esa_channels, 1)
        self.conv2 = nn.Conv2d(esa_channels, esa_channels, 3, stride=2)
        self.conv3 = nn.Conv2d(esa_channels, esa_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(esa_channels, num_feat, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, x.shape[-2:], mode="bilinear", align_corners=False)
        cf = self.conv_f(c1_)
        return x * self.sigmoid(self.conv4(c3 + cf))


class RLFB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int | None = None,
        out_channels: int | None = None,
        esa_channels: int = 16,
    ) -> None:
        super().__init__()
        mid_channels = in_channels if mid_channels is None else mid_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.c1_r = _conv(in_channels, mid_channels, 3)
        self.c2_r = _conv(mid_channels, mid_channels, 3)
        self.c3_r = _conv(mid_channels, in_channels, 3)
        self.c5 = _conv(in_channels, out_channels, 1)
        self.esa = ESA(esa_channels, out_channels)
        self.act = nn.LeakyReLU(0.05, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.act(self.c1_r(x))
        out = self.act(self.c2_r(out))
        out = self.act(self.c3_r(out)) + x
        return self.esa(self.c5(out))


class _RLFN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feature_channels: int,
        mid_channels: int,
        num_blocks: int,
        scale: int,
        esa_channels: int,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, feature_channels, mid_channels) <= 0:
            msg = "All channel counts must be positive."
            raise ValueError(msg)
        if num_blocks <= 0:
            msg = "num_blocks must be positive."
            raise ValueError(msg)
        if esa_channels <= 0:
            msg = "esa_channels must be positive."
            raise ValueError(msg)
        if not isinstance(scale, int) or scale <= 0:
            msg = f"upscale must be a positive integer, got {scale!r}."
            raise ValueError(msg)

        self.conv_1 = _conv(in_channels, feature_channels, 3)
        for index in range(1, num_blocks + 1):
            setattr(
                self,
                f"block_{index}",
                RLFB(feature_channels, mid_channels, feature_channels, esa_channels),
            )
        self.num_blocks = num_blocks
        self.conv_2 = _conv(feature_channels, feature_channels, 3)
        self.upsampler = nn.Sequential(
            _conv(feature_channels, out_channels * scale**2, 3), nn.PixelShuffle(scale)
        )

    def forward(self, x: Tensor) -> Tensor:
        if min(x.shape[-2:]) < 15:
            msg = "RLFN requires input height and width of at least 15 pixels."
            raise ValueError(msg)
        features = self.conv_1(x)
        out = features
        for index in range(1, self.num_blocks + 1):
            out = getattr(self, f"block_{index}")(out)
        return self.upsampler(self.conv_2(out) + features)


def _resolve_scale(upscale_factor: int, scale: int | None) -> int:
    return upscale_factor if scale is None else scale


@ARCH_REGISTRY.register()
class rlfn(_RLFN):
    """Full RLFN model (52 features, six RLFB blocks)."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        feature_channels: int = 52,
        num_blocks: int = 6,
        esa_channels: int = 16,
        upscale: int = upscale,
        scale: int | None = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            feature_channels,
            feature_channels,
            num_blocks,
            _resolve_scale(upscale, scale),
            esa_channels,
        )


@ARCH_REGISTRY.register()
class rlfn_s(_RLFN):
    """Small RLFN model (48 features, six RLFB blocks)."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        feature_channels: int = 48,
        num_blocks: int = 6,
        esa_channels: int = 16,
        upscale: int = upscale,
        scale: int | None = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            feature_channels,
            feature_channels,
            num_blocks,
            _resolve_scale(upscale, scale),
            esa_channels,
        )


@ARCH_REGISTRY.register()
class rlfn_prune(_RLFN):
    """Pruned RLFN submitted to the NTIRE 2022 efficient SR challenge."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        feature_channels: int = 46,
        mid_channels: int = 48,
        num_blocks: int = 4,
        esa_channels: int = 16,
        upscale: int = upscale,
        scale: int | None = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            feature_channels,
            mid_channels,
            num_blocks,
            _resolve_scale(upscale, scale),
            esa_channels,
        )


@ARCH_REGISTRY.register()
def rlfn_ntire(**kwargs) -> rlfn_prune:
    """Alias for the official checkpoint name ``rlfn_ntire_x4.pth``."""

    return rlfn_prune(**kwargs)
