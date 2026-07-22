"""Efficient Long-Range Attention Network (ELAN).

Adapted from the official implementation:
https://github.com/xindongzhang/ELAN
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F

from neosr.archs.arch_util import net_opt
from neosr.utils.registry import ARCH_REGISTRY

if TYPE_CHECKING:
    from collections.abc import Sequence

upscale, _ = net_opt()


class MeanShift(nn.Conv2d):
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


class ShiftConv2d1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        weight = torch.zeros(in_channels, 1, 3, 3)
        group_size = in_channels // 5
        weight[0 * group_size : 1 * group_size, 0, 1, 2] = 1.0
        weight[1 * group_size : 2 * group_size, 0, 1, 0] = 1.0
        weight[2 * group_size : 3 * group_size, 0, 2, 1] = 1.0
        weight[3 * group_size : 4 * group_size, 0, 0, 1] = 1.0
        weight[4 * group_size :, 0, 1, 1] = 1.0
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.conv2d(x, self.weight, padding=1, groups=x.shape[1])
        return self.conv1x1(x)


class ShiftConv2d0(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        group_size = in_channels // 5
        mask = torch.zeros(out_channels, in_channels, 3, 3)
        mask[:, 0 * group_size : 1 * group_size, 1, 2] = 1.0
        mask[:, 1 * group_size : 2 * group_size, 1, 0] = 1.0
        mask[:, 2 * group_size : 3 * group_size, 2, 1] = 1.0
        mask[:, 3 * group_size : 4 * group_size, 0, 1] = 1.0
        mask[:, 4 * group_size :, 1, 1] = 1.0
        self.w = conv.weight
        self.b = conv.bias
        self.m = nn.Parameter(mask, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.w * self.m, self.b, padding=1)


class ShiftConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_type: str = "fast-training-speed",
    ) -> None:
        super().__init__()
        if conv_type == "fast-training-speed":
            self.shift_conv = ShiftConv2d1(in_channels, out_channels)
        elif conv_type == "low-training-memory":
            self.shift_conv = ShiftConv2d0(in_channels, out_channels)
        else:
            msg = f"Unsupported shift-convolution type: {conv_type}."
            raise ValueError(msg)

    def forward(self, x: Tensor) -> Tensor:
        return self.shift_conv(x)


class LFE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int = 4,
        act_type: str = "relu",
    ) -> None:
        super().__init__()
        self.conv0 = ShiftConv2d(in_channels, out_channels * expansion_ratio)
        self.conv1 = ShiftConv2d(out_channels * expansion_ratio, out_channels)
        if act_type == "linear":
            self.act: nn.Module | None = None
        elif act_type == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act_type == "gelu":
            self.act = nn.GELU()
        else:
            msg = f"Unsupported LFE activation: {act_type}."
            raise ValueError(msg)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv0(x)
        if self.act is not None:
            x = self.act(x)
        return self.conv1(x)


class GMSA(nn.Module):
    def __init__(
        self,
        channels: int,
        shifts: int,
        window_sizes: tuple[int, int, int],
        calc_attn: bool,
    ) -> None:
        super().__init__()
        self.shifts = shifts
        self.window_sizes = window_sizes
        self.calc_attn = calc_attn
        projection_channels = channels * (2 if calc_attn else 1)
        split_channels = projection_channels // 3
        self.split_chns = [split_channels] * 3
        self.project_inp = nn.Sequential(
            nn.Conv2d(channels, projection_channels, 1),
            nn.BatchNorm2d(projection_channels),
        )
        self.project_out = nn.Conv2d(channels, channels, 1)

    def forward(
        self, x: Tensor, previous_attn: list[Tensor] | None = None
    ) -> tuple[Tensor, list[Tensor]]:
        batch, _, height, width = x.shape
        chunks = torch.split(self.project_inp(x), self.split_chns, dim=1)
        outputs: list[Tensor] = []
        attentions: list[Tensor] = []
        for index, chunk_item in enumerate(chunks):
            chunk = chunk_item
            window = self.window_sizes[index]
            if self.shifts > 0:
                chunk = torch.roll(
                    chunk, shifts=(-window // 2, -window // 2), dims=(2, 3)
                )
            if previous_attn is None:
                query, value = rearrange(
                    chunk,
                    "b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c",
                    qv=2,
                    dh=window,
                    dw=window,
                )
                attention = (query @ query.transpose(-2, -1)).softmax(dim=-1)
                attentions.append(attention)
            else:
                attention = previous_attn[index]
                value = rearrange(
                    chunk,
                    "b c (h dh) (w dw) -> (b h w) (dh dw) c",
                    dh=window,
                    dw=window,
                )
                attentions.append(attention)
            output = attention @ value
            output = rearrange(
                output,
                "(b h w) (dh dw) c -> b c (h dh) (w dw)",
                b=batch,
                h=height // window,
                w=width // window,
                dh=window,
                dw=window,
            )
            if self.shifts > 0:
                output = torch.roll(
                    output, shifts=(window // 2, window // 2), dims=(2, 3)
                )
            outputs.append(output)
        return self.project_out(torch.cat(outputs, dim=1)), attentions


class ELAB(nn.Module):
    def __init__(
        self,
        channels: int,
        expansion_ratio: int,
        shifts: int,
        window_sizes: tuple[int, int, int],
        shared_depth: int,
        act_type: str,
    ) -> None:
        super().__init__()
        self.shared_depth = shared_depth
        self.modules_lfe = nn.ModuleDict({
            f"lfe_{index}": LFE(channels, channels, expansion_ratio, act_type=act_type)
            for index in range(shared_depth + 1)
        })
        self.modules_gmsa = nn.ModuleDict({
            f"gmsa_{index}": GMSA(channels, shifts, window_sizes, calc_attn=index == 0)
            for index in range(shared_depth + 1)
        })

    def forward(self, x: Tensor) -> Tensor:
        attention: list[Tensor] | None = None
        for index in range(self.shared_depth + 1):
            x = self.modules_lfe[f"lfe_{index}"](x) + x
            y, attention = self.modules_gmsa[f"gmsa_{index}"](x, attention)
            x = y + x
        return x


@ARCH_REGISTRY.register()
class elan(nn.Module):
    """Official ELAN network. Defaults reproduce the full x4 configuration."""

    def __init__(
        self,
        colors: int = 3,
        m_elan: int = 36,
        c_elan: int = 180,
        n_share: int = 0,
        r_expand: int = 2,
        window_sizes: Sequence[int] = (4, 8, 16),
        act_type: str = "relu",
        rgb_range: float = 255.0,
        upscale: int = upscale,
        scale: int | None = None,
    ) -> None:
        super().__init__()
        if scale is not None:
            upscale = scale
        windows = tuple(window_sizes)
        if colors != 3:
            msg = "Official ELAN mean-shift layers require colors=3."
            raise ValueError(msg)
        if not isinstance(upscale, int) or upscale <= 0:
            msg = "upscale must be a positive integer."
            raise ValueError(msg)
        if m_elan <= 0 or c_elan <= 0 or r_expand <= 0 or n_share < 0:
            msg = "m_elan, c_elan and r_expand must be positive; n_share cannot be negative."
            raise ValueError(msg)
        if m_elan % (n_share + 1) != 0:
            msg = "m_elan must be divisible by n_share + 1."
            raise ValueError(msg)
        if c_elan % 3 != 0:
            msg = "c_elan must be divisible by three for the three attention branches."
            raise ValueError(msg)
        if len(windows) != 3 or any(
            not isinstance(size, int) or size <= 0 for size in windows
        ):
            msg = "window_sizes must contain exactly three positive integers."
            raise ValueError(msg)
        if not math.isfinite(rgb_range) or rgb_range <= 0:
            msg = "rgb_range must be a finite positive value."
            raise ValueError(msg)
        if act_type not in {"linear", "relu", "gelu"}:
            msg = f"Unsupported ELAN activation: {act_type}."
            raise ValueError(msg)

        self.scale = upscale
        self.window_sizes = windows
        self.pad_size = math.lcm(*windows)
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)
        self.head = nn.Sequential(nn.Conv2d(colors, c_elan, 3, padding=1))
        self.body = nn.Sequential(
            *(
                ELAB(
                    c_elan,
                    r_expand,
                    shifts=index % 2,
                    window_sizes=windows,
                    shared_depth=n_share,
                    act_type=act_type,
                )
                for index in range(m_elan // (n_share + 1))
            )
        )
        self.tail = nn.Sequential(
            nn.Conv2d(c_elan, colors * upscale**2, 3, padding=1),
            nn.PixelShuffle(upscale),
        )

    def _pad(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        pad_h = (-height) % self.pad_size
        pad_w = (-width) % self.pad_size
        if pad_h == 0 and pad_w == 0:
            return x
        mode = "reflect" if pad_h < height and pad_w < width else "replicate"
        return F.pad(x, (0, pad_w, 0, pad_h), mode=mode)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            msg = f"ELAN expects BCHW input with three channels, got {tuple(x.shape)}."
            raise ValueError(msg)
        height, width = x.shape[-2:]
        x = self._pad(x)
        x = self.head(self.sub_mean(x))
        x = self.tail(self.body(x) + x)
        x = self.add_mean(x)
        return x[:, :, : height * self.scale, : width * self.scale]
