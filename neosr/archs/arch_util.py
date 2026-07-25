from collections.abc import Callable, Iterable
from itertools import repeat

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def net_opt() -> tuple[int, bool]:
    """Return legacy architecture defaults without reading process-global CLI state.

    ``build_network`` injects the configured scale explicitly. The fallback remains for
    third-party code importing architecture constructors directly.
    """
    return 4, True


def mod_pad(x: Tensor, multiple: int | tuple[int, int]) -> Tensor:
    """Pad an NCHW tensor's bottom and right edges up to a multiple of ``multiple``.

    Window-based architectures can only partition feature maps whose height and
    width are exact multiples of their window size, so their forward pass has to
    pad first and crop the result back to the original extent afterwards. Padding
    on the bottom/right only keeps the surviving region aligned with the input,
    which is what makes that final crop valid.

    Reflect padding continues the edge gradient and is what the reference SwinIR
    implementations use, but it cannot pad by more than the source extent. Inputs
    smaller than one window therefore fall back to replicate, which has no such
    limit and still avoids the black border that constant padding would bleed
    into the output.

    Args:
    ----
        x (Tensor): Input tensor in NCHW layout.
        multiple (int | tuple[int, int]): Required alignment, either shared by
            both axes or given as ``(height, width)``.

    """
    mult_h, mult_w = (multiple, multiple) if isinstance(multiple, int) else multiple
    if mult_h < 1 or mult_w < 1:
        msg = f"mod_pad multiple must be positive, got {multiple}."
        raise ValueError(msg)
    height, width = x.shape[-2:]
    pad_h = -height % mult_h
    pad_w = -width % mult_w
    if pad_h == 0 and pad_w == 0:
        return x
    mode = "reflect" if pad_h < height and pad_w < width else "replicate"
    return F.pad(x, (0, pad_w, 0, pad_h), mode)


class DySample(nn.Module):
    """Adapted from 'Learning to Upsample by Learning to Sample':
    https://arxiv.org/abs/2308.15085
    https://github.com/tiny-smart/dysample
    """

    def _init_pos(self) -> Tensor:
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def __init__(
        self,
        in_channels: int,
        out_ch: int,
        scale: int = 2,
        groups: int = 4,
        end_convolution: bool = True,
    ) -> None:
        super().__init__()

        # Ordered so the modulo below can never divide by zero.
        if groups < 1:
            msg = f"DySample groups must be a positive integer, got {groups}."
            raise ValueError(msg)
        if in_channels < groups or in_channels % groups != 0:
            msg = (
                f"DySample in_channels ({in_channels}) must be a positive multiple "
                f"of groups ({groups})."
            )
            raise ValueError(msg)

        out_channels = 2 * groups * scale**2
        self.scale = scale
        self.groups = groups
        self.end_convolution = end_convolution
        if end_convolution:
            self.end_conv = nn.Conv2d(in_channels, out_ch, kernel_size=1)

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if self.training:
            nn.init.trunc_normal_(self.offset.weight, std=0.02)
            nn.init.constant_(self.scope.weight, val=0)

        self.register_buffer("init_pos", self._init_pos())

    def forward(self, x: Tensor) -> Tensor:
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5

        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device, non_blocking=True)
        )
        normalizer = torch.tensor(
            [W, H], dtype=x.dtype, device=x.device, pin_memory=True
        ).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = (
            F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        output = F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(B, -1, self.scale * H, self.scale * W)

        if self.end_convolution:
            output = self.end_conv(output)

        return output


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
) -> Tensor:
    """Drop paths (Stochastic Depth) per sample.
    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)

    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.
    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


# From PyTorch
def _ntuple(n: int) -> Callable:
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
