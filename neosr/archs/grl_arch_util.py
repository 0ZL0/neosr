"""Internal operators and building blocks for the GRL architecture."""

from __future__ import annotations

import math
from abc import ABC
from math import prod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neosr.archs.arch_util import DropPath, to_2tuple


def bchw_to_bhwc(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, C, H, W) to (B, H, W, C)."""
    return x.permute(0, 2, 3, 1)


def bhwc_to_bchw(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, H, W, C) to (B, C, H, W)."""
    return x.permute(0, 3, 1, 2)


def bchw_to_blc(x: torch.Tensor) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, C, H, W) to (B, L, C)."""
    return x.flatten(2).transpose(1, 2)


def blc_to_bchw(x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, L, C) to (B, C, H, W)."""
    B, _L, C = x.shape
    return x.transpose(1, 2).view(B, C, *x_size)


def blc_to_bhwc(x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, L, C) to (B, H, W, C)."""
    B, _L, C = x.shape
    return x.view(B, *x_size, C)


def window_partition(x, window_size: tuple[int, int]):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(
        B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C
    )
    return (
        x
        .permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size[0], window_size[1], C)
    )


def window_reverse(windows, window_size: tuple[int, int], img_size: tuple[int, int]):
    """
    Args:
        windows: (num_windows * B, window_size[0], window_size[1], C)
        window_size (Tuple[int, int]): Window size
        img_size (Tuple[int, int]): Image size

    Returns:
        x: (B, H, W, C)
    """
    H, W = img_size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(
        B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


def _fill_window(input_resolution, window_size, shift_size=None):
    if shift_size is None:
        shift_size = [s // 2 for s in window_size]

    img_mask = torch.zeros((1, *input_resolution, 1))  # 1 H W 1
    h_slices = (
        slice(0, -window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    )
    w_slices = (
        slice(0, -window_size[1]),
        slice(-window_size[1], -shift_size[1]),
        slice(-shift_size[1], None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    # nW, window_size, window_size, 1
    return mask_windows.view(-1, prod(window_size))


def calculate_mask(input_resolution, window_size, shift_size):
    """
    Use case: 1)
    """
    # calculate attention mask for SW-MSA
    if isinstance(shift_size, int):
        shift_size = to_2tuple(shift_size)
    mask_windows = _fill_window(input_resolution, window_size, shift_size)

    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, (-100.0)).masked_fill(
        attn_mask == 0, 0.0
    )  # nW, window_size**2, window_size**2


def calculate_mask_all(
    input_resolution,
    window_size,
    shift_size,
    anchor_window_down_factor=1,
    window_to_anchor=True,
):
    """
    Use case: 3)
    """
    # calculate attention mask for SW-MSA
    anchor_resolution = [s // anchor_window_down_factor for s in input_resolution]
    aws = [s // anchor_window_down_factor for s in window_size]
    anchor_shift = [s // anchor_window_down_factor for s in shift_size]

    # mask of window1: nW, Wh**Ww
    mask_windows = _fill_window(input_resolution, window_size, shift_size)
    # mask of window2: nW, AWh*AWw
    mask_anchor = _fill_window(anchor_resolution, aws, anchor_shift)

    if window_to_anchor:
        attn_mask = mask_windows.unsqueeze(2) - mask_anchor.unsqueeze(1)
    else:
        attn_mask = mask_anchor.unsqueeze(2) - mask_windows.unsqueeze(1)
    return attn_mask.masked_fill(attn_mask != 0, (-100.0)).masked_fill(
        attn_mask == 0, 0.0
    )  # nW, Wh**Ww, AWh*AWw


def calculate_win_mask(
    input_resolution1, input_resolution2, window_size1, window_size2
):
    """
    Use case: 2)
    """
    # calculate attention mask for SW-MSA

    # mask of window1: nW, Wh**Ww
    mask_windows1 = _fill_window(input_resolution1, window_size1)
    # mask of window2: nW, AWh*AWw
    mask_windows2 = _fill_window(input_resolution2, window_size2)

    attn_mask = mask_windows1.unsqueeze(2) - mask_windows2.unsqueeze(1)
    return attn_mask.masked_fill(attn_mask != 0, (-100.0)).masked_fill(
        attn_mask == 0, 0.0
    )  # nW, Wh**Ww, AWh*AWw


def _get_meshgrid_coords(start_coords, end_coords):
    coord_h = torch.arange(start_coords[0], end_coords[0])
    coord_w = torch.arange(start_coords[1], end_coords[1])
    coords = torch.stack(torch.meshgrid([coord_h, coord_w], indexing="ij"))  # 2, Wh, Ww
    return torch.flatten(coords, 1)  # 2, Wh*Ww


def get_relative_coords_table(
    window_size, pretrained_window_size=None, anchor_window_down_factor=1
):
    """
    Use case: 1)
    """
    # get relative_coords_table
    if pretrained_window_size is None:
        pretrained_window_size = [0, 0]
    ws = window_size
    aws = [w // anchor_window_down_factor for w in window_size]
    pws = pretrained_window_size
    paws = [w // anchor_window_down_factor for w in pretrained_window_size]

    ts = [(w1 + w2) // 2 for w1, w2 in zip(ws, aws, strict=False)]
    pts = [(w1 + w2) // 2 for w1, w2 in zip(pws, paws, strict=False)]

    # TODO: pretrained window size and pretrained anchor window size is only used here.
    # TODO: Investigate whether it is really important to use this setting when finetuning large window size
    # TODO: based on pretrained weights with small window size.

    coord_h = torch.arange(-(ts[0] - 1), ts[0], dtype=torch.float32)
    coord_w = torch.arange(-(ts[1] - 1), ts[1], dtype=torch.float32)
    table = torch.stack(torch.meshgrid([coord_h, coord_w], indexing="ij")).permute(
        1, 2, 0
    )
    table = table.contiguous().unsqueeze(0)  # 1, Wh+AWh-1, Ww+AWw-1, 2
    if pts[0] > 0:
        table[:, :, :, 0] /= pts[0] - 1
        table[:, :, :, 1] /= pts[1] - 1
    else:
        table[:, :, :, 0] /= ts[0] - 1
        table[:, :, :, 1] /= ts[1] - 1
    table *= 8  # normalize to -8, 8
    return torch.sign(table) * torch.log2(torch.abs(table) + 1.0) / np.log2(8)


def get_relative_coords_table_all(
    window_size, pretrained_window_size=None, anchor_window_down_factor=1
):
    """
    Use case: 3)

    Support all window shapes.
    Args:
        window_size:
        pretrained_window_size:
        anchor_window_down_factor:

    Returns:

    """
    # get relative_coords_table
    if pretrained_window_size is None:
        pretrained_window_size = [0, 0]
    ws = window_size
    aws = [w // anchor_window_down_factor for w in window_size]
    pws = pretrained_window_size
    paws = [w // anchor_window_down_factor for w in pretrained_window_size]

    # positive table size: (Ww - 1) - (Ww - AWw) // 2
    ts_p = [w1 - 1 - (w1 - w2) // 2 for w1, w2 in zip(ws, aws, strict=False)]
    # negative table size: -(AWw - 1) - (Ww - AWw) // 2
    ts_n = [-(w2 - 1) - (w1 - w2) // 2 for w1, w2 in zip(ws, aws, strict=False)]
    pts = [w1 - 1 - (w1 - w2) // 2 for w1, w2 in zip(pws, paws, strict=False)]

    # TODO: pretrained window size and pretrained anchor window size is only used here.
    # TODO: Investigate whether it is really important to use this setting when finetuning large window size
    # TODO: based on pretrained weights with small window size.

    coord_h = torch.arange(ts_n[0], ts_p[0] + 1, dtype=torch.float32)
    coord_w = torch.arange(ts_n[1], ts_p[1] + 1, dtype=torch.float32)
    table = torch.stack(torch.meshgrid([coord_h, coord_w], indexing="ij")).permute(
        1, 2, 0
    )
    table = table.contiguous().unsqueeze(0)  # 1, Wh+AWh-1, Ww+AWw-1, 2
    if pts[0] > 0:
        table[:, :, :, 0] /= pts[0]
        table[:, :, :, 1] /= pts[1]
    else:
        table[:, :, :, 0] /= ts_p[0]
        table[:, :, :, 1] /= ts_p[1]
    table *= 8  # normalize to -8, 8
    return torch.sign(table) * torch.log2(torch.abs(table) + 1.0) / np.log2(8)
    # 1, Wh+AWh-1, Ww+AWw-1, 2


def coords_diff(coords1, coords2, max_diff):
    # The coordinates starts from (-start_coord[0], -start_coord[1])
    coords = coords1[:, :, None] - coords2[:, None, :]  # 2, Wh*Ww, AWh*AWw
    coords = coords.permute(1, 2, 0).contiguous()  # Wh*Ww, AWh*AWw, 2
    coords[:, :, 0] += max_diff[0] - 1  # shift to start from 0
    coords[:, :, 1] += max_diff[1] - 1
    coords[:, :, 0] *= 2 * max_diff[1] - 1
    return coords.sum(-1)  # Wh*Ww, AWh*AWw


def get_relative_position_index(
    window_size, anchor_window_down_factor=1, window_to_anchor=True
):
    """
    Use case: 1)
    """
    # get pair-wise relative position index for each token inside the window
    ws = window_size
    aws = [w // anchor_window_down_factor for w in window_size]
    coords_anchor_end = [(w1 + w2) // 2 for w1, w2 in zip(ws, aws, strict=False)]
    coords_anchor_start = [(w1 - w2) // 2 for w1, w2 in zip(ws, aws, strict=False)]

    coords = _get_meshgrid_coords((0, 0), window_size)  # 2, Wh*Ww
    coords_anchor = _get_meshgrid_coords(coords_anchor_start, coords_anchor_end)
    # 2, AWh*AWw

    if window_to_anchor:
        idx = coords_diff(coords, coords_anchor, max_diff=coords_anchor_end)
    else:
        idx = coords_diff(coords_anchor, coords, max_diff=coords_anchor_end)
    return idx  # Wh*Ww, AWh*AWw or AWh*AWw, Wh*Ww


def coords_diff_odd(coords1, coords2, start_coord, max_diff):
    # The coordinates starts from (-start_coord[0], -start_coord[1])
    coords = coords1[:, :, None] - coords2[:, None, :]  # 2, Wh*Ww, AWh*AWw
    coords = coords.permute(1, 2, 0).contiguous()  # Wh*Ww, AWh*AWw, 2
    coords[:, :, 0] += start_coord[0]  # shift to start from 0
    coords[:, :, 1] += start_coord[1]
    coords[:, :, 0] *= max_diff
    return coords.sum(-1)  # Wh*Ww, AWh*AWw


def get_relative_position_index_all(
    window_size, anchor_window_down_factor=1, window_to_anchor=True
):
    """
    Use case: 3)
    Support all window shapes:
        square window - square window
        rectangular window - rectangular window
        window - anchor
        anchor - window
        [8, 8] - [8, 8]
        [4, 86] - [2, 43]
    """
    # get pair-wise relative position index for each token inside the window
    ws = window_size
    aws = [w // anchor_window_down_factor for w in window_size]
    coords_anchor_start = [(w1 - w2) // 2 for w1, w2 in zip(ws, aws, strict=False)]
    coords_anchor_end = [
        s + w2 for s, w2 in zip(coords_anchor_start, aws, strict=False)
    ]

    coords = _get_meshgrid_coords((0, 0), window_size)  # 2, Wh*Ww
    coords_anchor = _get_meshgrid_coords(coords_anchor_start, coords_anchor_end)
    # 2, AWh*AWw

    max_horizontal_diff = aws[1] + ws[1] - 1
    if window_to_anchor:
        offset = [w2 + s - 1 for s, w2 in zip(coords_anchor_start, aws, strict=False)]
        idx = coords_diff_odd(coords, coords_anchor, offset, max_horizontal_diff)
    else:
        offset = [w1 - s - 1 for s, w1 in zip(coords_anchor_start, ws, strict=False)]
        idx = coords_diff_odd(coords_anchor, coords, offset, max_horizontal_diff)
    return idx  # Wh*Ww, AWh*AWw or AWh*AWw, Wh*Ww


def get_relative_position_index_simple(
    window_size, anchor_window_down_factor=1, window_to_anchor=True
):
    """
    Use case: 3)
    This is a simplified version of get_relative_position_index_all
    The start coordinate of anchor window is also (0, 0)
    get pair-wise relative position index for each token inside the window
    """
    ws = window_size
    aws = [w // anchor_window_down_factor for w in window_size]

    coords = _get_meshgrid_coords((0, 0), window_size)  # 2, Wh*Ww
    coords_anchor = _get_meshgrid_coords((0, 0), aws)
    # 2, AWh*AWw

    max_horizontal_diff = aws[1] + ws[1] - 1
    if window_to_anchor:
        offset = [w2 - 1 for w2 in aws]
        idx = coords_diff_odd(coords, coords_anchor, offset, max_horizontal_diff)
    else:
        offset = [w1 - 1 for w1 in ws]
        idx = coords_diff_odd(coords_anchor, coords, offset, max_horizontal_diff)
    return idx  # Wh*Ww, AWh*AWw or AWh*AWw, Wh*Ww


def get_relative_win_position_index(window_size, anchor_window_size):
    """
    Use case: 2)
    """
    # get pair-wise relative position index for each token inside the window
    ws = window_size
    aws = anchor_window_size
    coords_anchor_end = [(w1 + w2) // 2 for w1, w2 in zip(ws, aws, strict=False)]
    coords_anchor_start = [(w1 - w2) // 2 for w1, w2 in zip(ws, aws, strict=False)]

    coords = _get_meshgrid_coords((0, 0), window_size)  # 2, Wh*Ww
    coords_anchor = _get_meshgrid_coords(coords_anchor_start, coords_anchor_end)
    # 2, AWh*AWw
    coords = coords[:, :, None] - coords_anchor[:, None, :]  # 2, Wh*Ww, AWh*AWw
    coords = coords.permute(1, 2, 0).contiguous()  # Wh*Ww, AWh*AWw, 2
    coords[:, :, 0] += coords_anchor_end[0] - 1  # shift to start from 0
    coords[:, :, 1] += coords_anchor_end[1] - 1
    coords[:, :, 0] *= 2 * coords_anchor_end[1] - 1
    return coords.sum(-1)  # Wh*Ww, AWh*AWw


def get_relative_win_coords_table(
    window_size,
    anchor_window_size,
    pretrained_window_size=None,
    pretrained_anchor_window_size=None,
):
    """
    Use case: 2)
    """
    # get relative_coords_table
    if pretrained_anchor_window_size is None:
        pretrained_anchor_window_size = [0, 0]
    if pretrained_window_size is None:
        pretrained_window_size = [0, 0]
    ws = window_size
    aws = anchor_window_size
    pws = pretrained_window_size
    paws = pretrained_anchor_window_size

    # TODO: pretrained window size and pretrained anchor window size is only used here.
    # TODO: Investigate whether it is really important to use this setting when finetuning large window size
    # TODO: based on pretrained weights with small window size.

    table_size = [(wsi + awsi) // 2 for wsi, awsi in zip(ws, aws, strict=False)]
    table_size_pretrained = [
        (pwsi + pawsi) // 2 for pwsi, pawsi in zip(pws, paws, strict=False)
    ]
    coord_h = torch.arange(-(table_size[0] - 1), table_size[0], dtype=torch.float32)
    coord_w = torch.arange(-(table_size[1] - 1), table_size[1], dtype=torch.float32)
    table = torch.stack(torch.meshgrid([coord_h, coord_w], indexing="ij")).permute(
        1, 2, 0
    )
    table = table.contiguous().unsqueeze(0)  # 1, Wh+AWh-1, Ww+AWw-1, 2
    if table_size_pretrained[0] > 0:
        table[:, :, :, 0] /= table_size_pretrained[0] - 1
        table[:, :, :, 1] /= table_size_pretrained[1] - 1
    else:
        table[:, :, :, 0] /= table_size[0] - 1
        table[:, :, :, 1] /= table_size[1] - 1
    table *= 8  # normalize to -8, 8
    return torch.sign(table) * torch.log2(torch.abs(table) + 1.0) / np.log2(8)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

    def forward(self, x):
        _B, _C, H, W = x.shape
        x = bchw_to_blc(x)
        x = super().forward(x)
        return blc_to_bchw(x, (H, W))


def build_last_conv(conv_type, dim):
    if conv_type == "1conv":
        block = nn.Conv2d(dim, dim, 3, 1, 1)
    elif conv_type == "3conv":
        # to save parameters and memory
        block = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim // 4, dim, 3, 1, 1),
        )
    elif conv_type == "1conv1x1":
        block = nn.Conv2d(dim, dim, 1, 1, 0)
    elif conv_type == "linear":
        block = Linear(dim, dim)
    return block


class CPB_MLP(nn.Sequential):
    def __init__(self, in_channels, out_channels, channels=512):
        m = [
            nn.Linear(in_channels, channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels, out_channels, bias=False),
        ]
        super().__init__(*m)


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, args):
        m = [
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                groups=in_channels,
                bias=bias,
            )
        ]
        if args.separable_conv_act:
            m.append(nn.GELU())
        m.append(nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias))
        super().__init__(*m)


class QKVProjection(nn.Module):
    def __init__(self, dim, qkv_bias, proj_type, args):
        super().__init__()
        self.proj_type = proj_type
        if proj_type == "linear":
            self.body = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.body = SeparableConv(dim, dim * 3, 3, 1, qkv_bias, args)

    def forward(self, x, x_size):
        if self.proj_type == "separable_conv":
            x = blc_to_bchw(x, x_size)
        x = self.body(x)
        if self.proj_type == "separable_conv":
            x = bchw_to_blc(x)
        return x


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.reduction = nn.Linear(4 * in_dim, out_dim, bias=False)

    def forward(self, x, x_size):
        """
        x: B, H*W, C
        """
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0, f"x height ({H}) is not even."
        assert W % 2 == 0, f"x width ({W}) is not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        return self.reduction(x)


class AnchorLinear(nn.Module):
    r"""Linear anchor projection layer
    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, in_channels, out_channels, down_factor, pooling_mode, bias):
        super().__init__()
        self.down_factor = down_factor
        if pooling_mode == "maxpool":
            self.pooling = nn.MaxPool2d(down_factor, down_factor)
        elif pooling_mode == "avgpool":
            self.pooling = nn.AvgPool2d(down_factor, down_factor)
        self.reduction = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x, x_size):
        """
        x: B, H*W, C
        """
        x = blc_to_bchw(x, x_size)
        x = bchw_to_blc(self.pooling(x))
        return blc_to_bhwc(self.reduction(x), [s // self.down_factor for s in x_size])


class AnchorProjection(nn.Module):
    def __init__(self, dim, proj_type, one_stage, anchor_window_down_factor, args):
        super().__init__()
        self.proj_type = proj_type
        self.body = nn.ModuleList([])
        if one_stage:
            if proj_type == "patchmerging":
                m = PatchMerging(dim, dim // 2)
            elif proj_type == "conv2d":
                kernel_size = anchor_window_down_factor + 1
                stride = anchor_window_down_factor
                padding = kernel_size // 2
                m = nn.Conv2d(dim, dim // 2, kernel_size, stride, padding)
            elif proj_type == "separable_conv":
                kernel_size = anchor_window_down_factor + 1
                stride = anchor_window_down_factor
                m = SeparableConv(
                    dim, dim // 2, kernel_size, stride, bias=True, args=args
                )
            elif proj_type.find("pool") >= 0:
                m = AnchorLinear(
                    dim, dim // 2, anchor_window_down_factor, proj_type, bias=True
                )
            self.body.append(m)
        else:
            for i in range(int(math.log2(anchor_window_down_factor))):
                cin = dim if i == 0 else dim // 2
                if proj_type == "patchmerging":
                    m = PatchMerging(cin, dim // 2)
                elif proj_type == "conv2d":
                    m = nn.Conv2d(cin, dim // 2, 3, 2, 1)
                elif proj_type == "separable_conv":
                    m = SeparableConv(cin, dim // 2, 3, 2, bias=True, args=args)
                self.body.append(m)

    def forward(self, x, x_size):
        if self.proj_type.find("conv") >= 0:
            x = blc_to_bchw(x, x_size)
            for m in self.body:
                x = m(x)
            x = bchw_to_bhwc(x)
        elif self.proj_type.find("pool") >= 0:
            for m in self.body:
                x = m(x, x_size)
        else:
            for i, m in enumerate(self.body):
                x = m(x, [s // 2**i for s in x_size])
            x = blc_to_bhwc(x, [s // 2 ** (i + 1) for s in x_size])
        return x


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        reduction (int): Channel reduction factor. Default: 16.
    """

    def __init__(self, num_feat, reduction=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // reduction, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // reduction, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=4, reduction=18):
        super().__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, reduction),
        )

    def forward(self, x, x_size):
        x = self.cab(blc_to_bchw(x, x_size).contiguous())
        return bchw_to_blc(x)


class AffineTransform(nn.Module):
    r"""Affine transformation of the attention map.
    The window could be a square window or a stripe window. Supports attention between different window sizes
    """

    def __init__(self, num_heads):
        super().__init__()
        logit_scale = torch.log(10 * torch.ones((num_heads, 1, 1)))
        self.logit_scale = nn.Parameter(logit_scale, requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = CPB_MLP(2, num_heads)

    def forward(self, attn, relative_coords_table, relative_position_index, mask):
        B_, H, N1, N2 = attn.shape
        # logit scale
        attn = attn * torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()

        bias_table = self.cpb_mlp(relative_coords_table)  # 2*Wh-1, 2*Ww-1, num_heads
        bias_table = bias_table.view(-1, H)

        bias = bias_table[relative_position_index.view(-1)]
        bias = bias.view(N1, N2, -1).permute(2, 0, 1).contiguous()
        # nH, Wh*Ww, Wh*Ww
        bias = 16 * torch.sigmoid(bias)
        attn = attn + bias.unsqueeze(0)

        # W-MSA/SW-MSA
        # shift attention mask
        if mask is not None:
            nW = mask.shape[0]
            mask = mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(B_ // nW, nW, H, N1, N2) + mask
            attn = attn.view(-1, H, N1, N2)

        return attn


def _get_stripe_info(stripe_size_in, stripe_groups_in, stripe_shift, input_resolution):
    stripe_size, shift_size = [], []
    for s, g, d in zip(
        stripe_size_in, stripe_groups_in, input_resolution, strict=False
    ):
        if g is None:
            stripe_size.append(s)
            shift_size.append(s // 2 if stripe_shift else 0)
        else:
            stripe_size.append(d // g)
            shift_size.append(0 if g == 1 else d // (g * 2))
    return stripe_size, shift_size


class Attention(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    def attn(self, q, k, v, attn_transform, table, index, mask, reshape=True):
        # q, k, v: # nW*B, H, wh*ww, dim
        # cosine attention map
        B_, _, H, head_dim = q.shape
        if self.euclidean_dist:
            # print("use euclidean distance")
            attn = torch.norm(q.unsqueeze(-2) - k.unsqueeze(-3), dim=-1)
        else:
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        attn = attn_transform(attn, table, index, mask)
        # attention
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v  # B_, H, N1, head_dim
        if reshape:
            x = x.transpose(1, 2).reshape(B_, -1, H * head_dim)
        # B_, N, C
        return x


class WindowAttention(Attention):
    r"""Window attention. QKV is the input to the forward method.
    Args:
        num_heads (int): Number of attention heads.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
        self,
        input_resolution,
        window_size,
        num_heads,
        window_shift=False,
        attn_drop=0.0,
        pretrained_window_size=None,
        args=None,
    ):

        if pretrained_window_size is None:
            pretrained_window_size = [0, 0]
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.shift_size = window_size[0] // 2 if window_shift else 0
        self.euclidean_dist = args.euclidean_dist

        self.attn_transform = AffineTransform(num_heads)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, x_size, table, index, mask):
        """
        Args:
            qkv: input QKV features with shape of (B, L, 3C)
            x_size: use x_size to determine whether the relative positional bias table and index
            need to be regenerated.
        """
        H, W = x_size
        B, L, C = qkv.shape
        qkv = qkv.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            qkv = torch.roll(
                qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )

        # partition windows
        qkv = window_partition(qkv, self.window_size)  # nW*B, wh, ww, C
        qkv = qkv.view(-1, prod(self.window_size), C)  # nW*B, wh*ww, C

        B_, N, _ = qkv.shape
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # nW*B, H, wh*ww, dim

        # attention
        x = self.attn(q, k, v, self.attn_transform, table, index, mask)

        # merge windows
        x = x.view(-1, *self.window_size, C // 3)
        x = window_reverse(x, self.window_size, x_size)  # B, H, W, C/3

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        return x.view(B, L, C // 3)

    def extra_repr(self) -> str:
        return (
            f"window_size={self.window_size}, shift_size={self.shift_size}, "
            f"pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}"
        )

    def flops(self, N):
        pass


class AnchorStripeAttention(Attention):
    r"""Stripe attention
    Args:
        stripe_size (tuple[int]): The height and width of the stripe.
        num_heads (int): Number of attention heads.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        pretrained_stripe_size (tuple[int]): The height and width of the stripe in pre-training.
    """

    def __init__(
        self,
        input_resolution,
        stripe_size,
        stripe_groups,
        stripe_shift,
        num_heads,
        attn_drop=0.0,
        pretrained_stripe_size=None,
        anchor_window_down_factor=1,
        args=None,
    ):

        if pretrained_stripe_size is None:
            pretrained_stripe_size = [0, 0]
        super().__init__()
        self.input_resolution = input_resolution
        self.stripe_size = stripe_size  # Wh, Ww
        self.stripe_groups = stripe_groups
        self.stripe_shift = stripe_shift
        self.num_heads = num_heads
        self.pretrained_stripe_size = pretrained_stripe_size
        self.anchor_window_down_factor = anchor_window_down_factor
        self.euclidean_dist = args.euclidean_dist

        self.attn_transform1 = AffineTransform(num_heads)
        self.attn_transform2 = AffineTransform(num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, qkv, anchor, x_size, table, index_a2w, index_w2a, mask_a2w, mask_w2a
    ):
        """
        Args:
            qkv: input features with shape of (B, L, C)
            anchor:
            x_size: use stripe_size to determine whether the relative positional bias table and index
            need to be regenerated.
        """
        H, W = x_size
        B, _L, C = qkv.shape
        qkv = qkv.view(B, H, W, C)

        stripe_size, shift_size = _get_stripe_info(
            self.stripe_size, self.stripe_groups, self.stripe_shift, x_size
        )
        anchor_stripe_size = [s // self.anchor_window_down_factor for s in stripe_size]
        anchor_shift_size = [s // self.anchor_window_down_factor for s in shift_size]
        # cyclic shift
        if self.stripe_shift:
            qkv = torch.roll(qkv, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            anchor = torch.roll(
                anchor,
                shifts=(-anchor_shift_size[0], -anchor_shift_size[1]),
                dims=(1, 2),
            )

        # partition windows
        qkv = window_partition(qkv, stripe_size)  # nW*B, wh, ww, C
        qkv = qkv.view(-1, prod(stripe_size), C)  # nW*B, wh*ww, C
        anchor = window_partition(anchor, anchor_stripe_size)
        anchor = anchor.view(-1, prod(anchor_stripe_size), C // 3)

        B_, N1, _ = qkv.shape
        N2 = anchor.shape[1]
        qkv = qkv.reshape(B_, N1, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        anchor = anchor.reshape(B_, N2, self.num_heads, -1).permute(0, 2, 1, 3)

        # attention
        x = self.attn(
            anchor,
            k,
            v,
            self.attn_transform1,
            table,
            index_a2w,
            mask_a2w,
            reshape=False,
        )
        x = self.attn(q, anchor, x, self.attn_transform2, table, index_w2a, mask_w2a)

        # merge windows
        x = x.view(B_, *stripe_size, C // 3)
        x = window_reverse(x, stripe_size, x_size)  # B H' W' C

        # reverse the shift
        if self.stripe_shift:
            x = torch.roll(x, shifts=shift_size, dims=(1, 2))

        return x.view(B, H * W, C // 3)

    def extra_repr(self) -> str:
        return (
            f"stripe_size={self.stripe_size}, stripe_groups={self.stripe_groups}, stripe_shift={self.stripe_shift}, "
            f"pretrained_stripe_size={self.pretrained_stripe_size}, num_heads={self.num_heads}, anchor_window_down_factor={self.anchor_window_down_factor}"
        )

    def flops(self, N):
        pass


class MixedAttention(nn.Module):
    r"""Mixed window attention and stripe attention
    Args:
        dim (int): Number of input channels.
        stripe_size (tuple[int]): The height and width of the stripe.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_stripe_size (tuple[int]): The height and width of the stripe in pre-training.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads_w,
        num_heads_s,
        window_size,
        window_shift,
        stripe_size,
        stripe_groups,
        stripe_shift,
        qkv_bias=True,
        qkv_proj_type="linear",
        anchor_proj_type="separable_conv",
        anchor_one_stage=True,
        anchor_window_down_factor=1,
        attn_drop=0.0,
        proj_drop=0.0,
        pretrained_window_size=None,
        pretrained_stripe_size=None,
        args=None,
    ):

        if pretrained_stripe_size is None:
            pretrained_stripe_size = [0, 0]
        if pretrained_window_size is None:
            pretrained_window_size = [0, 0]
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.args = args
        # print(args)
        self.qkv = QKVProjection(dim, qkv_bias, qkv_proj_type, args)
        # anchor is only used for stripe attention
        self.anchor = AnchorProjection(
            dim, anchor_proj_type, anchor_one_stage, anchor_window_down_factor, args
        )

        self.window_attn = WindowAttention(
            input_resolution,
            window_size,
            num_heads_w,
            window_shift,
            attn_drop,
            pretrained_window_size,
            args,
        )
        self.stripe_attn = AnchorStripeAttention(
            input_resolution,
            stripe_size,
            stripe_groups,
            stripe_shift,
            num_heads_s,
            attn_drop,
            pretrained_stripe_size,
            anchor_window_down_factor,
            args,
        )
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x_size, table_index_mask):
        """
        Args:
            x: input features with shape of (B, L, C)
            stripe_size: use stripe_size to determine whether the relative positional bias table and index
            need to be regenerated.
        """
        _B, _L, C = x.shape

        # qkv projection
        qkv = self.qkv(x, x_size)
        qkv_window, qkv_stripe = torch.split(qkv, C * 3 // 2, dim=-1)
        # anchor projection
        anchor = self.anchor(x, x_size)

        # attention
        x_window = self.window_attn(
            qkv_window,
            x_size,
            *self._get_table_index_mask(table_index_mask, window_attn=True),
        )
        x_stripe = self.stripe_attn(
            qkv_stripe,
            anchor,
            x_size,
            *self._get_table_index_mask(table_index_mask, window_attn=False),
        )
        x = torch.cat([x_window, x_stripe], dim=-1)

        # output projection
        x = self.proj(x)
        return self.proj_drop(x)

    def _get_table_index_mask(self, table_index_mask, window_attn=True):
        if window_attn:
            return (
                table_index_mask["table_w"],
                table_index_mask["index_w"],
                table_index_mask["mask_w"],
            )
        return (
            table_index_mask["table_s"],
            table_index_mask["index_a2w"],
            table_index_mask["index_w2a"],
            table_index_mask["mask_a2w"],
            table_index_mask["mask_w2a"],
        )

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}"

    def flops(self, N):
        pass


class EfficientMixAttnTransformerBlock(nn.Module):
    r"""Mix attention transformer block with shared QKV projection and output projection for mixed attention modules.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_stripe_size (int): Window size in pre-training.
        attn_type (str, optional): Attention type. Default: cwhv.
                    c: residual blocks
                    w: window attention
                    h: horizontal stripe attention
                    v: vertical stripe attention
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads_w,
        num_heads_s,
        window_size=7,
        window_shift=False,
        stripe_size=None,
        stripe_groups=None,
        stripe_shift=False,
        stripe_type="H",
        mlp_ratio=4.0,
        qkv_bias=True,
        qkv_proj_type="linear",
        anchor_proj_type="separable_conv",
        anchor_one_stage=True,
        anchor_window_down_factor=1,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=None,
        pretrained_stripe_size=None,
        res_scale=1.0,
        args=None,
    ):
        if pretrained_stripe_size is None:
            pretrained_stripe_size = [0, 0]
        if pretrained_window_size is None:
            pretrained_window_size = [0, 0]
        if stripe_groups is None:
            stripe_groups = [None, None]
        if stripe_size is None:
            stripe_size = [8, 8]
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads_w = num_heads_w
        self.num_heads_s = num_heads_s
        self.window_size = window_size
        self.window_shift = window_shift
        self.stripe_shift = stripe_shift
        self.stripe_type = stripe_type
        self.args = args
        if self.stripe_type == "W":
            self.stripe_size = stripe_size[::-1]
            self.stripe_groups = stripe_groups[::-1]
        else:
            self.stripe_size = stripe_size
            self.stripe_groups = stripe_groups
        self.mlp_ratio = mlp_ratio
        self.res_scale = res_scale

        self.attn = MixedAttention(
            dim,
            input_resolution,
            num_heads_w,
            num_heads_s,
            window_size,
            window_shift,
            self.stripe_size,
            self.stripe_groups,
            stripe_shift,
            qkv_bias,
            qkv_proj_type,
            anchor_proj_type,
            anchor_one_stage,
            anchor_window_down_factor,
            attn_drop,
            drop,
            pretrained_window_size,
            pretrained_stripe_size,
            args,
        )
        self.norm1 = norm_layer(dim)
        if self.args.local_connection:
            self.conv = CAB(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.norm2 = norm_layer(dim)

    def _get_table_index_mask(self, all_table_index_mask):
        table_index_mask = {
            "table_w": all_table_index_mask["table_w"],
            "index_w": all_table_index_mask["index_w"],
        }
        if self.stripe_type == "W":
            table_index_mask["table_s"] = all_table_index_mask["table_sv"]
            table_index_mask["index_a2w"] = all_table_index_mask["index_sv_a2w"]
            table_index_mask["index_w2a"] = all_table_index_mask["index_sv_w2a"]
        else:
            table_index_mask["table_s"] = all_table_index_mask["table_sh"]
            table_index_mask["index_a2w"] = all_table_index_mask["index_sh_a2w"]
            table_index_mask["index_w2a"] = all_table_index_mask["index_sh_w2a"]
        if self.window_shift:
            table_index_mask["mask_w"] = all_table_index_mask["mask_w"]
        else:
            table_index_mask["mask_w"] = None
        if self.stripe_shift:
            if self.stripe_type == "W":
                table_index_mask["mask_a2w"] = all_table_index_mask["mask_sv_a2w"]
                table_index_mask["mask_w2a"] = all_table_index_mask["mask_sv_w2a"]
            else:
                table_index_mask["mask_a2w"] = all_table_index_mask["mask_sh_a2w"]
                table_index_mask["mask_w2a"] = all_table_index_mask["mask_sh_w2a"]
        else:
            table_index_mask["mask_a2w"] = None
            table_index_mask["mask_w2a"] = None
        return table_index_mask

    def forward(self, x, x_size, all_table_index_mask):
        # Mixed attention
        table_index_mask = self._get_table_index_mask(all_table_index_mask)
        if self.args.local_connection:
            x = (
                x
                + self.res_scale
                * self.drop_path(self.norm1(self.attn(x, x_size, table_index_mask)))
                + self.conv(x, x_size)
            )
        else:
            x = x + self.res_scale * self.drop_path(
                self.norm1(self.attn(x, x_size, table_index_mask))
            )
        # FFN
        return x + self.res_scale * self.drop_path(self.norm2(self.mlp(x)))

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads=({self.num_heads_w}, {self.num_heads_s}), "
            f"window_size={self.window_size}, window_shift={self.window_shift}, "
            f"stripe_size={self.stripe_size}, stripe_groups={self.stripe_groups}, stripe_shift={self.stripe_shift}, self.stripe_type={self.stripe_type}, "
            f"mlp_ratio={self.mlp_ratio}, res_scale={self.res_scale}"
        )

    def flops(self):
        pass


class Upsample(nn.Module):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        super().__init__()
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log2(scale))):
                m.extend((
                    nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1),
                    nn.PixelShuffle(2),
                ))
        elif scale == 3:
            m.extend((nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1), nn.PixelShuffle(3)))
        else:
            msg = f"scale {scale} is not supported. Supported scales: 2^n and 3."
            raise ValueError(msg)
        self.up = nn.Sequential(*m)

    def forward(self, x):
        return self.up(x)


class UpsampleOneStep(nn.Module):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, num_out_ch):
        super().__init__()
        self.num_feat = num_feat
        m = []
        m.extend((
            nn.Conv2d(num_feat, scale**2 * num_out_ch, 3, 1, 1),
            nn.PixelShuffle(scale),
        ))
        self.up = nn.Sequential(*m)

    def forward(self, x):
        return self.up(x)
