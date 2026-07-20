from collections.abc import Sequence
from math import log2

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention

from neosr.archs.arch_util import DySample, net_opt
from neosr.utils.registry import ARCH_REGISTRY

upscale, __ = net_opt()


def attention(q: Tensor, k: Tensor, v: Tensor, bias: Tensor) -> Tensor:
    score = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
    score = score + bias
    score = F.softmax(score, dim=-1)
    return score @ v


def apply_rpe(table: Tensor, window_size: int):
    def bias_mod(score: Tensor, b: int, h: int, q_idx: int, kv_idx: int):  # noqa: ARG001
        q_h = q_idx // window_size
        q_w = q_idx % window_size
        k_h = kv_idx // window_size
        k_w = kv_idx % window_size
        rel_h = k_h - q_h + window_size - 1
        rel_w = k_w - q_w + window_size - 1
        rel_idx = rel_h * (2 * window_size - 1) + rel_w
        return score + table[h, rel_idx]

    return bias_mod


def feat_to_win(x: Tensor, window_size: Sequence[int], heads: int):
    return rearrange(
        x,
        "b (qkv heads c) (h wh) (w ww) -> qkv (b h w) heads (wh ww) c",
        heads=heads,
        wh=window_size[0],
        ww=window_size[1],
        qkv=3,
    )


def win_to_feat(x, window_size: Sequence[int], h_div: int, w_div: int):
    return rearrange(
        x,
        "(b h w) heads (wh ww) c -> b (heads c) (h wh) (w ww)",
        h=h_div,
        w=w_div,
        wh=window_size[0],
        ww=window_size[1],
    )


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        if self.data_format == "channels_first":
            if self.training:
                return (
                    F.layer_norm(
                        x.permute(0, 2, 3, 1).contiguous(),
                        self.normalized_shape,
                        self.weight,
                        self.bias,
                        self.eps,
                    )
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
            return F.layer_norm(
                x.permute(0, 2, 3, 1),
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            ).permute(0, 3, 1, 2)
        return None


class ConvolutionalAttention(nn.Module):
    def __init__(self, pdim: int):
        super().__init__()
        self.pdim = pdim
        self.sk_size = 3
        self.dwc_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(pdim, pdim // 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(pdim // 2, pdim * self.sk_size * self.sk_size, 1, 1, 0),
        )
        nn.init.zeros_(self.dwc_proj[-1].weight)
        nn.init.zeros_(self.dwc_proj[-1].bias)

    def forward(self, x: Tensor, lk_filter: Tensor) -> Tensor:
        if self.training:
            x1, x2 = torch.split(x, [self.pdim, x.shape[1] - self.pdim], dim=1)

            # Dynamic Conv
            bs = x1.shape[0]
            dynamic_kernel = self.dwc_proj(x[:, : self.pdim]).reshape(
                -1, 1, self.sk_size, self.sk_size
            )
            x1_ = rearrange(x1, "b c h w -> 1 (b c) h w")
            x1_ = F.conv2d(
                x1_,
                dynamic_kernel,
                stride=1,
                padding=self.sk_size // 2,
                groups=bs * self.pdim,
            )
            x1_ = rearrange(x1_, "1 (b c) h w -> b c h w", b=bs, c=self.pdim)

            # Static LK Conv + Dynamic Conv
            x1 = (
                F.conv2d(x1, lk_filter, stride=1, padding=lk_filter.shape[-1] // 2)
                + x1_
            )

            x = torch.cat([x1, x2], dim=1)
        else:
            # for GPU
            dynamic_kernel = self.dwc_proj(x[:, : self.pdim]).reshape(
                -1, 1, self.sk_size, self.sk_size
            )
            x[:, : self.pdim] = F.conv2d(
                x[:, : self.pdim], lk_filter, stride=1, padding=13 // 2
            ) + F.conv2d(
                x[:, : self.pdim],
                dynamic_kernel,
                stride=1,
                padding=self.sk_size // 2,
                groups=self.pdim,
            )

            # For Mobile Conversion, uncomment the following code:
            # x_1, x_2 = torch.split(x, [self.pdim, x.shape[1] - self.pdim], dim=1)
            # dynamic_kernel = self.dwc_proj(x_1).reshape(16, 1, 3, 3)
            # x_1 = F.conv2d(x_1, lk_filter, stride=1, padding=13 // 2) + F.conv2d(
            #    x_1, dynamic_kernel, stride=1, padding=1, groups=16
            # )
            # x = torch.cat([x_1, x_2], dim=1)

        return x


class ConvAttnWrapper(nn.Module):
    def __init__(self, dim: int, pdim: int):
        super().__init__()
        self.plk = ConvolutionalAttention(pdim)
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x: Tensor, lk_filter: Tensor) -> Tensor:
        x = self.plk(x, lk_filter)
        return self.aggr(x)


class DecomposedConvolutionalAttention(nn.Module):
    def __init__(self, pdim: int):
        super().__init__()
        self.pdim = pdim
        self.dynamic_kernel_size = 3
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(pdim, pdim // 4, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(
                pdim // 4,
                pdim * self.dynamic_kernel_size * self.dynamic_kernel_size,
                1,
                1,
                0,
            ),
        )

    def forward(self, x: Tensor, lk_channel: Tensor, lk_spatial: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.pdim, x.shape[1] - self.pdim], dim=1)
        b, c = x1.shape[:2]

        # Dynamic Conv
        kernel = self.proj(x1)
        kernel = rearrange(kernel, "b (c kh kw) 1 1 -> (b c) 1 kh kw", kh=3, kw=3)
        n_pad = (13 - self.dynamic_kernel_size) // 2
        kernel = F.pad(kernel, (n_pad, n_pad, n_pad, n_pad))

        x1 = F.conv2d(x1, lk_channel, padding=0)
        x1 = rearrange(x1, "b c h w -> 1 (b c) h w")
        lk_spatial = lk_spatial.repeat(b, 1, 1, 1)
        x1 = F.conv2d(x1, kernel + lk_spatial, padding=13 // 2, groups=b * c)
        x1 = rearrange(x1, "1 (b c) h w -> b c h w", b=b, c=c)
        return torch.cat([x1, x2], dim=1)


class DecomposedConvolutionalAttentionWrapper(nn.Module):
    def __init__(self, dim: int, pdim: int):
        super().__init__()
        self.pdim = pdim
        self.plk = DecomposedConvolutionalAttention(pdim)
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x: Tensor, lk_channel: Tensor, lk_spatial: Tensor) -> Tensor:
        x = self.plk(x, lk_channel, lk_spatial)
        return self.aggr(x)


class ConvFFN(nn.Module):
    def __init__(self, dim: int, kernel_size: int, exp_ratio: float):
        super().__init__()
        self.proj = nn.Conv2d(dim, int(dim * exp_ratio), 1, 1, 0)
        self.dwc = nn.Conv2d(
            int(dim * exp_ratio),
            int(dim * exp_ratio),
            kernel_size,
            1,
            kernel_size // 2,
            groups=int(dim * exp_ratio),
        )
        self.aggr = nn.Conv2d(int(dim * exp_ratio), dim, 1, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = F.gelu(self.proj(x))
        x = F.gelu(self.dwc(x)) + x
        return self.aggr(x)


class WindowAttention(nn.Module):
    def __init__(
        self, dim: int, window_size: int, num_heads: int, attn_func, attn_type: str
    ):
        super().__init__()
        self.dim = dim
        window_size = (
            (window_size, window_size) if isinstance(window_size, int) else window_size
        )
        self.window_size = window_size
        self.num_heads = num_heads
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, 1, 0)
        self.to_out = nn.Conv2d(dim, dim, 1, 1, 0)

        self.attn_func = attn_func
        self.attn_type = attn_type
        self.relative_position_bias = nn.Parameter(
            torch.randn(
                num_heads, (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
            ).to(torch.float32)
            * 0.001
        )
        if self.attn_type == "flex":
            self.get_rpe = apply_rpe(self.relative_position_bias, window_size[0])
        else:
            self.rpe_idxs = self.create_table_idxs(window_size[0], num_heads)
        self.is_mobile = False

    @staticmethod
    def create_table_idxs(window_size: int, heads: int):
        """
        Transposed idxs of original Swin Transformer
        But much easier to implement and the same relative position distance anyway
        """
        idxs_window = []
        for head in range(heads):
            for h in range(window_size**2):
                for w in range(window_size**2):
                    q_h = h // window_size
                    q_w = h % window_size
                    k_h = w // window_size
                    k_w = w % window_size
                    rel_h = k_h - q_h + window_size - 1
                    rel_w = k_w - q_w + window_size - 1
                    rel_idx = rel_h * (2 * window_size - 1) + rel_w
                    idxs_window.append((head, rel_idx))
        return torch.tensor(idxs_window, dtype=torch.long, requires_grad=False)

    def pad_to_win(self, x: Tensor, h: int, w: int) -> Tensor:
        pad_h = (self.window_size[0] - h % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - w % self.window_size[1]) % self.window_size[1]
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    def to_mobile(self):
        bias = self.relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]
        self.rpe_bias = nn.Parameter(
            bias.reshape(
                1,
                self.num_heads,
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
            )
        )

        del self.relative_position_bias
        del self.rpe_idxs

        self.is_mobile = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input features with shape of (B, C, H, W)
        """
        _, _, h, w = x.shape
        x = self.pad_to_win(x, h, w)
        h_div, w_div = (
            x.shape[2] // self.window_size[0],
            x.shape[3] // self.window_size[1],
        )

        qkv = self.to_qkv(x)
        dtype = qkv.dtype
        qkv = feat_to_win(qkv, self.window_size, self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.attn_type == "flex":
            out = self.attn_func(q, k, v, score_mod=self.get_rpe)
        elif self.attn_type == "sdpa":
            bias = self.relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]
            bias = bias.reshape(
                1,
                self.num_heads,
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
            )
            out = self.attn_func(q, k, v, attn_mask=bias, is_causal=False)
        elif self.attn_type == "naive":
            bias = self.relative_position_bias[self.rpe_idxs[:, 0], self.rpe_idxs[:, 1]]
            bias = bias.reshape(
                1,
                self.num_heads,
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
            )
            out = self.attn_func(q, k, v, bias)
        else:
            msg = f"Attention type {self.attn_type} is not supported."
            raise NotImplementedError(msg)

        out = win_to_feat(out, self.window_size, h_div, w_div)
        return self.to_out(out.to(dtype)[:, :, :h, :w])


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        pdim: int,
        conv_blocks: int,
        window_size: int,
        num_heads: int,
        exp_ratio: float,
        attn_func=None,
        attn_type="flex",
        is_fp: bool = False,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.is_fp = is_fp
        self.layer_norm = layer_norm
        self.ln_proj = LayerNorm(dim)
        if is_fp:
            self.proj = ConvFFN(dim, 3, 1.5)
        else:
            self.proj = ConvFFN(dim, 3, 2)
        self.ln_attn = LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, attn_func, attn_type)

        if is_fp:
            self.lns = nn.ModuleList([LayerNorm(dim) for _ in range(conv_blocks)])
            self.pconvs = nn.ModuleList([
                DecomposedConvolutionalAttentionWrapper(dim, pdim)
                for _ in range(conv_blocks)
            ])
        elif layer_norm:
            self.lns = nn.ModuleList([LayerNorm(dim) for _ in range(conv_blocks)])
            self.pconvs = nn.ModuleList([
                ConvAttnWrapper(dim, pdim) for _ in range(conv_blocks)
            ])
        else:
            self.pconvs = nn.ModuleList([
                ConvAttnWrapper(dim, pdim) for _ in range(conv_blocks)
            ])

        self.convffns = nn.ModuleList([
            ConvFFN(dim, 3, exp_ratio) for _ in range(conv_blocks)
        ])

        self.ln_out = LayerNorm(dim)
        self.conv_out = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(
        self,
        x: Tensor,
        plk_filter: Tensor = None,
        lk_channel: Tensor = None,
        lk_spatial: Tensor = None,
    ) -> Tensor:
        skip = x
        x = self.ln_proj(x)
        x = self.proj(x)
        x = x + self.attn(self.ln_attn(x))

        if self.is_fp:
            for ln, pconv, convffn in zip(
                self.lns, self.pconvs, self.convffns, strict=False
            ):
                x = x + pconv(convffn(ln(x)), lk_channel, lk_spatial)
        elif self.layer_norm:
            for ln, pconv, convffn in zip(
                self.lns, self.pconvs, self.convffns, strict=False
            ):
                x = x + pconv(convffn(ln(x)), plk_filter)
        else:
            for pconv, convffn in zip(self.pconvs, self.convffns, strict=False):
                x = x + pconv(convffn(x), plk_filter)

        x = self.conv_out(self.ln_out(x))
        return x + skip


def _geo_ensemble(k):
    """
    To enhance LK's structural inductive bias, we use Feature-level Geometric Re-parameterization
    as proposed in https://github.com/dslisleedh/IGConv
    """
    k_hflip = k.flip([3])
    k_vflip = k.flip([2])
    k_hvflip = k.flip([2, 3])
    k_rot90 = torch.rot90(k, -1, [2, 3])
    k_rot90_hflip = k_rot90.flip([3])
    k_rot90_vflip = k_rot90.flip([2])
    k_rot90_hvflip = k_rot90.flip([2, 3])
    return (
        k
        + k_hflip
        + k_vflip
        + k_hvflip
        + k_rot90
        + k_rot90_hflip
        + k_rot90_vflip
        + k_rot90_hvflip
    ) / 8


@ARCH_REGISTRY.register()
class esc(nn.Module):
    """
    ESC: Emulating Self-attention with Convolution for Efficient Image Super-Resolution.
    https://arxiv.org/abs/2503.06671

    Code adapted from:
    https://github.com/dslisleedh/ESC
    """

    def __init__(
        self,
        dim: int = 64,
        pdim: int = 16,
        kernel_size: int = 13,
        n_blocks: int = 5,
        conv_blocks: int = 5,
        window_size: int = 32,
        num_heads: int = 4,
        upscaling_factor: int = upscale,
        exp_ratio: float = 1.25,
        attn_type: str = "sdpa",
        is_fp: bool = False,
        use_dysample: bool = True,
        realsr: bool = True,
    ):
        super().__init__()

        self.is_fp = is_fp
        self.realsr = realsr
        self.upscaling_factor = upscaling_factor
        self.proj = nn.Conv2d(3, dim, 3, 1, 1)
        self.last = nn.Conv2d(dim, dim, 3, 1, 1)

        if use_dysample and not realsr:
            msg = "DySample can only be enabled on esc_real."
            raise ValueError(msg)

        attn_type = attn_type.lower()
        if attn_type == "naive":
            attn_func = attention
        elif attn_type == "sdpa":
            attn_func = F.scaled_dot_product_attention
        elif attn_type == "flex":
            attn_func = flex_attention
            #attn_func = torch.compile(flex_attention, dynamic=True)
        else:
            msg = f"Attention type {attn_type} is not supported."
            raise NotImplementedError(msg)

        if is_fp:
            self.lk_channel = nn.Parameter(torch.randn(pdim, pdim, 1, 1))
            self.lk_spatial = nn.Parameter(
                torch.randn(pdim, 1, kernel_size, kernel_size)
            )
            nn.init.orthogonal_(self.lk_spatial)
            self.ln_last = LayerNorm(dim)
        else:
            self.plk_func = _geo_ensemble
            self.plk_filter = nn.Parameter(
                torch.randn(pdim, pdim, kernel_size, kernel_size)
            )
            # Initializing LK filters using orthogonal initialization
            # is important for stabilizing early training phase.
            nn.init.orthogonal_(self.plk_filter)

        self.blocks = nn.ModuleList([
            Block(
                dim,
                pdim,
                conv_blocks,
                window_size,
                num_heads,
                exp_ratio,
                attn_func,
                attn_type,
                is_fp,
                realsr,  # layer_norm
            )
            for _ in range(n_blocks)
        ])

        if not realsr:
            self.to_img = nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1)
        else:
            if use_dysample:
                self.to_img = DySample(
                    dim,  # Cin
                    3,  # Cout
                    upscaling_factor,
                    groups=4,  # DySample groups. diversify coordinates estimation
                    end_convolution=True,
                )
            else:
                layers = []
                num_upscales = int(log2(self.upscaling_factor))
                for _ in range(num_upscales):
                    layers.extend([
                        nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv2d(dim, dim, 3, 1, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(dim, dim, 3, 1, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                    ])
                layers.append(nn.Conv2d(dim, 3, 3, 1, 1))
                self.to_img = nn.Sequential(*layers)

            self.skip = nn.Sequential(
                nn.Conv2d(3, dim * 2, 1, 1, 0),
                nn.Conv2d(
                    dim * 2, dim * 2, 7, 1, 3, groups=dim * 2, padding_mode="reflect"
                ),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim * 2, dim, 1, 1, 0),
            )

    @torch.no_grad()
    def convert(self):
        self.plk_filter = nn.Parameter(self.plk_func(self.plk_filter))
        self.plk_func = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        feat = self.proj(x)
        skip = feat

        if not self.is_fp:
            plk_filter = self.plk_func(self.plk_filter)

        if self.is_fp:
            for block in self.blocks:
                feat = block(
                    feat, lk_channel=self.lk_channel, lk_spatial=self.lk_spatial
                )
            feat = self.last(self.ln_last(feat)) + skip
            feat = self.to_img(feat)
            return F.pixel_shuffle(feat, self.upscaling_factor) + F.interpolate(
                x, scale_factor=self.upscaling_factor, mode="bicubic"
            )

        for block in self.blocks:
            feat = block(feat, plk_filter=plk_filter)
        feat_ = self.last(feat) + skip

        if self.realsr:
            return self.to_img(feat_ + self.skip(x))
        return F.pixel_shuffle(
            self.to_img(feat_)
            + torch.repeat_interleave(x, self.upscaling_factor**2, dim=1),
            self.upscaling_factor,
        )


@ARCH_REGISTRY.register()
def esc_light(**kwargs):
    return esc(n_blocks=3, **kwargs)


@ARCH_REGISTRY.register()
def esc_fp(**kwargs):
    kwargs.setdefault("use_dysample", False)
    return esc(dim=48, num_heads=3, is_fp=True, realsr=False, **kwargs)


@ARCH_REGISTRY.register()
def esc_large(**kwargs):
    return esc(n_blocks=10, exp_ratio=2.0, **kwargs)
