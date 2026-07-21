import torch
from torch import nn
from torch.nn import functional as F

from neosr.archs.arch_util import net_opt
from neosr.utils.registry import ARCH_REGISTRY

upscale, __ = net_opt()


class SECA(nn.Module):
    """
    Spatial-Efficient Channel Attention (SECA)
    """

    def __init__(self, channel, reduction=8):  # 16 or 8(mini)
        super().__init__()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False), nn.Sigmoid()
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # spacial attn:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_weight = torch.cat([avg_out, max_out], dim=1)
        spatial_weight = self.spatial_attention(spatial_weight)
        # channel attn (2 MLP)
        channel_weight = self.channel_attention(x)
        return x * spatial_weight * channel_weight


class CSA(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False), nn.Sigmoid()
        )
        # Bi-Directional
        self.channel_attention_forward = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv1d(
                1,
                1,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.Sigmoid(),
        )
        self.channel_attention_backward = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv1d(
                1,
                1,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B,1,H,W)
        spatial_weight = torch.cat([avg_out, max_out], dim=1)  # (B,2,H,W)
        spatial_weight = self.spatial_attention(spatial_weight)  # (B,1,H,W)

        # Bi-Directional attention
        _b, _c, _, _ = x.size()
        y = torch.mean(x, dim=(2, 3), keepdim=True)  # (B,16,1,1)

        y_forward = (
            self.channel_attention_forward(y.squeeze(-1).transpose(-1, -2))
            .transpose(-1, -2)
            .unsqueeze(-1)
        )  # (B,1,1,1)
        y_backward = (
            self.channel_attention_backward(
                y.squeeze(-1).transpose(-1, -2).flip(dims=[1])
            )
            .transpose(-1, -2)
            .unsqueeze(-1)
        )  # (B,1,1,1)

        channel_weight = (y_forward + y_backward.flip(dims=[1])) / 2
        channel_weight = channel_weight.expand_as(x)  # (B,16,H,W)

        return x * spatial_weight * channel_weight


class Conv(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(N, N * 2, 1),
            nn.BatchNorm2d(N * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(N * 2, N, 3, padding=1),
            nn.BatchNorm2d(N),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class FFN(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv2d(N, N * 2, 1),
            nn.BatchNorm2d(N * 2),
            nn.GELU(),
            nn.Conv2d(N * 2, N, 1),
            nn.BatchNorm2d(N),
        )

    def forward(self, x):
        return self.ffn(x) + x


class Attn(nn.Module):
    def __init__(self, N, mini=False):
        super().__init__()
        self.pre_mixer = Conv(N)
        self.post_mixer = FFN(N)
        self.attn = SECA(N, reduction=8) if mini else CSA()
        self.norm1 = nn.BatchNorm2d(N)
        self.norm2 = nn.BatchNorm2d(N)

    def forward(self, x):
        out = self.pre_mixer(x)
        out = self.norm1(out)
        out = self.attn(out)
        out = self.post_mixer(out)
        out = self.norm2(out)
        out += x
        return out


@ARCH_REGISTRY.register()
class sebica(nn.Module):
    def __init__(self, sr_rate=upscale, N=16, mini=False, dropout=0.0, **kwargs):
        super().__init__()
        self.scale = sr_rate
        dropout = dropout if self.training else 0.0
        self.head = nn.Sequential(
            nn.Conv2d(3, N, 3, padding=1), nn.BatchNorm2d(N), nn.ReLU(inplace=True)
        )

        self.body = nn.Sequential(*[
            Attn(N, mini=mini) for _ in range(4 if mini else 6)
        ])

        self.tail = nn.Sequential(
            nn.Conv2d(N, 3 * sr_rate * sr_rate, 1),
            nn.Dropout(dropout),
            nn.PixelShuffle(sr_rate),
        )

    def forward(self, x):
        body_out = self.head(x)
        for attn_layer in self.body:
            body_out = attn_layer(body_out)
        h = self.tail(body_out)
        base = torch.clamp(
            F.interpolate(
                x, scale_factor=self.scale, mode="bilinear", align_corners=False
            ),
            0,
            1,
        )
        return h + base


@ARCH_REGISTRY.register()
def sebica_mini(sr_rate=upscale, **kwargs):  # noqa: ARG001
    return sebica(sr_rate=sr_rate, N=8, mini=True)
