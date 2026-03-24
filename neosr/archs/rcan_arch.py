import math

import torch
from torch import nn

from neosr.archs.arch_util import net_opt
from neosr.utils.registry import ARCH_REGISTRY

upscale, __ = net_opt()


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, rgb_std, sign=-1):
        super().__init__(3, 3, kernel_size=1)
        device = self.weight.device
        dtype = self.weight.dtype
        std = torch.tensor(rgb_std, device=device, dtype=dtype)
        mean = torch.tensor(rgb_mean, device=device, dtype=dtype)

        with torch.no_grad():
            self.weight.copy_(torch.eye(3, device=device, dtype=dtype).view(3, 3, 1, 1))
            self.weight.div_(std.view(3, 1, 1, 1))
            self.bias.copy_(sign * mean)
            self.bias.div_(std)

        for parameter in self.parameters():
            parameter.requires_grad = False


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log2(scale))):
                m.extend((conv(n_feat, 4 * n_feat, 3, bias), nn.PixelShuffle(2)))
                if bn:
                    m.append(nn.BatchNorm2d(n_feat))
                if act:
                    m.append(act())
        elif scale == 3:
            m.extend((conv(n_feat, 9 * n_feat, 3, bias), nn.PixelShuffle(3)))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if act:
                m.append(act())
        else:
            raise NotImplementedError

        super().__init__(*m)


class CALayer(nn.Module):
    """
    Channel Attention (CA) Layer
    """

    def __init__(self, channel, reduction=16):
        super().__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    """
    Residual Channel Attention Block (RCAB)
    """

    def __init__(
        self,
        conv,
        n_feat,
        kernel_size,
        reduction,
        bias=True,
        bn=False,
        act=None,
        res_scale=1,
    ):
        super().__init__()

        if act is None:
            act = nn.ReLU(inplace=True)

        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        return res + x


class ResidualGroup(nn.Module):
    """
    Residual Group (RG)
    """

    def __init__(
        self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks
    ):
        super().__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv,
                n_feat,
                kernel_size,
                reduction,
                bias=True,
                bn=False,
                act=act,
                res_scale=res_scale,
            )
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


@ARCH_REGISTRY.register()
class rcan(nn.Module):
    """
    Residual Channel Attention Network (RCAN)
    """

    def __init__(
        self,
        n_resgroups=10,
        n_resblocks=20,
        n_feats=64,
        kernel_size=3,
        reduction=16,
        n_colors=3,
        scale=upscale,
        act=None,
        norm=False,
        **kwargs,  # noqa: ARG002
    ):
        super().__init__()

        self.norm = norm
        conv = default_conv

        if act is None:
            act = nn.ReLU(inplace=True)

        if self.norm:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            rgb_std = (1.0, 1.0, 1.0)
            self.sub_mean = MeanShift(rgb_mean, rgb_std)
            self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv,
                n_feats,
                kernel_size,
                reduction,
                act=act,
                res_scale=scale,
                n_resblocks=n_resblocks,
            )
            for _ in range(n_resgroups)
        ]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size),
        ]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        if self.norm:
            x = self.sub_mean(x)

        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)

        if self.norm:
            x = self.add_mean(x)

        return x
