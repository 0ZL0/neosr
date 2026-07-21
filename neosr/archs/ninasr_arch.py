import torch
import torch.nn as nn

from neosr.archs.arch_util import net_opt
from neosr.utils.registry import ARCH_REGISTRY

upscale, __ = net_opt()


class AttentionBlock(nn.Module):
    """
    A typical Squeeze-Excite attention block, with a local pooling instead of global
    """

    def __init__(self, n_feats, reduction=4, stride=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.AvgPool2d(
                2 * stride - 1,
                stride=stride,
                padding=stride - 1,
                count_include_pad=False,
            ),
            nn.Conv2d(n_feats, n_feats // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats // reduction, n_feats, 1, bias=True),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=stride, mode="nearest"),
        )

    def forward(self, x):
        res = self.body(x)
        if res.shape != x.shape:
            res = res[:, :, : x.shape[2], : x.shape[3]]
        return res * x


class ResBlock(nn.Module):
    def __init__(self, n_feats, mid_feats, in_scale, out_scale):
        super().__init__()

        self.in_scale = in_scale
        self.out_scale = out_scale

        m = []
        conv1 = nn.Conv2d(n_feats, mid_feats, 3, padding=1, bias=True)
        nn.init.kaiming_normal_(conv1.weight)
        nn.init.zeros_(conv1.bias)
        m.extend((conv1, nn.ReLU(inplace=True), AttentionBlock(mid_feats)))
        conv2 = nn.Conv2d(mid_feats, n_feats, 3, padding=1, bias=False)
        nn.init.kaiming_normal_(conv2.weight)
        # nn.init.zeros_(conv2.weight)
        m.append(conv2)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x * self.in_scale) * (2 * self.out_scale)
        res += x
        return res


class Rescale(nn.Module):
    def __init__(self, sign):
        super().__init__()
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # bias = sign * torch.Tensor(rgb_mean).reshape(1, 3, 1, 1)
        self.bias = nn.Parameter(torch.tensor(sign), requires_grad=True)

    def forward(self, x):
        return x + self.bias


@ARCH_REGISTRY.register()
class ninasr(nn.Module):
    def __init__(
        self, n_resblocks=26, n_feats=32, n_colors=3, scale=upscale, expansion=2.0
    ):
        super().__init__()
        self.scale = scale
        self.head = ninasr.make_head(n_colors, n_feats)
        self.body = ninasr.make_body(n_resblocks, n_feats, expansion)
        self.tail = ninasr.make_tail(n_colors, n_feats, scale)

    @staticmethod
    def make_head(n_colors, n_feats):
        m_head = [Rescale(-1.0), nn.Conv2d(n_colors, n_feats, 3, padding=1, bias=False)]
        return nn.Sequential(*m_head)

    @staticmethod
    def make_body(n_resblocks, n_feats, expansion):
        mid_feats = int(n_feats * expansion)
        out_scale = 4 / n_resblocks
        expected_variance = torch.tensor(1.0)
        m_body = []
        for _i in range(n_resblocks):
            in_scale = 1.0 / torch.sqrt(expected_variance)
            m_body.append(ResBlock(n_feats, mid_feats, in_scale, out_scale))
            expected_variance += out_scale**2
        return nn.Sequential(*m_body)

    @staticmethod
    def make_tail(n_colors, n_feats, scale):
        m_tail = [
            nn.Conv2d(n_feats, n_colors * scale**2, 3, padding=1, bias=True),
            nn.PixelShuffle(scale),
            Rescale(1.0),
        ]
        return nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        return self.tail(res)


@ARCH_REGISTRY.register()
def ninasr_b0(scale=upscale, **kwargs):  # noqa: ARG001
    return ninasr(n_resblocks=10, n_feats=16, scale=scale)


@ARCH_REGISTRY.register()
def ninasr_b2(scale=upscale, **kwargs):  # noqa: ARG001
    return ninasr(n_resblocks=84, n_feats=56, scale=scale)
