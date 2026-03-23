import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neosr.archs.arch_util import net_opt
from neosr.utils.registry import ARCH_REGISTRY

upscale, __ = net_opt()


def conv_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    padding: int | None = None,
) -> nn.Conv2d:
    if padding is None:
        padding = ((kernel_size - 1) // 2) * dilation
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=padding,
        bias=True,
        dilation=dilation,
        groups=groups,
    )


def activation(
    act_type: str,
    inplace: bool = True,
    neg_slope: float = 0.05,
    n_prelu: int = 1,
) -> nn.Module:
    act_type = act_type.lower()
    if act_type == "relu":
        return nn.ReLU(inplace=inplace)
    if act_type == "lrelu":
        return nn.LeakyReLU(neg_slope, inplace=inplace)
    if act_type == "prelu":
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)

    msg = f"Unsupported activation type: {act_type}"
    raise NotImplementedError(msg)


def sequential(*args: nn.Module | None) -> nn.Sequential | nn.Module:
    modules: list[nn.Module] = []
    for module in args:
        if module is None:
            continue
        if isinstance(module, nn.Sequential):
            modules.extend(module.children())
        else:
            modules.append(module)

    if len(modules) == 1:
        return modules[0]
    return nn.Sequential(*modules)


def conv_block(
    in_nc: int,
    out_nc: int,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    act_type: str | None = "relu",
) -> nn.Sequential | nn.Module:
    padding = ((kernel_size + (kernel_size - 1) * (dilation - 1)) - 1) // 2
    conv = nn.Conv2d(
        in_nc,
        out_nc,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        groups=groups,
    )
    act = activation(act_type) if act_type else None
    return sequential(conv, act)


def pixelshuffle_block(
    in_channels: int,
    out_channels: int,
    upscale_factor: int = 2,
    kernel_size: int = 3,
    stride: int = 1,
) -> nn.Sequential | nn.Module:
    conv = conv_layer(
        in_channels,
        out_channels * (upscale_factor**2),
        kernel_size,
        stride,
    )
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class ESA(nn.Module):
    def __init__(self, n_feats: int, conv: type[nn.Conv2d]) -> None:
        super().__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(
            c3, (x.size(2), x.size(3)), mode="bilinear", align_corners=False
        )
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


class RFDB(nn.Module):
    def __init__(self, in_channels: int, distillation_rate: float = 0.25) -> None:
        super().__init__()
        # The official implementation keeps half the channels regardless of
        # distillation_rate. Preserve that behavior for checkpoint compatibility.
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        self.distillation_rate = distillation_rate

        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation("lrelu", neg_slope=0.05)
        self.c5 = conv_layer(self.dc * 4, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, x: Tensor) -> Tensor:
        distilled_c1 = self.act(self.c1_d(x))
        r_c1 = self.c1_r(x)
        r_c1 = self.act(r_c1 + x)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.c2_r(r_c1)
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.c3_r(r_c2)
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))

        return out_fused


@ARCH_REGISTRY.register()
class rfdn(nn.Module):
    """Residual Feature Distillation Network."""

    def __init__(
        self,
        in_nc: int = 3,
        nf: int = 50,
        num_modules: int = 4,
        out_nc: int = 3,
        upscale: int = upscale,
        scale: int | None = None,
        **kwargs,  # noqa: ARG002
    ) -> None:
        super().__init__()

        if scale is not None:
            upscale = scale
        if num_modules != 4:
            msg = "RFDN only supports num_modules=4 to remain checkpoint-compatible."
            raise ValueError(msg)

        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = RFDB(in_channels=nf)
        self.B2 = RFDB(in_channels=nf)
        self.B3 = RFDB(in_channels=nf)
        self.B4 = RFDB(in_channels=nf)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type="lrelu")

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        self.upsampler = pixelshuffle_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0

    def forward(self, x: Tensor) -> Tensor:
        out_fea = self.fea_conv(x)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        return self.upsampler(out_lr)

    def set_scale(self, scale_idx: int) -> None:
        self.scale_idx = scale_idx
