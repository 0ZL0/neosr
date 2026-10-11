from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neosr.archs.vgg_arch import VGGFeatureExtractor
from neosr.losses.basic_loss import chc_loss
from neosr.utils.namespaces import normalize_builtin_criterion
from neosr.utils.registry import LOSS_REGISTRY

if TYPE_CHECKING:
    from collections.abc import Callable


class PatchesKernel3D(nn.Module):
    """Adapted from 'Patch Loss: A Generic Multi-Scale Perceptual Loss for
    Single Image Super-resolution':
    https://www.sciencedirect.com/science/article/pii/S0031320323002108
    https://github.com/Suanmd/Patch-Loss-for-Super-Resolution.
    """

    def __init__(
        self, kernelsize: int, kernelstride: int, kernelpadding: int = 0
    ) -> None:
        super().__init__()
        self.kernelsize = kernelsize
        self.stride = kernelstride
        self.padding = kernelpadding

    def forward(self, x: Tensor) -> Tensor:
        batchsize = x.shape[0]
        channels = x.shape[1]
        x = x.reshape(batchsize * channels, x.shape[-2], x.shape[-1]).unsqueeze(1)
        x = F.unfold(x, self.kernelsize, padding=self.padding, stride=self.stride)

        return (
            x.transpose(1, 2)
            .reshape(batchsize, channels, -1, self.kernelsize**2)
            .permute(0, 2, 1, 3)
        )


@LOSS_REGISTRY.register()
class vgg_perceptual_loss(nn.Module):
    """Perceptual loss with VGG19.

    Args:
    ----
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        loss_weight (float): If `loss_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        criterion (str): Criterion used for perceptual loss. Default: 'huber'.
        patchloss (bool): Enables PatchLoss. Default: False.
        ipk (bool): Enables Image Patch Kernel and adds to the final loss. Default: False.
        patch_weight (float): Feature patch weight (kappa in Eq. 18). Default: 1.0.

    """

    def __init__(
        self,
        layer_weights: OrderedDict | dict[str, float] | None = None,
        vgg_type: str = "vgg19",
        use_input_norm: bool = True,
        range_norm: bool = False,
        loss_weight: float = 1.0,
        criterion: str = "builtin:clipped_pseudo_huber_cosine",
        patchloss: bool = False,
        ipk: bool = False,
        patch_weight: float = 1.0,
        scale: int = 4,
        **kwargs,  # noqa: ARG002
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.patch_weights = patch_weight
        self.patchloss = patchloss
        self.ipk = ipk

        if layer_weights is not None:
            self.layer_weights = layer_weights
        else:
            self.layer_weights = {
                "conv1_2": 0.1,
                "conv2_2": 0.1,
                "conv3_4": 1.0,
                "conv4_4": 1.0,
                "conv5_4": 1.0,
            }

        if self.patchloss is False and self.ipk is True:
            msg = "Please enable PatchLoss to use IPK."
            raise ValueError(msg)

        if patchloss:
            if scale == 4:
                self.perceptual_kernels = [4, 8]
                self.ipk_kernels = [7, 11, 15]
            elif scale == 2:
                self.perceptual_kernels = [3, 6]
                self.ipk_kernels = [3, 5, 7]
            else:
                msg = f"PatchLoss does not support upscale ratio {scale}."
                raise NotImplementedError(msg)

        self.vgg = VGGFeatureExtractor(  # type: ignore[reportCallIssue]
            layer_name_list=list(self.layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm,
        )

        self.criterion: nn.L1Loss | nn.MSELoss | nn.HuberLoss | Callable
        self.criterion_type = normalize_builtin_criterion(criterion)
        if self.criterion_type == "l1":
            self.criterion = nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = nn.MSELoss()
        elif self.criterion_type == "huber":
            self.criterion = nn.HuberLoss()
        elif self.criterion_type == "chc":
            self.criterion = chc_loss(loss_lambda=0, clip_min=0, clip_max=1)  # type: ignore[reportCallIssue]
        else:
            msg = f"{criterion} criterion not supported."
            raise NotImplementedError(msg)

    def patch(self, x: Tensor, gt: Tensor, is_ipk: bool = False) -> Tensor:
        """Multi-scale cosine distance with per-image attention (paper Eqs. 15-18).

        The authors' code uses unnormalized std weights; here each scale's
        weights sum to one over the image's spatial patches and channels,
        or zero for a textureless image.
        """
        kernels = self.ipk_kernels if is_ipk else self.perceptual_kernels
        # Keep reductions out of autocast, including on CPU, and preserve FP64.
        with torch.autocast(device_type=x.device.type, enabled=False):
            dtype = torch.promote_types(x.dtype, gt.dtype)
            if dtype in (torch.float16, torch.bfloat16):
                dtype = torch.float32
            x = x.to(dtype=dtype)
            gt = gt.detach().to(dtype=dtype)
            target = gt  # The reference computes IPK attention before clipping.
            if is_ipk:
                x = x.clamp(1e-6, 0.999999)
                gt = gt.clamp(1e-6, 0.999999)

            loss = x.new_zeros(())
            for kernel in kernels:
                # Match the reference implementation's IPK and FPK sampling.
                stride = kernel // 2 + int(is_ipk)
                with torch.no_grad():
                    patches = target.unfold(2, kernel, stride).unfold(3, kernel, stride)
                    # Stable sample std; raw second moments lose low contrast.
                    weights = patches.std(dim=(-2, -1), correction=1)
                    # Do not normalize constant-patch roundoff into attention.
                    varying = patches.amax(dim=(-2, -1)) > patches.amin(dim=(-2, -1))
                    weights = weights * varying
                    total = weights.sum(dim=(1, 2, 3), keepdim=True)
                    # A textureless image has no attention and contributes zero.
                    weights = weights / torch.where(total > 0, total, 1)

                # Sufficient statistics avoid materializing prediction patches.
                # A 1e-4 norm floor extends cosine to zero-norm patches; clamp
                # before sqrt so backward never evaluates sqrt'(0).
                norm_floor2 = 1e-8
                x_norm = (
                    (F.avg_pool2d(x.square(), kernel, stride) * kernel**2)
                    .clamp_min(norm_floor2)
                    .sqrt()
                )
                gt_norm = (
                    (F.avg_pool2d(gt.square(), kernel, stride) * kernel**2)
                    .clamp_min(norm_floor2)
                    .sqrt()
                )
                dot = F.avg_pool2d(x * gt, kernel, stride) * kernel**2
                distance = 1 - dot / x_norm / gt_norm
                loss = loss + (weights * distance).sum(dim=(1, 2, 3)).mean()

        return loss

    def forward(self, x: Tensor, gt: Tensor) -> float:
        """Forward function.

        Args:
        ----
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
        -------
            Float: loss value.

        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())
        percep_loss: float = 0.0
        # calculate perceptual loss
        for k in x_features:
            if self.patchloss:
                percep_loss += (
                    self.patch(x_features[k], gt_features[k])
                    * self.layer_weights[k]
                    * self.patch_weights
                    + self.criterion(x_features[k] * 0.1, gt_features[k] * 0.1)
                    * self.layer_weights[k]
                )
            else:
                percep_loss += (
                    # decrease magnitude to balance other losses
                    self.criterion(x_features[k] * 0.1, gt_features[k] * 0.1)
                    * self.layer_weights[k]
                )
        # add IPK
        if self.patchloss and self.ipk:
            ipk = self.patch(x, gt, is_ipk=True)
            percep_loss += ipk

        return percep_loss * self.loss_weight
