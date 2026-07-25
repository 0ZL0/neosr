import math
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from neosr.utils.logger import get_root_logger
from neosr.utils.rng import rng

rng = rng(__name__)


def _validate_pair(img_gt: Tensor, img_lq: Tensor) -> None:
    if img_gt.ndim != 4 or img_lq.ndim != 4:
        msg = "img_gt and img_lq must be 4D NCHW tensors."
        raise ValueError(msg)
    if img_gt.size(0) != img_lq.size(0):
        msg = "img_gt and img_lq must have the same batch size."
        raise ValueError(msg)
    if img_gt.size(1) != img_lq.size(1):
        msg = "img_gt and img_lq must have the same number of channels."
        raise ValueError(msg)


def _scale_factors(img_gt: Tensor, img_lq: Tensor) -> tuple[int, int]:
    """Return the exact GT-to-LQ spatial scale for aligned SR pairs."""
    gt_h, gt_w = img_gt.shape[-2:]
    lq_h, lq_w = img_lq.shape[-2:]
    if lq_h == 0 or lq_w == 0 or gt_h % lq_h != 0 or gt_w % lq_w != 0:
        msg = (
            "img_gt spatial dimensions must be integer multiples of img_lq "
            "for spatial pair augmentations."
        )
        raise ValueError(msg)
    return gt_h // lq_h, gt_w // lq_w


def _rand_bbox(height: int, width: int, ratio: float) -> tuple[int, int, int, int]:
    """Sample a non-empty box in (top, bottom, left, right) order."""
    box_h = min(height, max(1, int(height * ratio)))
    box_w = min(width, max(1, int(width * ratio)))
    top = int(rng.integers(0, height - box_h + 1))  # type: ignore[attr-defined]
    left = int(rng.integers(0, width - box_w + 1))  # type: ignore[attr-defined]
    return top, top + box_h, left, left + box_w


def _paired_permutation(img_gt: Tensor, img_lq: Tensor) -> tuple[Tensor, Tensor]:
    index_gt = torch.randperm(img_gt.size(0), device=img_gt.device)
    return index_gt, index_gt.to(img_lq.device)


def _clamp_to_source_range(image: Tensor, source: Tensor) -> Tensor:
    """Limit interpolation overshoot without assuming an unnormalized [0, 1] domain."""
    lower = source.amin(dim=(-2, -1), keepdim=True)
    upper = source.amax(dim=(-2, -1), keepdim=True)
    return torch.maximum(torch.minimum(image, upper), lower)


@torch.no_grad()
def mixup(img_gt: Tensor, img_lq: Tensor, alpha: float = 1.2) -> tuple[Tensor, Tensor]:
    r"""Apply SR MixUp to an aligned GT/LQ batch at each tensor's native scale.

    The same permutation and :math:`\lambda \sim Beta(\alpha, \alpha)` are used
    for both sides of every training pair. GT and LQ therefore need matching batch
    and channel dimensions, but their spatial resolutions may differ.

    References:
        - Suppressing Model Overfitting for Image Super-Resolution Networks
          (CVPRW 2019), https://arxiv.org/abs/1906.04809
        - Rethinking Data Augmentation for Image Super-resolution (CVPR 2020),
          https://arxiv.org/abs/2004.00448

    Args:
        img_gt: Ground-truth images in NCHW format.
        img_lq: Corresponding low-quality images in NCHW format.
        alpha: Positive shape parameter of the symmetric Beta distribution.

    Returns:
        The mixed GT/LQ pair.

    """
    _validate_pair(img_gt, img_lq)
    if not math.isfinite(alpha) or alpha <= 0:
        msg = "MixUp alpha must be a finite value greater than zero."
        raise ValueError(msg)

    lam = float(rng.beta(alpha, alpha))  # type: ignore[attr-defined]
    index_gt, index_lq = _paired_permutation(img_gt, img_lq)
    img_gt = lam * img_gt + (1 - lam) * img_gt[index_gt]
    img_lq = lam * img_lq + (1 - lam) * img_lq[index_lq]
    return img_gt, img_lq


@torch.no_grad()
def cutmix(img_gt: Tensor, img_lq: Tensor, alpha: float = 0.9) -> tuple[Tensor, Tensor]:
    r"""Apply CutMix to corresponding native-scale regions of an SR pair.

    This project retains its historical sampling rule:
    :math:`\lambda \sim Uniform(0, \alpha)` and the pasted side-length ratio is
    :math:`\sqrt{1-\lambda}`. Increasing ``alpha`` therefore decreases the
    expected pasted area. With the default ``alpha=0.9``, its nominal mean is
    approximately 55% before integer rounding.

    References:
        - CutMix: Regularization Strategy to Train Strong Classifiers with
          Localizable Features (ICCV 2019), https://arxiv.org/abs/1905.04899
        - Rethinking Data Augmentation for Image Super-resolution (CVPR 2020),
          https://arxiv.org/abs/2004.00448

    Args:
        img_gt: Ground-truth images in NCHW format.
        img_lq: Corresponding low-quality images in NCHW format.
        alpha: Upper bound for the uniformly sampled ``lambda``.

    Returns:
        The region-mixed GT/LQ pair.

    """
    _validate_pair(img_gt, img_lq)
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        msg = "CutMix alpha must be a finite value in [0, 1]."
        raise ValueError(msg)

    scale_h, scale_w = _scale_factors(img_gt, img_lq)
    lam = float(rng.uniform(0, alpha))  # type: ignore[attr-defined]
    ratio = math.sqrt(1 - lam)
    top, bottom, left, right = _rand_bbox(*img_lq.shape[-2:], ratio)
    gt_top, gt_bottom = top * scale_h, bottom * scale_h
    gt_left, gt_right = left * scale_w, right * scale_w
    index_gt, index_lq = _paired_permutation(img_gt, img_lq)

    gt_aug = img_gt.clone()
    lq_aug = img_lq.clone()
    gt_aug[..., gt_top:gt_bottom, gt_left:gt_right] = img_gt[
        index_gt, :, gt_top:gt_bottom, gt_left:gt_right
    ]
    lq_aug[..., top:bottom, left:right] = img_lq[index_lq, :, top:bottom, left:right]
    return gt_aug, lq_aug


@torch.no_grad()
def resizemix(
    img_gt: Tensor, img_lq: Tensor, scope: tuple[float, float] = (0.2, 0.9)
) -> tuple[Tensor, Tensor]:
    r"""Resize a paired donor and paste it into corresponding native-scale regions.

    The same donor image and side-length ratio are used for GT and LQ. The donor
    is resized independently at each tensor's native resolution, then pasted into
    scale-mapped boxes. ``scope`` controls the uniformly sampled side-length
    ratio, so the nominal pasted-area ratio is its square.

    Reference:
        ResizeMix: Mixing Data with Preserved Object Information and True Labels,
        https://arxiv.org/abs/2012.11101

    Args:
        img_gt: Ground-truth images in NCHW format.
        img_lq: Corresponding low-quality images in NCHW format.
        scope: Lower and upper bounds for the donor side-length ratio.

    Returns:
        The resized-region-mixed GT/LQ pair.

    """
    _validate_pair(img_gt, img_lq)
    if (
        len(scope) != 2
        or not all(math.isfinite(value) for value in scope)
        or not 0 < scope[0] <= scope[1] <= 1
    ):
        msg = "ResizeMix scope must satisfy 0 < min <= max <= 1."
        raise ValueError(msg)

    scale_h, scale_w = _scale_factors(img_gt, img_lq)
    ratio = float(rng.uniform(*scope))  # type: ignore[attr-defined]
    top, bottom, left, right = _rand_bbox(*img_lq.shape[-2:], ratio)
    box_h, box_w = bottom - top, right - left
    gt_top, gt_bottom = top * scale_h, bottom * scale_h
    gt_left, gt_right = left * scale_w, right * scale_w
    index_gt, index_lq = _paired_permutation(img_gt, img_lq)

    gt_source = img_gt[index_gt]
    lq_source = img_lq[index_lq]
    gt_patch = _clamp_to_source_range(
        F.interpolate(
            gt_source,
            size=(box_h * scale_h, box_w * scale_w),
            mode="bicubic",
            align_corners=False,
        ),
        gt_source,
    )
    lq_patch = _clamp_to_source_range(
        F.interpolate(
            lq_source, size=(box_h, box_w), mode="bicubic", align_corners=False
        ),
        lq_source,
    )

    gt_aug = img_gt.clone()
    lq_aug = img_lq.clone()
    gt_aug[..., gt_top:gt_bottom, gt_left:gt_right] = gt_patch
    lq_aug[..., top:bottom, left:right] = lq_patch
    return gt_aug, lq_aug


_AUGMENTATIONS = {"cutmix": cutmix, "mixup": mixup, "resizemix": resizemix}
_BATCH_ONE_NOOPS = frozenset({"cutmix", "mixup"})


def validate_augment_options(
    augs: Sequence[str] | str, prob: Sequence[float] | float | None
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Validate and normalize one categorical pair-augmentation policy."""
    aug_names = (augs,) if isinstance(augs, str) else tuple(augs)
    if not aug_names:
        msg = "'augmentation' must contain at least one value."
        raise ValueError(msg)
    if len(set(aug_names)) != len(aug_names):
        msg = "'augmentation' must not contain duplicate values."
        raise ValueError(msg)
    if "cutblur" in aug_names:
        msg = (
            "CutBlur was intentionally removed because native-LR scale > 1 "
            "training cannot reproduce its HR-grid learning signal. Remove "
            "'cutblur' and its corresponding 'aug_prob' entry."
        )
        raise ValueError(msg)

    unknown = set(aug_names) - {"none", *_AUGMENTATIONS}
    if unknown:
        msg = f"Unsupported augmentation(s): {', '.join(sorted(unknown))}."
        raise ValueError(msg)

    if prob is None:
        msg = "'aug_prob' must be provided when 'augmentation' is enabled."
        raise ValueError(msg)
    probabilities = (
        (float(prob),) * len(aug_names)
        if isinstance(prob, int | float)
        else tuple(float(value) for value in prob)
    )
    if len(aug_names) != len(probabilities):
        msg = "Length of 'augmentation' and 'aug_prob' must match."
        raise ValueError(msg)
    if any(
        not math.isfinite(probability) or probability < 0
        for probability in probabilities
    ):
        msg = "'aug_prob' values must be finite and non-negative."
        raise ValueError(msg)
    total = math.fsum(probabilities)
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        msg = (
            "'aug_prob' values are categorical probabilities and must sum to "
            f"1.0; received {total:.12g}."
        )
        raise ValueError(msg)
    # Normalize the negligible floating-point summation error so NumPy receives
    # a distribution whose sum is exactly representable enough for choice().
    return aug_names, tuple(probability / total for probability in probabilities)


def resolve_augment_options(
    options: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[float, ...]] | None:
    """Resolve a dataset's optional pair-augmentation configuration."""
    augs = options.get("augmentation")
    probabilities = options.get("aug_prob")
    if augs is None:
        if probabilities is not None:
            msg = "'aug_prob' requires 'augmentation' to be configured."
            raise ValueError(msg)
        return None
    return validate_augment_options(augs, probabilities)


def _select_augmentation(
    augs: tuple[str, ...], probabilities: tuple[float, ...]
) -> str:
    index = int(rng.choice(len(augs), p=probabilities))  # type: ignore[attr-defined]
    return augs[index]


@cache
def _warn_batch_one_mix_noop() -> None:
    get_root_logger().warning(
        "Batch size 1 makes MixUp and CutMix no-ops because no different donor "
        "sample is available; the selected operation will be skipped."
    )


def apply_augment(
    img_gt: Tensor,
    img_lq: Tensor,
    augs: Sequence[str] | str,
    prob: Sequence[float] | float | None,
) -> tuple[Tensor, Tensor]:
    r"""Select at most one pair-preserving SR augmentation.

    ``prob`` values are the actual categorical probabilities and must sum to one.
    Exactly one outcome is drawn, including the exclusive ``none`` outcome. Pair
    augmentations are therefore never stacked with one another.

    All operations use the GT/LQ tensors at their native spatial resolutions and
    infer exact integer scale factors from their shapes.

    Args:
        img_gt: Ground-truth images in NCHW format.
        img_lq: Corresponding low-quality images in NCHW format.
        augs: One or more of ``none``, ``mixup``, ``cutmix`` and ``resizemix``.
        prob: Corresponding non-negative categorical probabilities summing to one.

    Returns:
        The augmented GT/LQ pair. An exclusive ``none`` draw returns the original
        tensor objects.

    """
    _validate_pair(img_gt, img_lq)
    aug_names, probabilities = validate_augment_options(augs, prob)
    selected = _select_augmentation(aug_names, probabilities)
    if selected == "none":
        return img_gt, img_lq

    if img_gt.size(0) == 1 and selected in _BATCH_ONE_NOOPS:
        _warn_batch_one_mix_noop()
        return img_gt, img_lq

    return _AUGMENTATIONS[selected](img_gt, img_lq)
