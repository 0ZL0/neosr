from copy import deepcopy
from typing import Any

from neosr.metrics.calculate import (
    SHAVED_METRICS,
    calculate_pyiqa,
    calculate_psnr,
    calculate_ssim,
)
from neosr.utils.namespaces import prepare_metric_config
from neosr.utils.registry import METRIC_REGISTRY

__all__ = [
    "calculate_pyiqa",
    "calculate_psnr",
    "calculate_ssim",
]


def calculate_metric(data, opt: dict[str, Any], *, scale: int | None = None) -> float:
    """Calculate metric from data and options.

    Args:
    ----
        data (dict): Metric inputs, i.e. ``img`` and optionally ``img2``.
        opt (dict): Configuration. It must contain:
            type (str): Model type.
        scale (int | None): Upscaling ratio. PSNR and SSIM shave this many pixels
            off every border unless the configuration overrides ``crop_border``,
            which is what the SISR literature reports and what makes the numbers
            comparable to published results.

    """
    opt = deepcopy(opt)
    opt, resolved = prepare_metric_config(opt)
    metric_type = opt.pop("type")
    # 'better' selects the best-result comparison and is not a metric argument.
    opt.pop("better", None)
    if scale is not None and resolved.registry_type in SHAVED_METRICS:
        opt.setdefault("crop_border", scale)
    return METRIC_REGISTRY.get(metric_type)(**data, **opt)  # type: ignore[operator,return-value]
