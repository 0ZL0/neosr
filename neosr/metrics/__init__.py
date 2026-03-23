from copy import deepcopy
from typing import Any

from neosr.metrics.calculate import (
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


def calculate_metric(data, opt: dict[str, Any]) -> float:
    """Calculate metric from data and options.

    Args:
    ----
        opt (dict): Configuration. It must contain:
            type (str): Model type.

    """
    opt = deepcopy(opt)
    opt, _ = prepare_metric_config(opt)
    metric_type = opt.pop("type")
    return METRIC_REGISTRY.get(metric_type)(**data, **opt)  # type: ignore[operator,return-value]
