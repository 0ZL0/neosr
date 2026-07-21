"""neosr package.

The package root intentionally stays lightweight. Public helpers are resolved lazily so
that importing :mod:`neosr` never parses command-line arguments or imports every model,
dataset, loss, and optional architecture dependency.
"""

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "AvgTimer": "neosr.utils",
    "DiffJPEG": "neosr.utils",
    "MessageLogger": "neosr.utils",
    "Registry": "neosr.utils",
    "bgr2ycbcr": "neosr.utils",
    "build_network": "neosr.archs",
    "build_dataloader": "neosr.data",
    "build_dataset": "neosr.data",
    "build_loss": "neosr.losses",
    "check_disk_space": "neosr.utils",
    "check_resume": "neosr.utils",
    "calculate_pyiqa": "neosr.metrics",
    "calculate_psnr": "neosr.metrics",
    "calculate_ssim": "neosr.metrics",
    "build_model": "neosr.models",
    "crop_border": "neosr.utils",
    "get_root_logger": "neosr.utils",
    "get_time_str": "neosr.utils",
    "imfrombytes": "neosr.utils",
    "img2tensor": "neosr.utils",
    "imwrite": "neosr.utils",
    "init_tb_logger": "neosr.utils",
    "init_wandb_logger": "neosr.utils",
    "make_exp_dirs": "neosr.utils",
    "mkdir_and_rename": "neosr.utils",
    "rgb2ycbcr": "neosr.utils",
    "rgb2ycbcr_pt": "neosr.utils",
    "scandir": "neosr.utils",
    "set_random_seed": "neosr.utils",
    "sizeof_fmt": "neosr.utils",
    "tc": "neosr.utils",
    "tensor2img": "neosr.utils",
    "toml_load": "neosr.utils",
    "ycbcr2bgr": "neosr.utils",
    "ycbcr2rgb": "neosr.utils",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_EXPORTS.keys())
