from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

_WARNED_MESSAGES: set[str] = set()


LEGACY_TRAIN_LOSS_KEYS: tuple[str, ...] = (
    "pixel_opt",
    "mssim_opt",
    "fdl_opt",
    "ncc_opt",
    "kl_opt",
    "consistency_opt",
    "msswd_opt",
    "perceptual_opt",
    "pyiqa_opt",
    "gan_opt",
    "ldl_opt",
    "ff_opt",
)

LOSS_TYPE_ALIASES: dict[str, str] = {
    "L1Loss": "native:l1",
    "MSELoss": "native:mse",
    "HuberLoss": "native:huber",
    "chc_loss": "native:clipped_pseudo_huber_cosine",
    "mssim_loss": "neosr:mssim",
    "fdl_loss": "neosr:fdl",
    "ncc_loss": "neosr:ncc",
    "kl_loss": "neosr:kl",
    "consistency_loss": "neosr:consistency",
    "msswd_loss": "neosr:msswd",
    "vgg_perceptual_loss": "neosr:vgg_perceptual",
    "pyiqa_loss": "pyiqa",
    "gan_loss": "neosr:gan",
    "ldl_loss": "neosr:ldl",
    "ff_loss": "neosr:ff",
}

METRIC_TYPE_ALIASES: dict[str, str] = {
    "calculate_psnr": "native:psnr",
    "calculate_ssim": "native:ssim",
    "calculate_pyiqa": "pyiqa",
}

CRITERION_ALIASES: dict[str, str] = {
    "l1": "l1",
    "native:l1": "l1",
    "l2": "l2",
    "mse": "l2",
    "native:l2": "l2",
    "native:mse": "l2",
    "huber": "huber",
    "native:huber": "huber",
    "chc": "chc",
    "native:chc": "chc",
    "native:clipped_pseudo_huber_cosine": "chc",
}

LEGACY_TRAIN_LOSS_NAMES: dict[str, str] = {
    "pixel_opt": "pixel",
    "mssim_opt": "mssim",
    "fdl_opt": "fdl",
    "ncc_opt": "ncc",
    "kl_opt": "kl",
    "consistency_opt": "consistency",
    "msswd_opt": "msswd",
    "perceptual_opt": "perceptual",
    "pyiqa_opt": "pyiqa",
    "gan_opt": "gan",
    "ldl_opt": "ldl",
    "ff_opt": "ff",
}


@dataclass(frozen=True)
class ResolvedLossType:
    canonical_type: str
    registry_type: str
    family: str | None
    call_kind: str


@dataclass(frozen=True)
class ResolvedMetricType:
    canonical_type: str
    registry_type: str


def _warn(message: str) -> None:
    if message in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(message)
    warnings.warn(message, UserWarning, stacklevel=3)


def _split_namespaced_type(value: str) -> tuple[str | None, str]:
    if ":" not in value:
        return None, value
    provider, name = value.split(":", 1)
    return provider, name


def resolve_loss_type(loss_type: str, *, warn_legacy: bool = True) -> ResolvedLossType:
    provider, name = _split_namespaced_type(loss_type)
    if provider is None:
        canonical = LOSS_TYPE_ALIASES.get(loss_type)
        if canonical is None:
            msg = f"Unsupported loss type '{loss_type}'."
            raise ValueError(msg)
        if warn_legacy:
            _warn(
                f"Legacy loss type '{loss_type}' is deprecated; use '{canonical}' instead."
            )
        return resolve_loss_type(canonical, warn_legacy=False)

    if provider == "native":
        native_map = {
            "l1": ("L1Loss", "reconstruction", "standard"),
            "mse": ("MSELoss", "reconstruction", "standard"),
            "l2": ("MSELoss", "reconstruction", "standard"),
            "huber": ("HuberLoss", "reconstruction", "standard"),
            "clipped_pseudo_huber_cosine": (
                "chc_loss",
                "reconstruction",
                "standard",
            ),
            "chc": ("chc_loss", "reconstruction", "standard"),
        }
        if name not in native_map:
            msg = f"Unsupported native loss '{loss_type}'."
            raise ValueError(msg)
        registry_type, family, call_kind = native_map[name]
        canonical = (
            "native:mse"
            if name == "l2"
            else (
                "native:clipped_pseudo_huber_cosine"
                if name == "chc"
                else f"native:{name}"
            )
        )
        if warn_legacy and canonical != loss_type:
            _warn(f"Loss alias '{loss_type}' is deprecated; use '{canonical}' instead.")
        return ResolvedLossType(canonical, registry_type, family, call_kind)

    if provider == "neosr":
        neosr_map = {
            "mssim": ("mssim_loss", "reconstruction", "standard"),
            "fdl": ("fdl_loss", "perceptual", "standard"),
            "ncc": ("ncc_loss", None, "standard"),
            "kl": ("kl_loss", None, "standard"),
            "consistency": ("consistency_loss", None, "match_target"),
            "msswd": ("msswd_loss", None, "match_target"),
            "vgg_perceptual": ("vgg_perceptual_loss", "perceptual", "standard"),
            "gan": ("gan_loss", "adversarial", "gan"),
            "ldl": ("ldl_loss", None, "standard"),
            "ff": ("ff_loss", None, "standard"),
        }
        if name not in neosr_map:
            msg = f"Unsupported neosr loss '{loss_type}'."
            raise ValueError(msg)
        registry_type, family, call_kind = neosr_map[name]
        return ResolvedLossType(loss_type, registry_type, family, call_kind)

    if provider == "pyiqa":
        if not name:
            msg = "pyiqa loss type must include a metric name, e.g. 'pyiqa:lpips'."
            raise ValueError(msg)
        return ResolvedLossType(loss_type, "pyiqa_loss", "perceptual", "standard")

    msg = f"Unsupported loss provider '{provider}' in '{loss_type}'."
    raise ValueError(msg)


def resolve_metric_type(
    metric_type: str, *, warn_legacy: bool = True
) -> ResolvedMetricType:
    provider, name = _split_namespaced_type(metric_type)
    if provider is None:
        canonical = METRIC_TYPE_ALIASES.get(metric_type)
        if canonical is None:
            msg = f"Unsupported metric type '{metric_type}'."
            raise ValueError(msg)
        if warn_legacy:
            _warn(
                f"Legacy metric type '{metric_type}' is deprecated; use '{canonical}' instead."
            )
        return resolve_metric_type(canonical, warn_legacy=False)

    if provider == "native":
        native_map = {
            "psnr": "calculate_psnr",
            "ssim": "calculate_ssim",
        }
        registry_type = native_map.get(name)
        if registry_type is None:
            msg = f"Unsupported native metric '{metric_type}'."
            raise ValueError(msg)
        return ResolvedMetricType(metric_type, registry_type)

    if provider == "pyiqa":
        if not name:
            msg = "pyiqa metric type must include a metric name, e.g. 'pyiqa:lpips'."
            raise ValueError(msg)
        return ResolvedMetricType(metric_type, "calculate_pyiqa")

    msg = f"Unsupported metric provider '{provider}' in '{metric_type}'."
    raise ValueError(msg)


def prepare_loss_config(
    opt: dict[str, Any], *, warn_legacy: bool = True
) -> tuple[dict[str, Any], ResolvedLossType]:
    cfg = deepcopy(opt)
    resolved = resolve_loss_type(cfg["type"], warn_legacy=warn_legacy)
    cfg["type"] = resolved.registry_type
    if "weight" in cfg:
        if "loss_weight" in cfg:
            msg = f"Loss '{cfg.get('name', resolved.canonical_type)}' cannot set both 'weight' and 'loss_weight'."
            raise ValueError(msg)
        cfg["loss_weight"] = cfg.pop("weight")
    if resolved.registry_type == "pyiqa_loss":
        metric_name = resolved.canonical_type.split(":", 1)[1]
        if cfg.get("metric_name") not in {None, metric_name}:
            msg = (
                f"Loss '{cfg.get('name', metric_name)}' has mismatched metric_name "
                f"'{cfg['metric_name']}' for type '{resolved.canonical_type}'."
            )
            raise ValueError(msg)
        cfg["metric_name"] = metric_name
    elif cfg.get("metric_name") is not None:
        _warn(
            f"Ignoring unexpected 'metric_name' on non-pyiqa loss "
            f"'{cfg.get('name', resolved.canonical_type)}'."
        )
        cfg.pop("metric_name", None)
    return cfg, resolved


def prepare_metric_config(
    opt: dict[str, Any], *, warn_legacy: bool = True
) -> tuple[dict[str, Any], ResolvedMetricType]:
    cfg = deepcopy(opt)
    resolved = resolve_metric_type(cfg["type"], warn_legacy=warn_legacy)
    cfg["type"] = resolved.registry_type
    if resolved.registry_type == "calculate_pyiqa":
        metric_name = resolved.canonical_type.split(":", 1)[1]
        if cfg.get("metric_name") not in {None, metric_name}:
            msg = (
                f"Metric '{cfg.get('name', metric_name)}' has mismatched metric_name "
                f"'{cfg['metric_name']}' for type '{resolved.canonical_type}'."
            )
            raise ValueError(msg)
        cfg["metric_name"] = metric_name
    return cfg, resolved


def normalize_native_criterion(value: str, field_name: str = "criterion") -> str:
    normalized = CRITERION_ALIASES.get(value)
    if normalized is None:
        msg = f"Unsupported {field_name} value '{value}'."
        raise ValueError(msg)

    if ":" not in value:
        canonical = {
            "l1": "native:l1",
            "l2": "native:mse",
            "mse": "native:mse",
            "huber": "native:huber",
            "chc": "native:clipped_pseudo_huber_cosine",
        }[value]
        _warn(f"Legacy {field_name} value '{value}' is deprecated; use '{canonical}' instead.")
    elif value == "native:l2":
        _warn("Criterion alias 'native:l2' is deprecated; use 'native:mse' instead.")
    elif value == "native:chc":
        _warn(
            "Criterion alias 'native:chc' is deprecated; use "
            "'native:clipped_pseudo_huber_cosine' instead."
        )

    return normalized
