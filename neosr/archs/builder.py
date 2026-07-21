import ast
import importlib
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from torch import nn

from neosr.utils import get_root_logger
from neosr.utils.registry import ARCH_REGISTRY

_SCALE_ARGUMENTS = {
    "scale",
    "sr_rate",
    "upscale",
    "upscale_factor",
    "upscaling_factor",
    "upsampling",
}


@dataclass(frozen=True)
class _ArchitectureSpec:
    module: str
    scale_argument: str | None


def _is_arch_registration(decorator: ast.expr) -> bool:
    return (
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "register"
        and isinstance(decorator.func.value, ast.Name)
        and decorator.func.value.id == "ARCH_REGISTRY"
    )


def _defaulted_arguments(node: ast.FunctionDef) -> list[tuple[str, ast.expr | None]]:
    positional = [*node.args.posonlyargs, *node.args.args]
    defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(node.args.defaults)
    ) + list(node.args.defaults)
    pairs = list(zip((argument.arg for argument in positional), defaults, strict=True))
    pairs.extend(
        (argument.arg, default)
        for argument, default in zip(
            node.args.kwonlyargs, node.args.kw_defaults, strict=True
        )
    )
    return pairs


def _infer_scale_argument(tree: ast.Module, module_stem: str) -> str | None:
    candidates: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for argument, default in _defaulted_arguments(node):
                if isinstance(default, ast.Name) and default.id == "upscale":
                    candidates.add(argument)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and key.value in _SCALE_ARGUMENTS
                    and isinstance(value, ast.Name)
                    and value.id == "upscale"
                ):
                    candidates.add(key.value)

    if len(candidates) > 1:
        joined = ", ".join(sorted(candidates))
        msg = (
            f"Architecture module '{module_stem}' has ambiguous scale arguments: "
            f"{joined}."
        )
        raise RuntimeError(msg)
    return next(iter(candidates), None)


@lru_cache(maxsize=1)
def _architecture_specs() -> dict[str, _ArchitectureSpec]:
    specs: dict[str, _ArchitectureSpec] = {}
    arch_folder = Path(__file__).resolve().parent
    for path in sorted(arch_folder.glob("*_arch.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        scale_argument = _infer_scale_argument(tree, path.stem)
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef | ast.ClassDef):
                continue
            if not any(_is_arch_registration(item) for item in node.decorator_list):
                continue
            if node.name in specs:
                msg = f"Architecture '{node.name}' is registered twice."
                raise RuntimeError(msg)
            specs[node.name] = _ArchitectureSpec(path.stem, scale_argument)
    return specs


def _registered_architecture(network_type: str):
    try:
        return ARCH_REGISTRY.get(network_type)
    except KeyError:
        spec = _architecture_specs().get(network_type)
        if spec is None:
            available = ", ".join(sorted(_architecture_specs()))
            msg = (
                f"Unknown architecture '{network_type}'. "
                f"Available architectures: {available}"
            )
            raise KeyError(msg) from None

        module_name = f"neosr.archs.{spec.module}"
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            dependency = exc.name or "an unknown dependency"
            msg = (
                f"Architecture '{network_type}' requires optional dependency "
                f"'{dependency}', which is not installed."
            )
            raise ModuleNotFoundError(msg, name=dependency) from exc
        return ARCH_REGISTRY.get(network_type)


def build_network(
    opt: dict[str, Any], *, scale: int | None = None
) -> nn.Module | object:
    """Build one network without importing unrelated architecture modules."""
    network_opt = deepcopy(opt)
    network_type = network_opt.pop("type")
    factory = _registered_architecture(network_type)

    spec = _architecture_specs().get(network_type)
    has_explicit_scale = any(key in network_opt for key in _SCALE_ARGUMENTS)
    if (
        scale is not None
        and spec is not None
        and spec.scale_argument is not None
        and not has_explicit_scale
    ):
        network_opt[spec.scale_argument] = scale

    net = factory(**network_opt)  # type: ignore[operator]
    logger = get_root_logger()
    logger.info(f"Using network [{net.__class__.__name__}].")
    return net
