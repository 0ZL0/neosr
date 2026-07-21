import ast
import importlib
from collections.abc import Callable
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from neosr.utils import get_root_logger
from neosr.utils.registry import MODEL_REGISTRY


def _is_model_registration(decorator: ast.expr) -> bool:
    return (
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "register"
        and isinstance(decorator.func.value, ast.Name)
        and decorator.func.value.id == "MODEL_REGISTRY"
    )


@lru_cache(maxsize=1)
def _model_modules() -> dict[str, str]:
    modules: dict[str, str] = {}
    model_folder = Path(__file__).resolve().parent
    for path in sorted(model_folder.glob("*.py")):
        if path.name in {"__init__.py", "builder.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef | ast.ClassDef):
                continue
            if not any(_is_model_registration(item) for item in node.decorator_list):
                continue
            if node.name in modules:
                msg = f"Model '{node.name}' is registered twice."
                raise RuntimeError(msg)
            modules[node.name] = path.stem
    return modules


def _registered_model(model_type: str):
    try:
        return MODEL_REGISTRY.get(model_type)
    except KeyError:
        module = _model_modules().get(model_type)
        if module is None:
            available = ", ".join(sorted(_model_modules()))
            msg = f"Unknown model type '{model_type}'. Available models: {available}"
            raise KeyError(msg) from None
        importlib.import_module(f"neosr.models.{module}")
        return MODEL_REGISTRY.get(model_type)


def build_model(opt: dict[str, Any]) -> Callable | object:
    """Build only the model selected by ``model_type``."""
    model_opt = deepcopy(opt)
    model_type = model_opt["model_type"]
    model = _registered_model(model_type)(model_opt)  # type: ignore[operator]
    logger = get_root_logger()
    logger.info(f"Using model [{model.__class__.__name__}].")
    return model
