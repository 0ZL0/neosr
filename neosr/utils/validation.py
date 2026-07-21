from collections.abc import Mapping
from typing import Any


def resolve_validation_save_img(
    dataloader: Any, validation_opt: Mapping[str, Any] | None
) -> bool:
    """Resolve image saving with per-dataset options taking precedence."""
    dataset = getattr(dataloader, "dataset", None)
    dataset_opt = getattr(dataset, "opt", {})
    if not isinstance(dataset_opt, Mapping):
        dataset_opt = {}
    global_default = (validation_opt or {}).get("save_img", True)
    return bool(dataset_opt.get("save_img", global_default))
