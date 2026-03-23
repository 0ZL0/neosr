import argparse
import os
import random
import sys
import time
import tomllib
import warnings
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path, PosixPath
from shutil import copyfile
from typing import Any

import torch

from neosr.utils.namespaces import LEGACY_TRAIN_LOSS_KEYS, LEGACY_TRAIN_LOSS_NAMES
from neosr.utils import set_random_seed, tc
from neosr.utils.dist_util import get_dist_info, init_dist, master_only

_WARNED_CONFIG_MESSAGES: set[str] = set()


def toml_load(f) -> dict[str, Any]:
    """Load TOML file
    Args:
        f (str): File path or a python string.

    Returns
    -------
        dict: Loaded dict.

    """
    try:
        with Path(f).open("rb") as f:
            return tomllib.load(f)
    except:
        msg = f"""
        {tc.red}
        Error decoding TOML file.
        If you are on Windows, make sure your paths uses single-quotes.
        {tc.end}
        """
        raise tomllib.TOMLDecodeError(msg)
        sys.exit(1)


def _warn_config_once(message: str) -> None:
    if message in _WARNED_CONFIG_MESSAGES:
        return
    _WARNED_CONFIG_MESSAGES.add(message)
    warnings.warn(message, UserWarning, stacklevel=3)


def _normalize_train_losses(opt: dict[str, Any]) -> None:
    train_opt = opt.get("train")
    if train_opt is None:
        return

    legacy_keys = [key for key in LEGACY_TRAIN_LOSS_KEYS if train_opt.get(key) is not None]
    has_new_losses = train_opt.get("losses") is not None

    if has_new_losses and legacy_keys:
        msg = (
            "Mixed legacy train loss keys and '[[train.losses]]' are not supported. "
            "Use exactly one configuration style."
        )
        raise ValueError(msg)

    if has_new_losses:
        losses = train_opt["losses"]
        if not isinstance(losses, list):
            msg = "The 'train.losses' field must be an array of tables."
            raise TypeError(msg)
    else:
        losses = []
        for key in LEGACY_TRAIN_LOSS_KEYS:
            legacy_opt = train_opt.pop(key, None)
            if legacy_opt is None:
                continue
            _warn_config_once(
                f"Legacy train loss key '{key}' is deprecated; migrate it to '[[train.losses]]'."
            )
            if not isinstance(legacy_opt, dict):
                msg = f"Legacy train loss '{key}' must be a table."
                raise TypeError(msg)
            loss_entry = deepcopy(legacy_opt)
            loss_entry.setdefault("name", LEGACY_TRAIN_LOSS_NAMES[key])
            losses.append(loss_entry)
        train_opt["losses"] = losses

    normalized_losses: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, loss_opt in enumerate(train_opt.get("losses", []), start=1):
        if not isinstance(loss_opt, dict):
            msg = f"train.losses entry #{index} must be a table."
            raise TypeError(msg)
        if loss_opt.get("name") is None:
            msg = f"train.losses entry #{index} is missing required field 'name'."
            raise ValueError(msg)
        name = loss_opt["name"]
        if not isinstance(name, str):
            msg = f"train.losses entry #{index} has a non-string 'name'."
            raise TypeError(msg)
        if name in seen_names:
            msg = f"Duplicate train loss name '{name}'."
            raise ValueError(msg)
        seen_names.add(name)
        normalized_losses.append(loss_opt)
    train_opt["losses"] = normalized_losses


def _normalize_val_metrics(opt: dict[str, Any]) -> None:
    val_opt = opt.get("val")
    if val_opt is None or val_opt.get("metrics") is None:
        return

    metrics = val_opt["metrics"]
    normalized_metrics: OrderedDict[str, dict[str, Any]] = OrderedDict()

    if isinstance(metrics, list):
        for index, metric_opt in enumerate(metrics, start=1):
            if not isinstance(metric_opt, dict):
                msg = f"val.metrics entry #{index} must be a table."
                raise TypeError(msg)
            name = metric_opt.get("name")
            if name is None:
                msg = f"val.metrics entry #{index} is missing required field 'name'."
                raise ValueError(msg)
            if not isinstance(name, str):
                msg = f"val.metrics entry #{index} has a non-string 'name'."
                raise TypeError(msg)
            if name in normalized_metrics:
                msg = f"Duplicate validation metric name '{name}'."
                raise ValueError(msg)
            metric_cfg = deepcopy(metric_opt)
            metric_cfg.pop("name", None)
            normalized_metrics[name] = metric_cfg
    elif isinstance(metrics, dict):
        _warn_config_once(
            "Legacy '[val.metrics.<name>]' tables are deprecated; migrate them to '[[val.metrics]]'."
        )
        for name, metric_opt in metrics.items():
            if not isinstance(metric_opt, dict):
                msg = f"Validation metric '{name}' must be a table."
                raise TypeError(msg)
            if name in normalized_metrics:
                msg = f"Duplicate validation metric name '{name}'."
                raise ValueError(msg)
            metric_cfg = deepcopy(metric_opt)
            metric_cfg.pop("name", None)
            normalized_metrics[name] = metric_cfg
    else:
        msg = "The 'val.metrics' field must be an array of tables or a metrics table."
        raise TypeError(msg)

    val_opt["metrics"] = normalized_metrics


def parse_options(
    root_path: PosixPath | str, is_train: bool = True
) -> tuple[dict[str, Any], argparse.Namespace]:
    parser = argparse.ArgumentParser(
        prog="neosr",
        usage=argparse.SUPPRESS,
        description="""-------- neosr command-line options --------""",
    )

    parser._optionals.title = "training and inference"

    parser.add_argument(
        "-opt", type=str, required=False, help="Path to option TOML file."
    )

    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm"],
        default="none",
        help="job launcher",
    )

    parser.add_argument("--auto_resume", action="store_true", default=False)

    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--local_rank", type=int, default=0)

    # Options for convert.py script

    group = parser.add_argument_group("model conversion")

    group.add_argument(
        "--input", type=str, required=False, help="Input Pytorch model path."
    )

    group.add_argument(
        "-onnx",
        "--onnx",
        action="store_true",
        help="Enables ONNX conversion.",
        default=False,
    )

    group.add_argument(
        "-prune",
        "--prune",
        action="store_true",
        help="Enables network prune if supported.",
        default=False,
    )

    group.add_argument(
        "-net", "--network", type=str, required=False, help="Generator network."
    )

    group.add_argument("-s", "--scale", type=int, help="Model scale ratio.", default=4)

    group.add_argument(
        "-window", "--window", type=int, help="Model scale ratio.", default=None
    )

    group.add_argument(
        "-opset", "--opset", type=int, help="ONNX opset. (default: 17)", default=17
    )

    group.add_argument(
        "-static",
        "--static",
        type=int,
        nargs=3,
        help='Set static shape for ONNX conversion. Example: -static "3,640,640".',
        default=None,
    )

    group.add_argument(
        "-nocheck",
        "--nocheck",
        action="store_true",
        help="Disables checking against original pytorch model on ONNX conversion.",
        default=False,
    )

    group.add_argument(
        "-fp16",
        "--fp16",
        action="store_true",
        help="Enable half-precision. (default: false)",
        default=False,
    )

    group.add_argument(
        "-optimize",
        "--optimize",
        action="store_true",
        help="Run ONNX optimizations",
        default=False,
    )

    group.add_argument(
        "-fulloptimization",
        "--fulloptimization",
        action="store_true",
        help="Run full ONNX optimizations",
        default=False,
    )

    group.add_argument(
        "--output",
        type=str,
        required=False,
        help="Output ONNX model path.",
        default=root_path,
    )

    args = parser.parse_args()

    # error if no config file exists
    if args.input is None and args.opt is None:
        msg = f"""
        {tc.red}
        Didn't get a config! Please link the config file using -opt /path/to/config.toml
        {tc.end}
        """
        raise ValueError(msg)

    if args.input is None:
        # error if not toml
        if not args.opt.endswith(".toml"):
            msg = f"""
            {tc.light_blue}
            neosr has switched to TOML configuration files!
            Please see templates on the options/ folder.
            {tc.end}
            """
            raise ValueError(msg)

        # parse toml to dict
        opt = toml_load(args.opt)

        # distributed settings
        if args.launcher == "none":
            opt["dist"] = False
        else:
            opt["dist"] = True
            if args.launcher == "slurm" and "dist_params" in opt:
                init_dist(args.launcher, **opt["dist_params"])
            else:
                init_dist(args.launcher)
        opt["rank"], opt["world_size"] = get_dist_info()

        # random seed
        seed = opt.get("manual_seed")
        if seed is None:
            opt["deterministic"] = False
            seed = random.randint(1024, 10000)
            opt["manual_seed"] = seed
            torch.utils.deterministic.fill_uninitialized_memory = False
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.benchmark_limit = 0
            os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        else:
            # Determinism
            opt["deterministic"] = True
            os.environ["PYTHONHASHSEED"] = str(seed)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(mode=True, warn_only=True)
            torch.utils.deterministic.fill_uninitialized_memory = True
        set_random_seed(seed + opt["rank"])

        opt["auto_resume"] = args.auto_resume
        opt["is_train"] = is_train

        # debug setting
        if args.debug and not opt["name"].startswith("debug"):
            opt["name"] = "debug_" + opt["name"]

        _normalize_train_losses(opt)
        _normalize_val_metrics(opt)

        if opt.get("num_gpu", "auto") == "auto":
            opt["num_gpu"] = torch.cuda.device_count()

        # datasets
        for phase, dataset in opt["datasets"].items():
            # for multiple datasets, e.g., val_1, val_2; test_1, test_2
            phase_ = phase.split("_")[0]
            dataset["phase"] = phase_
            if "scale" in opt:
                dataset["scale"] = opt["scale"]
            if dataset.get("dataroot_gt") is not None:
                dataset["dataroot_gt"] = str(Path(dataset["dataroot_gt"]).expanduser())
            if dataset.get("dataroot_lq") is not None:
                dataset["dataroot_lq"] = str(Path(dataset["dataroot_lq"]).expanduser())

        # paths
        if opt.get("path") is not None:
            for key, val in opt["path"].items():
                if (val is not None) and (
                    "resume_state" in key or "pretrain_network" in key
                ):
                    opt["path"][key] = str(Path(val).expanduser())

        if is_train:
            experiments_root = opt.get("path")
            if experiments_root is not None:
                experiments_root = experiments_root.get("experiments_root")
            if experiments_root is None:
                experiments_root = Path(root_path) / "experiments"
            experiments_root = Path(experiments_root) / opt["name"]

            if opt.get("path") is None:
                opt["path"] = {}

            opt["path"]["experiments_root"] = experiments_root
            opt["path"]["models"] = Path(experiments_root) / "models"
            opt["path"]["training_states"] = Path(experiments_root) / "training_states"
            opt["path"]["log"] = experiments_root
            opt["path"]["visualization"] = Path(experiments_root) / "visualization"

            # change some options for debug mode
            if "debug" in opt["name"]:
                if "val" in opt:
                    opt["val"]["val_freq"] = 8
                opt["logger"]["print_freq"] = 1
                opt["logger"]["save_checkpoint_freq"] = 8
        else:  # test
            results_root = opt["path"].get("results_root")
            if results_root is None:
                results_root = Path(root_path) / "experiments" / "results"
            results_root = Path(results_root) / opt["name"]

            opt["path"]["results_root"] = results_root
            opt["path"]["log"] = results_root
            opt["path"]["visualization"] = results_root
    else:
        opt = {}

    return opt, args


@master_only
def copy_opt_file(opt_file: str, experiments_root: str) -> None:
    # copy the toml file to the experiment root
    cmd = " ".join(sys.argv)
    original_path = Path(opt_file)
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    new_filename = f"{original_path.stem}_{time_str}{original_path.suffix}"
    filename = Path(experiments_root) / new_filename
    copyfile(opt_file, filename)

    with Path(filename).open("r+", encoding="utf-8") as f:
        lines = f.readlines()
        lines.insert(0, f"# GENERATE TIME: {time.asctime()}\n# CMD:\n# {cmd}\n\n")
        f.seek(0)
        f.writelines(lines)
