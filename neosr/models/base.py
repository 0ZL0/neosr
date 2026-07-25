import gc
import os
import sys
import time
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor, nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim.optimizer import Optimizer

from neosr.optimizers import adamw_sf, adamw_win, adan, adan_sf, soap_sf
from neosr.utils import get_root_logger, tc
from neosr.utils.dist_util import get_dist_info, master_only
from neosr.utils.rng import capture_rng_state, restore_rng_state

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class base:
    """Default model."""

    def __init__(self, opt: dict[str, Any]) -> None:
        self.opt = opt
        self.device = torch.device("cuda")
        self.is_train = opt["is_train"]
        self.optimizers: list[Any] = []
        self.schedulers: list[Any] = []
        self.optimizer_g: Optimizer
        self.optimizer_d: Optimizer
        self.log_dict: dict[str, Any]
        self.n_accumulated: int
        self.optimizer_step_succeeded: list[bool] | None = None
        self._print_different_keys_loading: Callable

        if self.is_train:
            # Schedule-Free Generator
            self.sf_optim_g = opt["train"]["optim_g"].get("schedule_free", False)
            # Schedule-Free Discriminator
            self.net_d = opt.get("network_d")
            if self.net_d is not None:
                self.sf_optim_d = opt["train"]["optim_d"].get("schedule_free", False)
        else:
            self.sf_optim_g = None
            self.sf_optim_d = None

    def feed_data(self, data: dict[str, str | Tensor]) -> None:
        pass

    def optimize_parameters(self, current_iter: int) -> bool:
        del current_iter
        return False

    def get_current_visuals(self) -> OrderedDict | None:
        pass

    def save(
        self,
        epoch: int,
        current_iter: int,
        training_progress: Mapping[str, object] | None = None,
    ) -> None:
        del epoch, current_iter, training_progress

    def get_training_state(self) -> dict[str, Any]:
        """Return model-specific mutable state not owned by an optimizer."""
        return {}

    def load_training_state(self, state: Mapping[str, Any]) -> None:
        """Restore state returned by :meth:`get_training_state`."""
        del state

    def dist_validation(
        self, dataloader, current_iter: int, tb_logger, save_img: bool = True
    ) -> None:
        pass

    def nondist_validation(
        self, dataloader, current_iter: int, tb_logger, save_img: bool = True
    ) -> None:
        pass

    def validation(
        self, dataloader, current_iter: int, tb_logger, save_img: bool = True
    ) -> None:
        """Validation function.

        Args:
        ----
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.

        """
        if self.opt["dist"]:
            self.dist_validation(dataloader, current_iter, tb_logger, save_img)
        else:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def _initialize_best_metric_results(self, dataset_name: str) -> None:
        """Initialize the best metric results dict for recording the best metric value and iteration."""
        self.best_metric_results: dict[Any, Any]
        if (
            hasattr(self, "best_metric_results")
            and dataset_name in self.best_metric_results
        ):
            return
        if not hasattr(self, "best_metric_results"):
            self.best_metric_results = {}

        # add a dataset record
        record = {}
        for metric, content in self.opt["val"]["metrics"].items():
            better = content.get("better", "higher")
            init_val = float("-inf") if better == "higher" else float("inf")
            record[metric] = {"better": better, "val": init_val, "iter": -1}
        self.best_metric_results[dataset_name] = record

    def _update_best_metric_result(
        self, dataset_name, metric: str, val, current_iter: int
    ) -> None:
        if self.best_metric_results[dataset_name][metric]["better"] == "higher":
            if val >= self.best_metric_results[dataset_name][metric]["val"]:
                self.best_metric_results[dataset_name][metric]["val"] = val
                self.best_metric_results[dataset_name][metric]["iter"] = current_iter
        elif val <= self.best_metric_results[dataset_name][metric]["val"]:
            self.best_metric_results[dataset_name][metric]["val"] = val
            self.best_metric_results[dataset_name][metric]["iter"] = current_iter

    def get_current_log(self) -> dict[str, Any]:
        return self.log_dict

    def model_to_device(self, net: nn.Module) -> nn.Module:
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
        ----
            net (nn.Module)

        """
        if self.opt.get("use_amp", False) is True:
            net = net.to(  # type: ignore[call-overload]
                self.device, non_blocking=True, memory_format=torch.channels_last
            )
        else:
            net = net.to(self.device, non_blocking=True)  # type: ignore[attr-defined]

        if self.opt.get("compile", False) is True:
            net = torch.compile(net, mode="max-autotune", dynamic=True)  # type: ignore[assignment]
            # see option fullgraph=True

        if self.opt["dist"]:
            find_unused_parameters = self.opt.get("find_unused_parameters", False)
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=find_unused_parameters,
            )
        elif self.opt["num_gpu"] > 1:
            net = DataParallel(net)  # type: ignore[type-var]
        return net

    def get_optimizer(self, optim_type: str, params, lr: float, **kwargs) -> Optimizer:
        if optim_type in {"Adam", "adam"}:
            optimizer: Optimizer = torch.optim.Adam(params, lr, **kwargs)  # type: ignore[reportPrivateImportUsage]
        elif optim_type in {"AdamW", "adamw"}:
            optimizer = torch.optim.AdamW(params, lr, **kwargs)  # type: ignore[reportPrivateImportUsage]
        elif optim_type in {"NAdam", "nadam"}:
            optimizer = torch.optim.NAdam(params, lr, **kwargs)  # type: ignore[reportPrivateImportUsage]
        elif optim_type in {"Adan", "adan"}:
            optimizer = adan(params, lr, **kwargs)
        elif optim_type in {"AdamW_Win", "adamw_win"}:
            optimizer = adamw_win(params, lr, **kwargs)
        elif optim_type in {"AdamW_SF", "adamw_sf"}:
            optimizer = adamw_sf(params, lr, **kwargs)
        elif optim_type in {"Adan_SF", "adan_sf"}:
            optimizer = adan_sf(params, lr, **kwargs)
        elif optim_type in {"SOAP_SF", "soap_sf"}:
            optimizer = soap_sf(params, lr, **kwargs)
        else:
            logger = get_root_logger()
            msg = f"{tc.red}Optimizer {optim_type} is not supported yet.{tc.end}"
            logger.error(msg)
            sys.exit(1)

        return cast("Optimizer", optimizer)

    def setup_schedulers(self) -> None:
        """Set up schedulers."""
        train_opt = self.opt["train"]
        has_scheduler = self.opt["train"].get("scheduler", None)
        if has_scheduler is not None:
            scheduler_type = train_opt["scheduler"].pop("type")
            if scheduler_type in {"MultiStepLR", "multisteplr"}:
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.MultiStepLR(
                            optimizer, **train_opt["scheduler"]
                        )
                    )
            elif scheduler_type in {"CosineAnnealing", "cosineannealing"}:
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, **train_opt["scheduler"]
                        )
                    )
            else:
                logger = get_root_logger()
                msg = f"{tc.red}Scheduler {scheduler_type} is not implemented yet.{tc.end}"
                logger.error(msg)
                sys.exit(1)

    def get_bare_model(self, net: nn.Module) -> nn.Module:
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, DataParallel | DistributedDataParallel):
            net = net.module
        return net

    def _set_lr(
        self, lr_groups_l, optimizer_step_succeeded: list[bool] | None = None
    ) -> None:
        """Set learning rate for warm-up.

        Args:
        ----
            lr_groups_l (list): List for lr_groups, each for an optimizer.

        """
        if optimizer_step_succeeded is None:
            optimizer_step_succeeded = [True] * len(self.optimizers)
        for optimizer, lr_groups, step_succeeded in zip(
            self.optimizers, lr_groups_l, optimizer_step_succeeded, strict=True
        ):
            if not step_succeeded:
                continue
            for param_group, lr in zip(optimizer.param_groups, lr_groups, strict=True):
                param_group["lr"] = lr

    def _get_init_lr(self) -> list[list[Any]]:
        """Get the initial lr, which is set by the scheduler."""
        init_lr_groups_l = []
        init_lr_groups_l.extend([
            [v["initial_lr"] for v in optimizer.param_groups]
            for optimizer in self.optimizers
        ])
        return init_lr_groups_l

    def update_learning_rate(self, current_iter: int, warmup_iter: int = -1) -> None:
        """Update learning rate.

        Args:
        ----
            current_iter (int): Current iteration.
            warmup_iter (int): Warm-up iter numbers. -1 for no warm-up.
                Default: -1.

        """
        optimizer_step_succeeded = getattr(self, "optimizer_step_succeeded", None)
        if optimizer_step_succeeded is None:
            optimizer_step_succeeded = [True] * len(self.optimizers)
        if len(optimizer_step_succeeded) != len(self.optimizers):
            msg = "Optimizer step results do not match the configured optimizers."
            raise RuntimeError(msg)
        if self.schedulers and len(self.schedulers) != len(self.optimizers):
            msg = "Schedulers and optimizers must have matching lengths."
            raise RuntimeError(msg)
        if current_iter > 0 and self.schedulers:
            for scheduler, step_succeeded in zip(
                self.schedulers, optimizer_step_succeeded, strict=True
            ):
                if step_succeeded:
                    scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            warm_up_lr_l.extend([
                [v / warmup_iter * current_iter for v in init_lr_g]
                for init_lr_g in init_lr_g_l
            ])
            # set learning rate
            self._set_lr(warm_up_lr_l, optimizer_step_succeeded)

    def get_current_learning_rate(self):
        return [param_group["lr"] for param_group in self.optimizers[0].param_groups]

    @master_only
    def print_network(self, net: nn.Module) -> None:
        """Print the str and parameter number of a network.

        Args:
        ----
            net (nn.Module)

        """
        if isinstance(net, DataParallel | DistributedDataParallel):
            net_cls_str = f"{net.__class__.__name__} - {net.module.__class__.__name__}"
        else:
            net_cls_str = f"{net.__class__.__name__}"

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(x.numel() for x in net.parameters())

        logger = get_root_logger()
        logger.info(f"Network: {net_cls_str}, with parameters: {net_params:,d}")
        logger.info(net_str)

    @contextmanager
    def evaluation_weights(self) -> "Iterator[None]":
        """Hold schedule-free optimizers at the iterate that has to be read.

        Schedule-free methods step on the interpolated ``y`` iterate but must be
        evaluated and serialized at ``x``. ``sf_optim_*`` is None outside
        training, so no additional flag is needed here -- and in particular no
        flag that validation mutates may be consulted.
        """
        optimizers = []
        if self.sf_optim_g:
            optimizers.append(self.optimizer_g)
        if self.net_d is not None and self.sf_optim_d:
            optimizers.append(self.optimizer_d)

        for optimizer in optimizers:
            optimizer.eval()  # type: ignore[attr-defined]
        try:
            yield
        finally:
            for optimizer in optimizers:
                optimizer.train()  # type: ignore[attr-defined]

    @staticmethod
    def _save_with_retry(obj: Any, path: Path, what: str, attempts: int = 3) -> None:
        """Serialize ``obj`` to ``path`` atomically, retrying transient OS errors.

        The payload goes to a sibling temporary file that is renamed into place,
        so an interrupted or failing write cannot leave a half-written
        checkpoint behind -- auto-resume always picks the highest iteration it
        finds and would otherwise choke on it.
        """
        logger = get_root_logger()
        tmp = path.with_name(f"{path.name}.tmp")
        for attempt in range(1, attempts + 1):
            try:
                with tmp.open("wb") as f:
                    torch.save(obj, f)
                    f.flush()
                    os.fsync(f.fileno())
                tmp.replace(path)
            except OSError:
                tmp.unlink(missing_ok=True)
                logger.warning(
                    f"{tc.red}Save {what} error. "
                    f"Remaining retry times: {attempts - attempt}{tc.end}"
                )
                if attempt < attempts:
                    time.sleep(1)
            else:
                return

        msg = f"{tc.red}Cannot save {path}, aborting.{tc.end}"
        logger.error(msg)
        sys.exit(1)

    @master_only
    def save_network(
        self,
        net: nn.Module | list[nn.Module],
        net_label: list[nn.Module],
        current_iter: int | str,
        param_key: str | list[str] = "params",
    ) -> None:
        """Save networks.

        Args:
        ----
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.

        """
        if current_iter == -1:
            current_iter = "latest"
        save_filename = f"{net_label}_{current_iter}.pth"
        save_path = Path(self.opt["path"]["models"]) / save_filename

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), (
            "The lengths of net and param_key should be the same."
        )

        # The copy has to happen inside the context: it is what gets written, so
        # bracketing only the write would export the training iterate instead.
        save_dict = {}
        with self.evaluation_weights():
            for net_, param_key_ in zip(net, param_key, strict=True):
                net__ = self.get_bare_model(net_)
                new_state_dict = OrderedDict()

                for key, param in net__.state_dict().items():
                    key = key.removeprefix("module.")  # noqa: PLW2901
                    if key.startswith("n_averaged"):  # skip n_averaged from EMA
                        continue
                    new_state_dict[key] = param.cpu()
                save_dict[param_key_] = new_state_dict

        self._save_with_retry(save_dict, save_path, "model")

    @staticmethod
    def _reject_legacy_spectral_norm(
        net: nn.Module, load_net: Mapping[str, Any], load_path: str
    ) -> None:
        """Refuse discriminators saved with the pre-parametrization spectral norm.

        Renaming ``weight_orig``/``weight_u``/``weight_v`` to their parametrization
        equivalents makes such a checkpoint load, but does not make it usable: it
        was trained while the power iteration never ran, so its weights are
        calibrated against a random spectral-norm estimate. Once the estimate is
        computed correctly the network it encodes is gone -- measured on a
        discriminator trained that way, the decision gap collapses ~170x and
        flips sign within three iterations, and no rescaling of the stored weights
        recovers it because ``W / sigma(W)`` is scale-invariant in ``W``. A silent
        migration would therefore hide a full reset behind a successful load.
        """
        legacy = [k for k in load_net if k.endswith("weight_orig")]
        if not legacy or not any(
            k.endswith("parametrizations.weight.original") for k in net.state_dict()
        ):
            return

        msg = f"""
              {tc.red}
              {load_path}
              was saved with the legacy spectral-norm implementation ({len(legacy)} layers).
              It cannot be resumed: those weights were trained while the power
              iteration was never updated, so they are calibrated against a random
              spectral-norm estimate and do not survive a correct one. Renaming the
              keys, or loading non-strictly, would reset the discriminator silently.

              Resume the generator and let the discriminator restart instead:
                  [path]
                  ignore_resume_networks = [ "network_d" ]
              {tc.end}
              """
        get_root_logger().error(msg)
        raise ValueError(msg)

    def load_network(
        self, net: nn.Module, load_path: str, param_key: str | None, strict: bool = True
    ) -> str | None:
        """Load network.

        Args:
        ----
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: None.

        """
        logger = get_root_logger()
        net = self.get_bare_model(net)

        # error if path not found
        if not Path(load_path).exists():
            msg = f"{tc.red}Path to model doesn't exist. Please check your configuration.{tc.end}"
            logger.error(msg)
            sys.exit(1)

        # Paired raw+EMA checkpoints contain two full parameter sets. Stage them
        # on CPU so resumption does not transiently require several model copies
        # in VRAM.
        load_net = torch.load(load_path, map_location="cpu", weights_only=True)

        if isinstance(load_net, Mapping):
            available_param_keys = [
                key for key in ("params-ema", "params_ema", "params") if key in load_net
            ]
            selected_param_key = param_key
            if selected_param_key == "params_ema" and "params-ema" in load_net:
                selected_param_key = "params-ema"
            elif selected_param_key == "params-ema" and "params_ema" in load_net:
                selected_param_key = "params_ema"
            if selected_param_key not in load_net:
                if selected_param_key is not None and available_param_keys:
                    logger.warning(
                        f"Param key [{selected_param_key}] is unavailable in {load_path}; "
                        f"falling back to [{available_param_keys[0]}]."
                    )
                selected_param_key = (
                    available_param_keys[0] if available_param_keys else None
                )
            if selected_param_key is not None:
                load_net = load_net[selected_param_key]
            param_key = selected_param_key

        if param_key:
            logger.info(
                f"Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}]."
            )
        else:
            logger.info(f"Loading {net.__class__.__name__} model from {load_path}.")

        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith("module."):
                load_net[k[7:]] = v
                load_net.pop(k)

        self._reject_legacy_spectral_norm(net, load_net, load_path)
        self._print_different_keys_loading(net, load_net, strict)

        try:
            net.load_state_dict(load_net, strict=strict)
        except Exception:
            msg = f"{tc.red}Failed to load model. Please check scale and net parameters, or disable strict_load.{tc.end}"
            logger.exception(msg)
            sys.exit(1)

        torch.cuda.empty_cache()
        gc.collect()
        return param_key

    def save_training_state(
        self,
        epoch: int,
        current_iter: int,
        training_progress: Mapping[str, object] | None = None,
    ) -> None:
        """Save training states during training, which will be used for
        resuming.

        Args:
        ----
            epoch (int): Current epoch.
            current_iter (int): Current iteration.

        """
        if current_iter != -1:
            rank, world_size = get_dist_info()
            rank_state = {
                "rng_state": capture_rng_state(),
                "model_training_state": self.get_training_state(),
            }
            distributed = world_size > 1
            if distributed:
                rank_state_dir = (
                    Path(self.opt["path"]["training_states"]) / "rank_states"
                )
                rank_state_dir.mkdir(parents=True, exist_ok=True)
                rank_state_path = (
                    rank_state_dir / f"{int(current_iter)}.rank{rank}.state"
                )
                self._save_with_retry(rank_state, rank_state_path, "rank state")
                torch.distributed.barrier()
            if rank != 0:
                torch.distributed.barrier()
                return

            state: dict[str, Any] = {
                "epoch": epoch,
                "iter": current_iter,
                "optimizers": [],
                "schedulers": [],
                "accumulation_steps": getattr(self, "accum_iters", 1),
            }
            if training_progress is not None:
                state["training_progress"] = dict(training_progress)
            if distributed:
                state["distributed_rank_states"] = True
            else:
                state["rank_state"] = rank_state
            # Captured inside the context so the recorded train_mode matches the
            # iterate save_network wrote; resume_training swaps both back.
            with self.evaluation_weights():
                for o in self.optimizers:
                    state["optimizers"].append(o.state_dict())  # type: ignore[attr-defined]
            for s in self.schedulers:
                state["schedulers"].append(s.state_dict())  # type: ignore[attr-defined]
            grad_scalers = {}
            for name in ("gradscaler_g", "gradscaler_d"):
                scaler = getattr(self, name, None)
                if scaler is not None and (scaler_state := scaler.state_dict()):
                    grad_scalers[name] = scaler_state
            if grad_scalers:
                state["grad_scalers"] = grad_scalers
            save_filename = f"{int(current_iter)}.state"
            save_path = Path(self.opt["path"]["training_states"]) / save_filename

            self._save_with_retry(state, save_path, "training state")

            if distributed:
                torch.distributed.barrier()

        torch.cuda.empty_cache()
        gc.collect()

    def resume_training(self, resume_state: dict[Any, Any]) -> None:
        """Reload the optimizers and schedulers for resumed training.

        Args:
        ----
            resume_state (dict): Resume state.

        """
        resume_optimizers = resume_state["optimizers"]
        resume_schedulers = resume_state["schedulers"]
        assert len(resume_optimizers) == len(self.optimizers), (
            "Wrong lengths of optimizers"
        )
        assert len(resume_schedulers) == len(self.schedulers), (
            "Wrong lengths of schedulers"
        )
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

        # Checkpoints store the evaluation iterate; step() refuses to run until
        # the schedule-free optimizers have swapped back to the training one.
        if self.sf_optim_g:
            self.optimizer_g.train()  # type: ignore[attr-defined]
        if self.net_d is not None and self.sf_optim_d:
            self.optimizer_d.train()  # type: ignore[attr-defined]

        resume_grad_scalers = resume_state.get("grad_scalers", {})
        for name, scaler_state in resume_grad_scalers.items():
            scaler = getattr(self, name, None)
            if scaler is not None:
                scaler.load_state_dict(scaler_state)
        if self.opt.get("use_amp", False) and not resume_grad_scalers:
            logger = get_root_logger()
            logger.warning(
                "The checkpoint has no AMP GradScaler state; resuming with a fresh scale."
            )

        rank_state = resume_state.get("rank_state")
        if rank_state is not None:
            self.load_training_state(rank_state.get("model_training_state", {}))
            restore_rng_state(rank_state["rng_state"])
        else:
            logger = get_root_logger()
            logger.warning(
                "The checkpoint has no per-rank model/RNG state; exact resumption "
                "is unavailable for this legacy checkpoint."
            )

        torch.cuda.empty_cache()
        gc.collect()

    def reduce_loss_dict(self, loss_dict: dict[Any, Any]) -> OrderedDict:
        """Reduce loss dict.

        In distributed training, it averages the losses among different GPUs .

        Args:
        ----
            loss_dict (OrderedDict): Loss dict.

        """
        with torch.inference_mode():
            if self.opt["dist"]:
                keys = tuple(loss_dict)
                losses = torch.stack([
                    loss_dict[name].detach().mean() for name in keys
                ])
                torch.distributed.all_reduce(losses)
                losses /= torch.distributed.get_world_size()
                loss_dict = dict(zip(keys, losses, strict=True))

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict
