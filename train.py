import datetime
import logging
import re
import sys
import time
from os import path as osp
from os import popen
from pathlib import Path
from typing import Any

import torch
from torch.utils import data

from neosr.data import build_dataloader, build_dataset
from neosr.data.data_sampler import EnlargedSampler
from neosr.data.prefetch_dataloader import CUDAPrefetcher
from neosr.models import build_model
from neosr.models.training_utils import (
    AccumulationPlan,
    ResumePosition,
    normalize_accumulation_steps,
    resume_position,
)
from neosr.utils import (
    AvgTimer,
    MessageLogger,
    check_disk_space,
    check_resume,
    get_root_logger,
    get_time_str,
    init_tb_logger,
    init_wandb_logger,
    make_exp_dirs,
    mkdir_and_rename,
    scandir,
    set_random_seed,
    tc,
)
from neosr.utils.options import copy_opt_file, parse_options
from neosr.utils.rng import preserve_rng_state
from neosr.utils.validation import resolve_validation_save_img

# supported Python versions
if not (3, 11) <= sys.version_info < (3, 14):
    msg = f"{tc.red}Python 3.11-3.13 is required.{tc.end}"
    raise ValueError(msg)


def init_tb_loggers(opt: dict[str, Any]):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (
        (opt["logger"].get("wandb") is not None)
        and (opt["logger"]["wandb"].get("project") is not None)
        and ("debug" not in opt["name"])
    ):
        assert opt["logger"].get("use_tb_logger") is True, (
            "should turn on tensorboard when using wandb"
        )
        init_wandb_logger(opt)
    tb_logger = None
    if opt["logger"].get("use_tb_logger") and "debug" not in opt["name"]:
        tb_logger = init_tb_logger(
            log_dir=Path(opt["root_path"]) / "experiments" / "tb_logger" / opt["name"]
        )
    return tb_logger


def create_train_val_dataloader(
    opt: dict[str, Any], logger: logging.Logger
) -> tuple[data.DataLoader, EnlargedSampler, list[data.DataLoader], AccumulationPlan]:
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    train_sampler: EnlargedSampler | None = None
    accumulation_plan: AccumulationPlan | None = None

    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            dataset_enlarge_ratio = dataset_opt.get("dataset_enlarge_ratio", 1)
            # add degradations section to dataset_opt
            if opt.get("degradations") is not None:
                dataset_opt.update(opt["degradations"])
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(
                train_set,
                opt["world_size"],
                opt["rank"],
                dataset_enlarge_ratio,
                seed=opt["manual_seed"],
            )
            num_gpu = opt.get("num_gpu", "auto")
            train_loader = build_dataloader(
                train_set,  # type: ignore[reportArgumentType]
                dataset_opt,
                num_gpu=num_gpu,
                dist=opt["dist"],
                sampler=train_sampler,
                seed=opt["manual_seed"],
            )

            accumulation_steps = normalize_accumulation_steps(
                dataset_opt.get("accumulate", 1)
            )
            accumulation_plan = AccumulationPlan(
                total_optimizer_steps=int(opt["logger"].get("total_iter", 1_000_000)),
                accumulation_steps=accumulation_steps,
                micro_batches_per_epoch=len(train_loader),
            )
            process_batch_size = int(
                train_loader.batch_size or dataset_opt["batch_size"]
            )
            effective_batch_size = (
                process_batch_size * opt["world_size"] * accumulation_steps
            )
            logger.info(
                "Training informations:"
                f"\n-------- Starting model: {opt['name']}"
                f"\n-------- GPUs detected: {opt['world_size']}"
                f"\n-------- Patch size: {dataset_opt['patch_size']}"
                f"\n-------- Dataset size: {len(train_set)}"  # type: ignore[reportArgumentType]
                f"\n-------- Batch size per gpu: {dataset_opt['batch_size']}"
                f"\n-------- Gradient accumulation steps: {accumulation_steps}"
                f"\n-------- Effective global batch size: {effective_batch_size}"
                f"\n-------- Micro-batches per epoch: {accumulation_plan.micro_batches_per_epoch}"
                f"\n-------- Total micro-batches: {accumulation_plan.total_micro_batches}"
                f"\n-------- Total epochs: {accumulation_plan.total_epochs}"
                f"\n-------- Total optimizer steps: {accumulation_plan.total_optimizer_steps}."
            )
        elif phase.split("_")[0] == "val":
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set,  # type: ignore[reportArgumentType]
                dataset_opt,
                num_gpu=opt["num_gpu"],
                dist=opt["dist"],
                sampler=None,
                seed=opt["manual_seed"],
            )
            logger.info(f"Number of val images/folders: {len(val_set)}")  # type: ignore[reportArgumentType]
            val_loaders.append(val_loader)
        else:
            msg = f"{tc.red}Dataset phase {phase} is not recognized.{tc.end}"
            logger.error(msg)
            sys.exit(1)

    if train_loader is None or train_sampler is None or accumulation_plan is None:
        msg = "A training dataset must be configured under datasets.train."
        raise ValueError(msg)
    return train_loader, train_sampler, val_loaders, accumulation_plan


def load_resume_state(opt: dict[str, Any]):
    resume_state_path = None
    if opt["auto_resume"]:
        state_path = opt["path"]["training_states"]
        if Path.is_dir(state_path):
            states = list(
                scandir(state_path, suffix="state", recursive=False, full_path=False)
            )
            if len(states) != 0:
                states = [float(v.split(".state")[0]) for v in states]
                resume_state_path = Path(state_path) / f"{max(states):.0f}.state"
                opt["path"]["resume_state"] = resume_state_path

    elif opt["path"].get("resume_state"):
        resume_state_path = opt["path"]["resume_state"]

    if resume_state_path is None:
        resume_state = None
    else:
        resume_state = torch.load(
            resume_state_path, map_location="cpu", weights_only=True
        )
        training_progress = resume_state.get("training_progress", {})
        saved_sampler_seed = (
            training_progress.get("sampler_seed")
            if isinstance(training_progress, dict)
            else None
        )
        if (
            not opt["deterministic"]
            and not isinstance(saved_sampler_seed, bool)
            and isinstance(saved_sampler_seed, int)
        ):
            # A configuration without manual_seed gets a fresh seed on every
            # launch. Reuse the checkpoint seed before constructing the sampler.
            opt["manual_seed"] = saved_sampler_seed
            set_random_seed(saved_sampler_seed + opt["rank"])
        if resume_state.get("distributed_rank_states", False):
            rank_state_path = (
                Path(resume_state_path).parent
                / "rank_states"
                / f"{int(resume_state['iter'])}.rank{opt['rank']}.state"
            )
            if not rank_state_path.exists():
                msg = f"Missing per-rank resume state: {rank_state_path}"
                raise FileNotFoundError(msg)
            resume_state["rank_state"] = torch.load(
                rank_state_path, map_location="cpu", weights_only=True
            )
        check_resume(opt, resume_state["iter"])
    return resume_state


def train_pipeline(root_path: str) -> None:
    # raise error if pytorch <2.4
    if int(torch.__version__.split(".")[1]) < 4:
        msg = f"{tc.red}Pytorch >=2.4 is required, please upgrade.{tc.end}"
        raise NotImplementedError(msg)

    # raise error if not CUDA
    if not torch.cuda.is_available():
        msg = f"{tc.red}CUDA not available. Please install pytorch with cuda support.{tc.end}"
        raise NotImplementedError(msg)

    # check if system cuda version is not lower than pytorch target
    try:
        nvcc_cmd = "nvcc --version"
        nvcc_cuda = re.search(r"release (\d+\.\d+)", popen(nvcc_cmd).read())[1]  # noqa: S605
        torch_cuda = torch.version.cuda
        if tuple(map(int, torch_cuda.split("."))) > tuple(
            map(int, nvcc_cuda.split("."))
        ):
            msg = f"{tc.red}Your system CUDA version appears to be {nvcc_cuda} while pytorch is higher ({torch_cuda})!{tc.end}"
            raise RuntimeError(msg)
    except:
        pass

    # default device
    torch.set_default_device("cuda")

    # parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=True)
    opt["root_path"] = root_path
    opt["datasets"]["train"]["accumulate"] = normalize_accumulation_steps(
        opt["datasets"]["train"].get("accumulate", 1)
    )
    if opt["dist"] and not opt["deterministic"]:
        # Keep an automatically generated sampler seed common across ranks.
        shared_seed = torch.tensor(opt["manual_seed"], device="cuda", dtype=torch.int64)
        torch.distributed.broadcast(shared_seed, src=0)
        opt["manual_seed"] = int(shared_seed.item())
        set_random_seed(opt["manual_seed"] + opt["rank"])

    # Triton doesn't support Windows yet
    if sys.platform.startswith("win") and opt.get("compile", False) is True:
        msg = f"{tc.red}Compile is not supported on Windows, please disable it on your configuration file.{tc.end}"
        raise NotImplementedError(msg)

    # enable tensorfloat32 and possibly bfloat16 matmul
    fast_matmul = opt.get("fast_matmul", False)
    if fast_matmul:
        torch.set_float32_matmul_precision("medium")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_accumulation = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if (
            opt["logger"].get("use_tb_logger")
            and "debug" not in opt["name"]
            and opt["rank"] == 0
        ):
            mkdir_and_rename(
                Path(opt["root_path"]) / "experiments" / "tb_logger" / opt["name"]
            )

    # copy the toml file to the experiment root
    try:
        copy_opt_file(args.opt, opt["path"]["experiments_root"])
    except:
        msg = f"{tc.red}Failed. Make sure the option 'name' in your config file is the same as the previous state!{tc.end}"
        raise ValueError(msg)

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = Path(opt["path"]["log"]) / f"train_{opt['name']}_{get_time_str()}.log"
    logger = get_root_logger(
        logger_name="neosr", log_level=logging.INFO, log_file=str(log_file)
    )

    smi_cmd = "nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits"
    driver_version = (
        popen(smi_cmd)  # noqa: S605
        .read()
        .strip()
    )

    logger.info(
        f"\n------------------------ neosr ------------------------\nPytorch Version: {torch.__version__}. Running on gpu {torch.cuda.get_device_name()}, with driver {driver_version}."
    )

    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, accumulation_plan = result
    process_batch_size = int(train_loader.batch_size or 1)

    # create model
    model = build_model(opt)

    if resume_state:  # resume training
        # handle optimizers and schedulers
        model.resume_training(resume_state)  # type: ignore[reportAttributeAccessIssue,attr-defined]
        logger.info(
            f"{tc.light_green}Resuming training from epoch: {resume_state['epoch']}, iter: {int(resume_state['iter'])}{tc.end}"
        )
        position = resume_position(resume_state, accumulation_plan, train_sampler.seed)
        if "training_progress" not in resume_state:
            logger.warning(
                "Legacy checkpoint has no explicit data cursor; reconstructing it "
                "from optimizer steps and accumulation metadata."
            )
        torch.cuda.empty_cache()
    else:
        position = resume_position(None, accumulation_plan, train_sampler.seed)

    start_epoch = position.epoch
    current_optimizer_step = position.optimizer_step
    current_micro_batch = position.global_micro_batch
    batch_in_epoch = position.batch_in_epoch

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, tb_logger, current_optimizer_step)

    # dataloader prefetcher
    prefetcher = CUDAPrefetcher(train_loader, opt)

    # log AMP (automatic mixed precision)
    if opt.get("use_amp", False) and opt.get("bfloat16", False):
        logger.info("AMP enabled with BF16.")
    elif opt.get("use_amp", False) and not opt.get("bfloat16", False):
        logger.info("AMP enabled.")
    else:
        logger.info("AMP disabled.")

    # error if bf16 is enabled by not amp
    if not opt.get("use_amp", False) and opt.get("bfloat16", False):
        msg = f"{tc.red}bfloat16 option has no effect without use_amp.{tc.end}"
        logger.error(msg)
        sys.exit(1)

    # detect GPU architecture
    is_ampere = torch.cuda.get_device_capability()[0] >= 8
    is_turing = torch.cuda.get_device_capability()[0] == 7
    is_pascal = torch.cuda.get_device_capability()[0] <= 6

    # detect Ampere and recommend bf16
    if opt.get("use_amp", False) is False and is_ampere:
        msg = f"{tc.light_yellow}Modern GPU detected. Consider enabling AMP with bfloat16.{tc.end}"
        logger.warning(msg)

    # detect Turing or older and error if bf16 is enabled
    if opt.get("bfloat16", False) is True and is_turing:
        msg = f"{tc.light_yellow}Turing GPU detected. Consider disabling bfloat16.{tc.end}"
        logger.warning(msg)

    # detect Pascal or older and warn about AMP
    if opt.get("use_amp", False) is True and is_pascal:
        msg = f"{tc.light_yellow}Pascal GPU doesn't have tensor cores. Consider disabling AMP.{tc.end}"
        logger.warning(msg)

    # log deterministic mode
    if opt["deterministic"]:
        logger.info("Deterministic mode enabled.")

    # training log vars
    print_freq = opt["logger"].get("print_freq", 100)
    save_checkpoint_freq = opt["logger"]["save_checkpoint_freq"]
    val_freq = opt["val"]["val_freq"] if opt.get("val") is not None else 100

    # training
    logger.info(
        f"{tc.light_green}Start training from epoch: {start_epoch}, iter: {current_optimizer_step}{tc.end}"
    )
    # data_timer, iter_timer = AvgTimer(), AvgTimer()
    iter_timer = AvgTimer()
    start_time = time.time()
    first_optimizer_step = current_optimizer_step + 1
    epoch = start_epoch
    checkpoint_safe = True

    def save_resumable_checkpoint() -> None:
        progress_epoch, progress_batch = divmod(
            current_micro_batch, accumulation_plan.micro_batches_per_epoch
        )
        progress = ResumePosition(
            current_optimizer_step, current_micro_batch, progress_epoch, progress_batch
        ).state_dict(accumulation_plan, train_sampler.seed)
        model.save(  # type: ignore[reportFunctionMemberAccess,attr-defined]
            progress_epoch, current_optimizer_step, progress
        )

    try:
        while current_micro_batch < accumulation_plan.total_micro_batches:
            train_sampler.set_epoch(
                epoch, start_index=batch_in_epoch * process_batch_size
            )
            prefetcher.reset()
            train_data = prefetcher.next()

            while (
                train_data is not None
                and current_micro_batch < accumulation_plan.total_micro_batches
            ):
                # data_timer.record()

                # From here until a complete effective-batch update (including its
                # scheduler step) is committed, gradients/model/RNG form an
                # in-flight transaction that cannot be resumed from disk.
                checkpoint_safe = False
                next_micro_batch = current_micro_batch + 1
                pending_optimizer_step = (
                    accumulation_plan.optimizer_step_for_micro_batch(next_micro_batch)
                )
                expected_update_boundary = accumulation_plan.should_step(
                    next_micro_batch
                )

                # training
                model.feed_data(train_data)  # type: ignore[reportAttributeAccessIssue,attr-defined]
                completed_effective_batch = model.optimize_parameters(  # type: ignore[reportFunctionMemberAccess,attr-defined]
                    pending_optimizer_step
                )
                if completed_effective_batch != expected_update_boundary:
                    msg = (
                        "Model and training-loop accumulation state diverged at "
                        f"micro-batch {next_micro_batch}."
                    )
                    raise RuntimeError(msg)
                current_micro_batch = next_micro_batch
                batch_in_epoch += 1

                if completed_effective_batch:
                    current_optimizer_step = pending_optimizer_step
                    # Schedulers and warm-up advance once per optimizer update.
                    model.update_learning_rate(  # type: ignore[reportFunctionMemberAccess,attr-defined]
                        current_optimizer_step,
                        warmup_iter=opt["train"].get("warmup_iter", -1),
                    )
                    checkpoint_safe = True
                    iter_timer.record()
                    if current_optimizer_step == first_optimizer_step:
                        # Exclude initialization and the first update from ETA.
                        msg_logger.reset_start_time()

                    # log
                    if current_optimizer_step % print_freq == 0:
                        log_vars = {"epoch": epoch, "iter": current_optimizer_step}
                        log_vars.update({
                            "lrs": model.get_current_learning_rate()  # type: ignore[reportFunctionMemberAccess,attr-defined]
                        })
                        log_vars.update({
                            "time": iter_timer.get_avg_time()
                            # "data_time": data_timer.get_avg_time(),
                        })
                        log_vars.update(model.get_current_log())  # type: ignore[reportFunctionMemberAccess,attr-defined]
                        msg_logger(log_vars)

                    # save models and training states
                    if current_optimizer_step % save_checkpoint_freq == 0:
                        # check if there's enough disk space
                        free_space = check_disk_space()
                        if free_space < 500:
                            msg = f"""
                            {tc.red}
                            Not enough free disk space in {Path.cwd()}.
                            Please free up at least 500 MB of space.
                            Attempting to save current progress...
                            {tc.end}
                            """
                            logger.error(msg)
                            save_resumable_checkpoint()
                            sys.exit(1)

                        logger.info(
                            f"{tc.light_green}Saving models and training states.{tc.end}"
                        )
                        save_resumable_checkpoint()

                    # Validation is observational and must not move the training
                    # RNG stream past the state captured in the checkpoint above.
                    if (
                        opt.get("val") is not None
                        and current_optimizer_step % val_freq == 0
                    ):
                        with preserve_rng_state():
                            for val_loader in val_loaders:
                                model.validation(  # type: ignore[reportFunctionMemberAccess,attr-defined]
                                    val_loader,
                                    current_optimizer_step,
                                    tb_logger,
                                    resolve_validation_save_img(
                                        val_loader, opt.get("val")
                                    ),
                                )

                # data_timer.start()
                train_data = prefetcher.next()
            # end of iter
            epoch += 1
            batch_in_epoch = 0

        # end of epoch

        consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        logger.info(
            f"{tc.light_green}End of training. Time consumed: {consumed_time}{tc.end}"
        )
        logger.info(f"{tc.light_green}Save the latest model.{tc.end}")
        # -1 stands for the latest
        model.save(epoch=-1, current_iter=-1)  # type: ignore[reportFunctionMemberAccess,attr-defined]

    except KeyboardInterrupt:
        if checkpoint_safe:
            msg = f"{tc.light_green}Interrupted, saving resumable state.{tc.end}"
            logger.info(msg)
            save_resumable_checkpoint()
        else:
            logger.warning(
                "Interrupted inside an accumulation window. Partial gradients are "
                "not checkpoint-safe; saving inference weights only and preserving "
                "the previous resumable checkpoint."
            )
            model.save(epoch=-1, current_iter=-1)  # type: ignore[reportFunctionMemberAccess,attr-defined]
        sys.exit(0)

    if opt.get("val") is not None:
        for val_loader in val_loaders:
            model.validation(  # type: ignore[reportFunctionMemberAccess,attr-defined]
                val_loader,
                current_optimizer_step,
                tb_logger,
                resolve_validation_save_img(val_loader, opt.get("val")),
            )
    if tb_logger:
        tb_logger.close()


if __name__ == "__main__":
    root_path = Path.resolve(Path(__file__) / osp.pardir)
    train_pipeline(str(root_path))
