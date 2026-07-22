from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True, slots=True)
class AccumulationPlan:
    """Translate loader yields into the optimizer-step units used by configs."""

    total_optimizer_steps: int
    accumulation_steps: int
    micro_batches_per_epoch: int

    def __post_init__(self) -> None:
        for name, value in (
            ("total_optimizer_steps", self.total_optimizer_steps),
            ("accumulation_steps", self.accumulation_steps),
            ("micro_batches_per_epoch", self.micro_batches_per_epoch),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                msg = f"{name} must be an integer"
                raise TypeError(msg)
            if value < 1:
                msg = f"{name} must be at least 1"
                raise ValueError(msg)

    @property
    def total_micro_batches(self) -> int:
        return self.total_optimizer_steps * self.accumulation_steps

    @property
    def total_epochs(self) -> int:
        return (
            self.total_micro_batches + self.micro_batches_per_epoch - 1
        ) // self.micro_batches_per_epoch

    def optimizer_step_for_micro_batch(self, micro_batch: int) -> int:
        """Return the optimizer step whose effective batch is being assembled."""
        self._validate_micro_batch(micro_batch)
        return (micro_batch - 1) // self.accumulation_steps + 1

    def should_step(self, micro_batch: int) -> bool:
        """Report whether ``micro_batch`` completes an effective batch."""
        self._validate_micro_batch(micro_batch)
        return micro_batch % self.accumulation_steps == 0

    def _validate_micro_batch(self, micro_batch: int) -> None:
        if isinstance(micro_batch, bool) or not isinstance(micro_batch, int):
            msg = "micro_batch must be an integer"
            raise TypeError(msg)
        if not 1 <= micro_batch <= self.total_micro_batches:
            msg = (
                "micro_batch must be between 1 and "
                f"{self.total_micro_batches}, got {micro_batch}"
            )
            raise ValueError(msg)


def normalize_accumulation_steps(value: object) -> int:
    """Normalize the legacy disabled values and reject ambiguous configurations."""
    if value is None:
        return 1
    if isinstance(value, bool) or not isinstance(value, int):
        msg = "datasets.train.accumulate must be an integer."
        raise TypeError(msg)
    if value == 0:
        return 1
    if value < 1:
        msg = "datasets.train.accumulate must be a positive integer."
        raise ValueError(msg)
    return value


def advance_accumulation(current: int, accumulation_steps: int) -> tuple[int, bool]:
    """Advance one micro-batch and report whether optimizers should step."""
    if isinstance(accumulation_steps, bool) or not isinstance(accumulation_steps, int):
        msg = "accumulation_steps must be an integer"
        raise TypeError(msg)
    if accumulation_steps < 1:
        msg = "accumulation_steps must be at least 1"
        raise ValueError(msg)
    if isinstance(current, bool) or not isinstance(current, int):
        msg = "current must be an integer"
        raise TypeError(msg)
    if not 0 <= current < accumulation_steps:
        msg = (
            "current must be a valid in-progress micro-batch count, got "
            f"{current} for {accumulation_steps} accumulation steps"
        )
        raise ValueError(msg)
    current += 1
    if current >= accumulation_steps:
        return 0, True
    return current, False


def grad_scaler_step_succeeded(previous_scale: float, current_scale: float) -> bool:
    """Infer whether GradScaler ran the optimizer instead of backing off."""
    return current_scale >= previous_scale


@contextmanager
def accumulation_sync_context(
    networks: Iterable[nn.Module | None], *, should_sync: bool
) -> Iterator[None]:
    """Skip DDP gradient reductions until an effective batch is complete."""
    with ExitStack() as stack:
        if not should_sync:
            for network in networks:
                if isinstance(network, DistributedDataParallel):
                    stack.enter_context(network.no_sync())
        yield


def generator_adversarial_loss(
    discriminator: nn.Module, output: Tensor, criterion: Callable[..., Tensor]
) -> Tensor:
    """Evaluate a frozen discriminator while retaining gradients to ``output``."""
    was_training = discriminator.training
    discriminator.eval()
    try:
        prediction = discriminator(output)
        return criterion(prediction, target_is_real=True, is_disc=False)
    finally:
        discriminator.train(was_training)
