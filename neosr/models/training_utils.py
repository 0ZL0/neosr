from collections.abc import Callable

from torch import Tensor, nn


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
    if accumulation_steps < 1:
        msg = "accumulation_steps must be at least 1"
        raise ValueError(msg)
    current += 1
    if current >= accumulation_steps:
        return 0, True
    return current, False


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
