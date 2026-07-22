from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from torch import nn

import neosr.models.training_utils as training_utils
from neosr.models.training_utils import (
    AccumulationPlan,
    accumulation_sync_context,
    advance_accumulation,
    grad_scaler_step_succeeded,
    normalize_accumulation_steps,
)
from neosr.utils.logger import MessageLogger


def test_plan_uses_micro_batches_per_epoch() -> None:
    plan = AccumulationPlan(
        total_optimizer_steps=1_000, accumulation_steps=4, micro_batches_per_epoch=10
    )

    assert plan.total_micro_batches == 4_000
    assert plan.total_epochs == 400


def test_all_micro_batches_share_the_pending_optimizer_step() -> None:
    plan = AccumulationPlan(
        total_optimizer_steps=3, accumulation_steps=4, micro_batches_per_epoch=5
    )

    assert [plan.optimizer_step_for_micro_batch(i) for i in range(1, 9)] == [
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
    ]
    assert [plan.should_step(i) for i in range(1, 9)] == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "total_optimizer_steps": 0,
            "accumulation_steps": 1,
            "micro_batches_per_epoch": 1,
        },
        {
            "total_optimizer_steps": 1,
            "accumulation_steps": 0,
            "micro_batches_per_epoch": 1,
        },
        {
            "total_optimizer_steps": 1,
            "accumulation_steps": 1,
            "micro_batches_per_epoch": 0,
        },
    ],
)
def test_plan_rejects_invalid_counts(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="must be at least 1"):
        AccumulationPlan(**kwargs)


def test_plan_rejects_out_of_range_micro_batch() -> None:
    plan = AccumulationPlan(2, 2, 1)

    with pytest.raises(ValueError, match="micro_batch must be between"):
        plan.optimizer_step_for_micro_batch(0)
    with pytest.raises(ValueError, match="micro_batch must be between"):
        plan.should_step(plan.total_micro_batches + 1)


@pytest.mark.parametrize("value", [True, 1.5, "4"])
def test_accumulation_rejects_ambiguous_values(value: object) -> None:
    with pytest.raises(TypeError):
        normalize_accumulation_steps(value)


def test_legacy_disabled_accumulation_values_are_normalized() -> None:
    assert normalize_accumulation_steps(None) == 1
    assert normalize_accumulation_steps(0) == 1
    assert normalize_accumulation_steps(4) == 4
    with pytest.raises(ValueError, match="must be a positive integer"):
        normalize_accumulation_steps(-1)


def test_advance_only_steps_at_the_effective_batch_boundary() -> None:
    accumulated = 0
    boundaries = []

    for _ in range(8):
        accumulated, should_step = advance_accumulation(accumulated, 4)
        boundaries.append(should_step)

    assert boundaries == [False, False, False, True, False, False, False, True]
    assert accumulated == 0


def test_advance_rejects_divergent_accumulation_state() -> None:
    with pytest.raises(ValueError, match="valid in-progress micro-batch count"):
        advance_accumulation(current=4, accumulation_steps=4)


@pytest.mark.parametrize(
    ("previous_scale", "current_scale", "succeeded"),
    [(32.0, 32.0, True), (32.0, 64.0, True), (32.0, 16.0, False)],
)
def test_grad_scaler_backoff_identifies_a_skipped_optimizer_step(
    previous_scale: float, current_scale: float, *, succeeded: bool
) -> None:
    assert grad_scaler_step_succeeded(previous_scale, current_scale) is succeeded


def test_real_grad_scaler_backoff_skips_nonfinite_optimizer_update() -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=32.0)
    optimizer.zero_grad(set_to_none=True)

    scaler.scale(parameter * torch.tensor(float("inf"))).backward()
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()

    torch.testing.assert_close(parameter.detach(), torch.tensor(1.0))
    assert not grad_scaler_step_succeeded(previous_scale, scaler.get_scale())


class _FakeDistributedDataParallel:
    def __init__(self) -> None:
        self.events: list[str] = []

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        self.events.append("enter")
        try:
            yield
        finally:
            self.events.append("exit")


def test_ddp_no_sync_wraps_the_whole_intermediate_micro_batch() -> None:
    network = _FakeDistributedDataParallel()

    with (
        patch.object(
            training_utils, "DistributedDataParallel", _FakeDistributedDataParallel
        ),
        accumulation_sync_context((network,), should_sync=False),
    ):
        network.events.append("forward-backward")

    assert network.events == ["enter", "forward-backward", "exit"]


def test_ddp_boundary_micro_batch_keeps_gradient_synchronization() -> None:
    network = _FakeDistributedDataParallel()

    with (
        patch.object(
            training_utils, "DistributedDataParallel", _FakeDistributedDataParallel
        ),
        accumulation_sync_context((network,), should_sync=True),
    ):
        network.events.append("forward-backward")

    assert network.events == ["forward-backward"]


def test_equal_micro_batches_match_one_effective_batch_update() -> None:
    torch.manual_seed(7)
    full_batch_model = nn.Linear(3, 2, dtype=torch.float64)
    accumulated_model = deepcopy(full_batch_model)
    full_batch_optimizer = torch.optim.SGD(full_batch_model.parameters(), lr=0.1)
    accumulated_optimizer = torch.optim.SGD(accumulated_model.parameters(), lr=0.1)
    inputs = torch.randn(8, 3, dtype=torch.float64)
    targets = torch.randn(8, 2, dtype=torch.float64)
    loss_fn = nn.MSELoss()

    full_batch_optimizer.zero_grad(set_to_none=True)
    loss_fn(full_batch_model(inputs), targets).backward()
    full_batch_optimizer.step()

    accumulated_optimizer.zero_grad(set_to_none=True)
    accumulation_steps = 4
    for input_chunk, target_chunk in zip(
        inputs.chunk(accumulation_steps), targets.chunk(accumulation_steps), strict=True
    ):
        loss = loss_fn(accumulated_model(input_chunk), target_chunk)
        (loss / accumulation_steps).backward()
    accumulated_optimizer.step()

    for full_parameter, accumulated_parameter in zip(
        full_batch_model.parameters(), accumulated_model.parameters(), strict=True
    ):
        torch.testing.assert_close(
            full_parameter, accumulated_parameter, rtol=1e-12, atol=1e-12
        )


class _CaptureLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)


def test_logger_does_not_divide_optimizer_step_or_throughput_again() -> None:
    opt = {
        "name": "accumulation-test",
        "datasets": {"train": {"accumulate": 4}},
        "logger": {"print_freq": 1, "total_iter": 100, "use_tb_logger": False},
    }
    message_logger = MessageLogger(opt, tb_logger=None, start_iter=0)
    capture = _CaptureLogger()
    message_logger.logger = capture

    message_logger({
        "epoch": 0,
        "iter": 8,
        "lrs": [1e-4],
        "time": 0.5,
        "l_g_total": 1.0,
    })

    message = capture.messages[-1]
    assert "iter:      8" in message
    assert "performance: 2.000 it/s" in message
