from pathlib import Path
from unittest.mock import patch

import torch

from neosr.models.base import base


class _FakeGradScaler:
    def __init__(self, scale: float = 32.0) -> None:
        self.scale = scale
        self.loaded_state: dict[str, float] | None = None

    def state_dict(self) -> dict[str, float]:
        return {"scale": self.scale}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.loaded_state = state


class _FakeScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


def _make_base_model(training_states: Path) -> base:
    model = object.__new__(base)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    model.opt = {"path": {"training_states": training_states}, "use_amp": True}
    model.optimizers = [torch.optim.SGD([parameter], lr=0.1)]
    model.schedulers = []
    model.gradscaler_g = _FakeGradScaler()
    model.accum_iters = 4
    model.sf_optim_g = False
    model.net_d = None
    model.is_train = True
    return model


def test_training_state_round_trips_grad_scaler_and_accumulation_metadata(
    tmp_path: Path,
) -> None:
    model = _make_base_model(tmp_path)

    with (
        patch("neosr.models.base.torch.cuda.empty_cache"),
        patch("neosr.models.base.gc.collect"),
    ):
        model.save_training_state(
            epoch=2,
            current_iter=7,
            training_progress={
                "version": 1,
                "global_micro_batch": 28,
                "micro_batches_per_epoch": 10,
            },
        )

    state_path = tmp_path / "7.state"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    assert state["iter"] == 7
    assert state["accumulation_steps"] == 4
    assert state["training_progress"]["global_micro_batch"] == 28
    assert "rng_state" in state["rank_state"]
    assert state["grad_scalers"]["gradscaler_g"] == {"scale": 32.0}

    resumed_model = _make_base_model(tmp_path)
    with (
        patch("neosr.models.base.torch.cuda.empty_cache"),
        patch("neosr.models.base.gc.collect"),
    ):
        resumed_model.resume_training(state)

    assert resumed_model.gradscaler_g.loaded_state == {"scale": 32.0}


def test_scheduler_advances_on_optimizer_steps_not_micro_batch_state() -> None:
    model = object.__new__(base)
    scheduler = _FakeScheduler()
    model.schedulers = [scheduler]
    model.optimizers = [
        torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.0))], lr=0.1)
    ]
    model.n_accumulated = 2
    model.optimizer_step_succeeded = [True]

    model.update_learning_rate(current_iter=5)

    assert scheduler.steps == 1


def test_scheduler_does_not_advance_when_grad_scaler_skips_optimizer() -> None:
    model = object.__new__(base)
    scheduler = _FakeScheduler()
    model.schedulers = [scheduler]
    model.optimizers = [
        torch.optim.SGD([torch.nn.Parameter(torch.tensor(1.0))], lr=0.1)
    ]
    model.optimizer_step_succeeded = [False]

    model.update_learning_rate(current_iter=5)

    assert scheduler.steps == 0


def test_legacy_training_state_without_grad_scaler_still_resumes(
    tmp_path: Path,
) -> None:
    model = _make_base_model(tmp_path)
    model.opt["use_amp"] = False
    legacy_state = {
        "epoch": 1,
        "iter": 2,
        "optimizers": [model.optimizers[0].state_dict()],
        "schedulers": [],
    }

    with (
        patch("neosr.models.base.torch.cuda.empty_cache"),
        patch("neosr.models.base.gc.collect"),
    ):
        model.resume_training(legacy_state)

    assert model.gradscaler_g.loaded_state is None
