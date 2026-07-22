import random
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset

from neosr.data.data_sampler import EnlargedSampler, SeededDataset, SeededIndex
from neosr.models.base import base
from neosr.utils.misc import check_resume
from neosr.utils.rng import capture_rng_state, restore_rng_state, rng

_dataset_generator = rng("tests.exact_resume.dataset")


class _RandomDataset(Dataset):
    def __len__(self) -> int:
        return 12

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor([
            float(index),
            random.random(),
            float(np.random.random()),  # noqa: NPY002
            torch.rand(()).item(),
            float(_dataset_generator.random()),
        ])


def _random_draw(generator) -> tuple[float, float, float, float]:
    return (
        random.random(),
        float(np.random.random()),  # noqa: NPY002
        torch.rand(()).item(),
        float(generator.random()),
    )


def test_seeded_dataset_is_worker_independent_and_does_not_consume_main_rng() -> None:
    dataset = SeededDataset(_RandomDataset())
    request = SeededIndex(index=4, seed=123456)
    caller_generator = rng("tests.exact_resume.caller")
    state = capture_rng_state(include_cuda=False)

    first = dataset[request]
    second = dataset[request]
    actual_next_draw = _random_draw(caller_generator)
    restore_rng_state(state)
    expected_next_draw = _random_draw(caller_generator)

    torch.testing.assert_close(first, second)
    assert actual_next_draw == expected_next_draw


def test_enlarged_sampler_resumes_at_an_exact_sample_offset() -> None:
    dataset = _RandomDataset()
    sampler = EnlargedSampler(dataset, num_replicas=1, rank=0, ratio=1, seed=9)
    sampler.set_epoch(3)
    complete_epoch = list(sampler)

    sampler.set_epoch(3, start_index=5)

    assert list(sampler) == complete_epoch[5:]
    assert len(sampler) == len(complete_epoch) - 5


def _collect_worker_samples(num_workers: int, start_index: int = 0) -> torch.Tensor:
    dataset = _RandomDataset()
    sampler = EnlargedSampler(dataset, num_replicas=1, rank=0, ratio=1, seed=27)
    sampler.set_epoch(2, start_index=start_index)
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(27)
    loader = DataLoader(
        SeededDataset(dataset),
        batch_size=2,
        sampler=sampler,
        num_workers=num_workers,
        generator=loader_generator,
        persistent_workers=False,
    )
    return torch.cat(list(loader))


def test_sample_stream_is_independent_of_worker_count_and_resumes_exactly() -> None:
    complete = _collect_worker_samples(num_workers=2)
    different_worker_count = _collect_worker_samples(num_workers=1)
    resumed_tail = _collect_worker_samples(num_workers=1, start_index=4)

    torch.testing.assert_close(complete, different_worker_count)
    torch.testing.assert_close(resumed_tail, complete[4:])


def test_rng_checkpoint_is_weights_only_serializable_and_round_trips(
    tmp_path: Path,
) -> None:
    generator = rng("tests.exact_resume.roundtrip")
    saved_state = capture_rng_state(include_cuda=False)
    state_path = tmp_path / "rng.state"
    torch.save(saved_state, state_path)
    loaded_state = torch.load(state_path, map_location="cpu", weights_only=True)

    restore_rng_state(loaded_state)
    expected = _random_draw(generator)
    _random_draw(generator)
    restore_rng_state(loaded_state)

    assert _random_draw(generator) == expected


def _make_saver(models_path: Path) -> base:
    model = object.__new__(base)
    model.opt = {"path": {"models": models_path}}
    model.device = torch.device("cpu")
    model.sf_optim_g = False
    model.sf_optim_d = False
    model.net_d = None
    model.is_train = True
    model._print_different_keys_loading = lambda *_args, **_kwargs: None
    return model


def test_network_checkpoint_keeps_raw_and_ema_parameters_separate(
    tmp_path: Path,
) -> None:
    saver = _make_saver(tmp_path)
    raw = nn.Linear(2, 1)
    ema = AveragedModel(raw)
    with torch.no_grad():
        raw.weight.fill_(2)
        ema.module.weight.fill_(5)

    saver.save_network([raw, ema], "net_g", 7, param_key=["params", "params_ema"])
    checkpoint = torch.load(
        tmp_path / "net_g_7.pth", map_location="cpu", weights_only=True
    )

    assert set(checkpoint) == {"params", "params_ema"}
    torch.testing.assert_close(checkpoint["params"]["weight"], raw.weight)
    torch.testing.assert_close(checkpoint["params_ema"]["weight"], ema.module.weight)

    loaded_raw = nn.Linear(2, 1)
    loaded_ema = nn.Linear(2, 1)
    assert (
        saver.load_network(loaded_raw, tmp_path / "net_g_7.pth", "params") == "params"
    )
    assert (
        saver.load_network(loaded_ema, tmp_path / "net_g_7.pth", "params_ema")
        == "params_ema"
    )
    torch.testing.assert_close(loaded_raw.weight, raw.weight)
    torch.testing.assert_close(loaded_ema.weight, ema.module.weight)

    opt = {
        "network_g": {},
        "network_d": {},
        "path": {
            "resume_state": tmp_path / "7.state",
            "models": tmp_path,
            "param_key_g": "params_ema",
            "param_key_d": "params_ema",
        },
    }

    check_resume(opt, resume_iter=7)

    assert opt["path"]["pretrain_network_g"] == tmp_path / "net_g_7.pth"
    assert opt["path"]["pretrain_network_d"] == tmp_path / "net_d_7.pth"
    assert opt["path"]["param_key_g"] == "params"
    assert opt["path"]["param_key_d"] == "params"

    legacy_path = tmp_path / "legacy.pth"
    torch.save({"params": raw.state_dict()}, legacy_path)
    target = nn.Linear(2, 1)
    assert saver.load_network(target, legacy_path, "params_ema") == "params"
    torch.testing.assert_close(target.weight, raw.weight)


def test_ema_update_counter_round_trips() -> None:
    sys.modules.setdefault("pywt", types.ModuleType("pywt"))
    from neosr.models.image import image  # noqa: PLC0415

    source = nn.Linear(2, 1)
    original = object.__new__(image)
    original.ema = 0.999
    original.net_g_ema = AveragedModel(source)
    original.net_g_ema.n_averaged.fill_(17)
    original.net_d = None

    state = original.get_training_state()
    resumed = object.__new__(image)
    resumed.ema = 0.999
    resumed.net_g_ema = AveragedModel(source)
    resumed.net_d = None
    resumed.load_training_state(state)

    assert resumed.net_g_ema.n_averaged.item() == 17


def test_otf_training_pair_queue_round_trips() -> None:
    sys.modules.setdefault("pywt", types.ModuleType("pywt"))
    from neosr.models.otf import otf  # noqa: PLC0415

    original = object.__new__(otf)
    original.ema = -1
    original.net_d = None
    original.device = torch.device("cpu")
    original.queue_size = 4
    original.queue_lr = torch.arange(8, dtype=torch.float32).reshape(4, 1, 1, 2)
    original.queue_gt = original.queue_lr + 100
    original.queue_ptr = 2

    state = original.get_training_state()
    resumed = object.__new__(otf)
    resumed.ema = -1
    resumed.net_d = None
    resumed.device = torch.device("cpu")
    resumed.queue_size = 4
    resumed.load_training_state(state)

    torch.testing.assert_close(resumed.queue_lr, original.queue_lr)
    torch.testing.assert_close(resumed.queue_gt, original.queue_gt)
    assert resumed.queue_ptr == 2
