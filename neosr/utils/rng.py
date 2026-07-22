import hashlib
import os
import random
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import get_worker_info

_registered_generators: dict[str, Any] = {}
_sample_context = threading.local()


def _rng_context() -> tuple[int, int, int, int]:
    """Return the process-local context that determines an RNG stream."""
    sample_seed = getattr(_sample_context, "seed", None)
    if sample_seed is not None:
        # A sample seed must not depend on worker assignment. This makes a resumed
        # sample identical even when the DataLoader schedules it on another worker.
        return os.getpid(), int(sample_seed), 0, -1
    worker = get_worker_info()
    worker_id = worker.id if worker is not None else -1
    torch_seed = torch.initial_seed()
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    return os.getpid(), int(torch_seed), rank, worker_id


def _namespace_entropy(namespace: str) -> tuple[int, int]:
    digest = hashlib.blake2b(namespace.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="little", signed=False)
    return value & 0xFFFFFFFF, value >> 32


def _make_generator(
    namespace: str, context: tuple[int, int, int, int]
) -> np.random.Generator:
    _, torch_seed, rank, worker_id = context
    namespace_low, namespace_high = _namespace_entropy(namespace)
    entropy = [
        torch_seed & 0xFFFFFFFF,
        (torch_seed >> 32) & 0xFFFFFFFF,
        rank & 0xFFFFFFFF,
        (worker_id + 1) & 0xFFFFFFFF,
        namespace_low,
        namespace_high,
    ]
    return np.random.default_rng(np.random.SeedSequence(entropy))


class LazyGenerator:
    """A NumPy generator isolated by distributed rank, worker and module.

    The underlying generator is created on first use. This is important for data-loader
    workers: module imports happen before worker seeding on some platforms, while the
    first random draw happens after PyTorch has assigned the worker's unique seed. The
    process ID is tracked so a fork also creates a fresh local generator.
    """

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self._local = threading.local()
        _registered_generators[namespace] = self

    def _generator(self) -> np.random.Generator:
        context = _rng_context()
        if getattr(self._local, "context", None) != context:
            self._local.context = context
            self._local.generator = _make_generator(self.namespace, context)
        return self._local.generator

    def __getattr__(self, name: str) -> Any:
        return getattr(self._generator(), name)


def rng(namespace: str = "neosr") -> LazyGenerator:
    """Create a lazily seeded RNG stream for a stable module namespace."""
    return LazyGenerator(namespace)


def _lazy_rng_state() -> dict[str, dict[str, Any]]:
    return {
        namespace: deepcopy(generator._local.generator.bit_generator.state)
        for namespace, generator in _registered_generators.items()
        if hasattr(generator._local, "generator")
    }


def _restore_lazy_rng_state(states: Mapping[str, Mapping[str, Any]]) -> None:
    for namespace, generator in _registered_generators.items():
        generator._local.__dict__.clear()
        if namespace in states:
            generator._generator().bit_generator.state = deepcopy(
                dict(states[namespace])
            )


def _capture_cuda_rng_state() -> tuple[bool, list[torch.Tensor]] | None:
    if not torch.cuda.is_available():
        return None
    if dist.is_available() and dist.is_initialized():
        # A DDP process owns one current device. Querying every device here would
        # initialize needless CUDA contexts in every rank.
        return True, [torch.cuda.get_rng_state()]
    return False, torch.cuda.get_rng_state_all()


def _restore_cuda_rng_state(state: tuple[bool, list[torch.Tensor]]) -> None:
    current_only, cuda_states = state
    if current_only:
        torch.cuda.set_rng_state(cuda_states[0].cpu())
        return
    if len(cuda_states) != torch.cuda.device_count():
        msg = (
            "CUDA device count changed since the checkpoint; exact RNG "
            "restoration is not possible."
        )
        raise RuntimeError(msg)
    torch.cuda.set_rng_state_all([cuda_state.cpu() for cuda_state in cuda_states])


def capture_rng_state(*, include_cuda: bool = True) -> dict[str, Any]:
    """Capture all main-process RNG streams required for reproducible resumption."""
    numpy_state = np.random.get_state()  # noqa: NPY002
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "lazy_generators": _lazy_rng_state(),
    }
    if include_cuda and (cuda_state := _capture_cuda_rng_state()) is not None:
        state["torch_cuda"] = cuda_state
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state created by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((  # noqa: NPY002
        numpy_state["bit_generator"],
        numpy_state["keys"].cpu().numpy(),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        _restore_cuda_rng_state(state["torch_cuda"])
    _restore_lazy_rng_state(state.get("lazy_generators", {}))


@contextmanager
def preserve_rng_state() -> Iterator[None]:
    """Run observational work without advancing training RNG streams."""
    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)


@contextmanager
def deterministic_sample(seed: int) -> Iterator[None]:
    """Isolate dataset randomness behind a worker-independent sample seed."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        msg = "sample seed must be an integer"
        raise TypeError(msg)

    worker_process = get_worker_info() is not None
    lazy_states = {
        generator: generator._local.__dict__.copy()
        for generator in _registered_generators.values()
    }
    if not worker_process:
        python_state = random.getstate()
        numpy_state = np.random.get_state()  # noqa: NPY002
        torch_state = torch.get_rng_state()
        cuda_state = (
            _capture_cuda_rng_state()
            if torch.cuda.is_available() and torch.get_default_device().type == "cuda"
            else None
        )

    _sample_context.seed = seed
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)  # noqa: NPY002
    torch.random.default_generator.manual_seed(seed)
    if not worker_process and cuda_state is not None:
        if cuda_state[0]:
            torch.cuda.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        _sample_context.__dict__.pop("seed", None)
        for generator, generator_state in lazy_states.items():
            generator._local.__dict__.clear()
            generator._local.__dict__.update(generator_state)
        if not worker_process:
            random.setstate(python_state)
            np.random.set_state(numpy_state)  # noqa: NPY002
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                _restore_cuda_rng_state(cuda_state)
