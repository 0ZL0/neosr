import hashlib
import os
import threading
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import get_worker_info


def _rng_context() -> tuple[int, int, int, int]:
    """Return the process-local context that determines an RNG stream."""
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
