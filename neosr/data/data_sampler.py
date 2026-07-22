import math
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from neosr.utils.rng import deterministic_sample


@dataclass(frozen=True, slots=True)
class SeededIndex:
    """A dataset index paired with randomness independent of worker assignment."""

    index: int
    seed: int


class SeededDataset(Dataset):
    """Run each sample under the deterministic seed supplied by the sampler."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, request: int | SeededIndex):
        if isinstance(request, SeededIndex):
            with deterministic_sample(request.seed):
                return self.dataset[request.index]
        return self.dataset[request]


class EnlargedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    Modified from torch.utils.data.distributed.DistributedSampler
    Support enlarging the dataset for iteration-based training, for saving
    time when restart the dataloader after each epoch

    Args:
    ----
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        num_replicas (int | None): Number of processes participating in
            the training. It is usually the world_size.
        rank (int | None): Rank of the current process within num_replicas.
        ratio (int): Enlarging ratio. Default: 1.

    """

    def __init__(
        self,
        dataset,
        num_replicas: int = 1,
        rank: int = 1,
        ratio: int = 1,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.start_index = 0
        self.num_samples = math.ceil(len(self.dataset) * ratio / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int, start_index: int = 0) -> None:
        if not 0 <= start_index <= self.num_samples:
            msg = f"start_index must be between 0 and {self.num_samples}"
            raise ValueError(msg)
        self.epoch = epoch
        self.start_index = start_index

    def _sample_seed(self, position: int) -> int:
        # Allocate a non-overlapping seed range to each epoch and rank.
        return (
            self.seed
            + self.epoch * self.total_size
            + self.rank * self.num_samples
            + position
        ) & ((1 << 63) - 1)

    def __iter__(self) -> Iterator[SeededIndex]:
        # deterministically shuffle based on epoch
        # Keep the historical CUDA permutation during actual training so legacy
        # checkpoints resume on the same sample order. CPU is only a fallback for
        # tests and non-CUDA tooling.
        generator_device = "cuda" if torch.cuda.is_available() else "cpu"
        g = torch.Generator(device=generator_device)
        g.manual_seed(self.epoch)
        indices = torch.randperm(
            self.total_size, generator=g, device=generator_device
        ).tolist()

        dataset_size = len(self.dataset)
        indices = [v % dataset_size for v in indices]

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        seeded_indices = (
            SeededIndex(index, self._sample_seed(position))
            for position, index in enumerate(indices)
            if position >= self.start_index
        )
        return iter(seeded_indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index
