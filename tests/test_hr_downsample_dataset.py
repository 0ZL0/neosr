from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from neosr.data import build_dataset
from neosr.data.data_sampler import SeededDataset, SeededIndex


def _write_image(path: Path, height: int, width: int) -> None:
    yy, xx = np.mgrid[:height, :width]
    image = np.stack(
        (
            (xx * 3 + yy) % 256,
            (xx + yy * 5) % 256,
            (xx * 7 + yy * 11) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    assert cv2.imwrite(str(path), image)


def _options(root: Path, **overrides):
    options = {
        "type": "hr_downsample",
        "dataroot_gt": str(root),
        "phase": "train",
        "scale": 4,
        "patch_size": 16,
        "downsample_mode": "bicubic",
        "antialias": True,
        "use_hflip": False,
        "use_rot": False,
    }
    options.update(overrides)
    return options


def test_builds_exact_aligned_pair(tmp_path: Path) -> None:
    _write_image(tmp_path / "sample.png", 96, 104)
    dataset = build_dataset(_options(tmp_path))

    sample = SeededDataset(dataset)[SeededIndex(0, 1234)]

    assert sample["gt"].shape == (3, 64, 64)
    assert sample["lq"].shape == (3, 16, 16)
    expected = F.interpolate(
        sample["gt"].unsqueeze(0),
        size=(16, 16),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).squeeze(0).clamp(0, 1)
    torch.testing.assert_close(sample["lq"], expected, rtol=0, atol=0)


def test_seeded_sample_is_worker_independent(tmp_path: Path) -> None:
    _write_image(tmp_path / "sample.png", 101, 113)
    dataset = SeededDataset(
        build_dataset(_options(tmp_path, use_hflip=True, use_rot=True))
    )

    first = dataset[SeededIndex(0, 987654)]
    second = dataset[SeededIndex(0, 987654)]

    torch.testing.assert_close(first["gt"], second["gt"], rtol=0, atol=0)
    torch.testing.assert_close(first["lq"], second["lq"], rtol=0, atol=0)


def test_validation_mod_crops_before_downsampling(tmp_path: Path) -> None:
    _write_image(tmp_path / "sample.png", 101, 103)
    dataset = build_dataset(
        _options(tmp_path, phase="val", patch_size=None, scale=4)
    )

    sample = dataset[0]

    assert sample["gt"].shape == (3, 100, 100)
    assert sample["lq"].shape == (3, 25, 25)


def test_small_image_is_padded_to_training_patch(tmp_path: Path) -> None:
    _write_image(tmp_path / "sample.png", 20, 30)
    dataset = build_dataset(_options(tmp_path, patch_size=16))

    sample = dataset[0]

    assert sample["gt"].shape == (3, 64, 64)
    assert sample["lq"].shape == (3, 16, 16)


def test_small_image_can_fail_instead_of_padding(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    _write_image(image_path, 20, 30)
    dataset = build_dataset(
        _options(tmp_path, patch_size=16, pad_if_smaller=False)
    )

    with pytest.raises(ValueError, match=str(image_path)):
        dataset[0]


def test_area_mode_has_no_extra_quantization(tmp_path: Path) -> None:
    _write_image(tmp_path / "sample.png", 64, 64)
    dataset = build_dataset(_options(tmp_path, downsample_mode="area"))

    sample = dataset[0]
    scaled = sample["lq"] * 255

    assert torch.any(scaled != scaled.round())
