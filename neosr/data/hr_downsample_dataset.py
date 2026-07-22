import random
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from torch import Tensor
from torch.nn import functional as F
from torch.utils import data
from torchvision.transforms.functional import normalize

from neosr.data.data_util import paths_from_lmdb
from neosr.data.file_client import FileClient
from neosr.data.transforms import basic_augment, mod_crop
from neosr.utils import imfrombytes, img2tensor, scandir
from neosr.utils.registry import DATASET_REGISTRY

_SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
_SUPPORTED_MODES = {"area", "bilinear", "bicubic"}


@DATASET_REGISTRY.register()
class hr_downsample(data.Dataset):
    """Create aligned LQ/GT training pairs from HR images only.

    ``patch_size`` follows the existing neosr convention and specifies the LQ
    patch size. The HR crop is therefore ``patch_size * scale``.
    """

    def __init__(self, opt: dict[str, Any]) -> None:
        super().__init__()
        self.opt = opt
        self.file_client: FileClient | None = None
        self.gt_folder = str(opt["dataroot_gt"])
        self.phase = str(opt["phase"])
        self.scale = int(opt["scale"])
        self.patch_size = opt.get("patch_size")
        self.downsample_mode = str(opt.get("downsample_mode", "bicubic")).lower()
        self.antialias = bool(opt.get("antialias", True))
        self.pad_if_smaller = bool(opt.get("pad_if_smaller", True))
        self.use_hflip = bool(opt.get("use_hflip", True))
        self.use_rot = bool(opt.get("use_rot", True))
        self.mean = opt.get("mean")
        self.std = opt.get("std")
        self.color = opt.get("color") != "y"

        if self.scale < 1:
            msg = f"scale must be a positive integer, received {self.scale}."
            raise ValueError(msg)
        if self.phase == "train" and (
            isinstance(self.patch_size, bool)
            or not isinstance(self.patch_size, int)
            or self.patch_size < 1
        ):
            msg = "A positive integer patch_size is required for training."
            raise ValueError(msg)
        if self.downsample_mode not in _SUPPORTED_MODES:
            msg = (
                f"downsample_mode must be one of {sorted(_SUPPORTED_MODES)}, "
                f"received {self.downsample_mode!r}."
            )
            raise ValueError(msg)
        if (self.mean is None) != (self.std is None):
            msg = "mean and std must be configured together."
            raise ValueError(msg)

        if self.gt_folder.endswith(".lmdb"):
            self.io_backend_opt: dict[str, Any] = {
                "type": "lmdb",
                "db_paths": [self.gt_folder],
                "client_keys": ["gt"],
            }
            self.paths = paths_from_lmdb(self.gt_folder)
        else:
            self.io_backend_opt = {"type": "disk"}
            meta_info = opt.get("meta_info_file", opt.get("meta_info"))
            if meta_info is not None:
                with Path(str(meta_info)).open(encoding="utf-8") as file:
                    names = [
                        line.strip().split(" ")[0]
                        for line in file
                        if line.strip()
                    ]
                self.paths = [str(Path(self.gt_folder) / name) for name in names]
            else:
                self.paths = sorted(
                    str(path)
                    for path in scandir(
                        self.gt_folder, recursive=True, full_path=True
                    )
                    if str(path).lower().endswith(_SUPPORTED_EXTENSIONS)
                )

        if not self.paths:
            msg = f"No supported HR images found in {self.gt_folder}."
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.paths)

    def _get_file_client(self) -> FileClient:
        if self.file_client is None:
            backend_type = self.io_backend_opt.pop("type")
            self.file_client = FileClient(backend_type, **self.io_backend_opt)
        return self.file_client

    def _read_gt(self, gt_path: str) -> np.ndarray:
        img_bytes = self._get_file_client().get(gt_path, "gt")
        if img_bytes is None:
            msg = f"No image data returned for {gt_path}."
            raise FileNotFoundError(msg)
        try:
            return cast("np.ndarray", imfrombytes(img_bytes, float32=True))
        except Exception as error:
            msg = f"Failed to decode HR image: {gt_path}."
            raise RuntimeError(msg) from error

    def _pad_to_patch(
        self, img_gt: np.ndarray, target_size: int, gt_path: str
    ) -> np.ndarray:
        height, width = img_gt.shape[:2]
        if height >= target_size and width >= target_size:
            return img_gt
        if not self.pad_if_smaller:
            msg = (
                f"HR image {gt_path} has shape ({height}, {width}), but at least "
                f"({target_size}, {target_size}) is required."
            )
            raise ValueError(msg)

        pad_h = max(0, target_size - height)
        pad_w = max(0, target_size - width)
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        border_type = (
            cv2.BORDER_REFLECT_101
            if height > 1 and width > 1
            else cv2.BORDER_REPLICATE
        )
        return cv2.copyMakeBorder(
            img_gt,
            pad_top,
            pad_h - pad_top,
            pad_left,
            pad_w - pad_left,
            border_type,
        )

    def _random_aligned_crop(
        self, img_gt: np.ndarray, gt_path: str
    ) -> np.ndarray:
        patch_size = cast("int", self.patch_size)
        gt_patch_size = patch_size * self.scale
        img_gt = self._pad_to_patch(img_gt, gt_patch_size, gt_path)
        height, width = img_gt.shape[:2]
        top = random.randint(0, (height - gt_patch_size) // self.scale) * self.scale
        left = random.randint(0, (width - gt_patch_size) // self.scale) * self.scale
        return img_gt[
            top : top + gt_patch_size,
            left : left + gt_patch_size,
            ...,
        ]

    def _downsample(self, img_gt: Tensor) -> Tensor:
        if self.scale == 1:
            return img_gt.clone()

        output_size = (
            img_gt.shape[-2] // self.scale,
            img_gt.shape[-1] // self.scale,
        )
        if self.downsample_mode == "area":
            img_lq = F.interpolate(
                img_gt.unsqueeze(0), size=output_size, mode="area"
            )
        else:
            img_lq = F.interpolate(
                img_gt.unsqueeze(0),
                size=output_size,
                mode=self.downsample_mode,
                align_corners=False,
                antialias=self.antialias,
            )
        return img_lq.squeeze(0).clamp_(0, 1)

    def __getitem__(self, index: int) -> dict[str, str | Tensor]:
        gt_path = str(self.paths[index])
        img_gt = self._read_gt(gt_path)

        if self.phase == "train":
            img_gt = self._random_aligned_crop(img_gt, gt_path)
            img_gt = cast(
                "np.ndarray",
                basic_augment(
                    img_gt, hflip=self.use_hflip, rotation=self.use_rot
                ),
            )
        else:
            img_gt = mod_crop(img_gt, self.scale)
            if img_gt.shape[0] == 0 or img_gt.shape[1] == 0:
                msg = f"HR image {gt_path} is smaller than scale {self.scale}."
                raise ValueError(msg)

        img_gt = np.ascontiguousarray(img_gt)
        gt = cast(
            "Tensor",
            img2tensor(
                img_gt, bgr2rgb=True, float32=True, color=self.color
            ),
        )
        lq = self._downsample(gt)

        if self.mean is not None:
            normalize(gt, self.mean, self.std, inplace=True)
            normalize(lq, self.mean, self.std, inplace=True)

        return {
            "lq": lq,
            "gt": gt,
            "lq_path": gt_path,
            "gt_path": gt_path,
        }
