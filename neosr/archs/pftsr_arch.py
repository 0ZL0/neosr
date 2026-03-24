import importlib
import sys
import types
from pathlib import Path

import torch

from neosr.archs.arch_util import net_opt
from neosr.utils.registry import ARCH_REGISTRY

upscale, __ = net_opt()

_PFTSR_ROOT = Path(__file__).resolve().parents[2] / "PFT-SR"
_PFTSR_OPS = _PFTSR_ROOT / "ops_smm"
_PFT_CLASS = None

_PFTSR_NUM_TOPK = [
    1024, 1024, 1024, 1024,
    256, 256, 256, 256,
    128, 128, 128, 128,
    64, 64, 64, 64, 64, 64,
    32, 32, 32, 32, 32, 32,
    16, 16, 16, 16, 16, 16,
]

_PFTSR_LIGHT_NUM_TOPK = [
    1024, 1024,
    256, 256, 256, 256,
    128, 128, 128, 128, 128, 128,
    64, 64, 64, 64, 64, 64,
    32, 32, 32, 32, 32, 32,
]


def _ensure_search_paths() -> None:
    for path in (str(_PFTSR_OPS), str(_PFTSR_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _ensure_fairscale_stub() -> None:
    try:
        import fairscale.nn  # noqa: F401
    except ImportError:
        fairscale_module = sys.modules.setdefault("fairscale", types.ModuleType("fairscale"))
        fairscale_nn_module = types.ModuleType("fairscale.nn")

        def checkpoint_wrapper(module, *args, **kwargs):
            return module

        fairscale_nn_module.checkpoint_wrapper = checkpoint_wrapper
        fairscale_module.nn = fairscale_nn_module
        sys.modules["fairscale.nn"] = fairscale_nn_module


def _smm_qmk_forward(A: torch.Tensor, B: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    batch, query_count, feature_dim = A.shape
    topk = index.shape[-1]
    expanded_B = B.unsqueeze(1).expand(-1, query_count, -1, -1)
    gather_index = index.long().unsqueeze(2).expand(-1, -1, feature_dim, -1)
    selected_B = torch.gather(expanded_B, 3, gather_index)
    return torch.einsum("bnd,bndk->bnk", A, selected_B).contiguous()


def _smm_qmk_backward(
    grad_output: torch.Tensor, A: torch.Tensor, B: torch.Tensor, index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, query_count, feature_dim = A.shape
    key_count = B.shape[-1]
    expanded_B = B.unsqueeze(1).expand(-1, query_count, -1, -1)
    gather_index = index.long().unsqueeze(2).expand(-1, -1, feature_dim, -1)
    selected_B = torch.gather(expanded_B, 3, gather_index)

    grad_A = torch.einsum("bnk,bndk->bnd", grad_output, selected_B)
    contrib = A.unsqueeze(-1) * grad_output.unsqueeze(2)
    grad_B_full = torch.zeros(
        (batch, query_count, feature_dim, key_count),
        device=B.device,
        dtype=B.dtype,
    )
    grad_B_full.scatter_add_(3, gather_index, contrib)
    grad_B = grad_B_full.sum(dim=1)
    return grad_A.contiguous(), grad_B.contiguous()


def _smm_amv_forward(A: torch.Tensor, B: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    batch, query_count, topk = A.shape
    value_dim = B.shape[-1]
    expanded_B = B.unsqueeze(1).expand(-1, query_count, -1, -1)
    gather_index = index.long().unsqueeze(-1).expand(-1, -1, -1, value_dim)
    selected_B = torch.gather(expanded_B, 2, gather_index)
    return torch.einsum("bnk,bnkd->bnd", A, selected_B).contiguous()


def _smm_amv_backward(
    grad_output: torch.Tensor, A: torch.Tensor, B: torch.Tensor, index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, query_count, value_dim = grad_output.shape
    token_count = B.shape[1]
    gather_index = index.long().unsqueeze(-1).expand(-1, -1, -1, value_dim)
    selected_B = torch.gather(B.unsqueeze(1).expand(-1, query_count, -1, -1), 2, gather_index)

    grad_A = torch.einsum("bnd,bnkd->bnk", grad_output, selected_B)
    contrib = A.unsqueeze(-1) * grad_output.unsqueeze(2)
    grad_B_full = torch.zeros(
        (batch, query_count, token_count, value_dim),
        device=B.device,
        dtype=B.dtype,
    )
    grad_B_full.scatter_add_(2, gather_index, contrib)
    grad_B = grad_B_full.sum(dim=1)
    return grad_A.contiguous(), grad_B.contiguous()


def _ensure_smm_fallback() -> None:
    try:
        import smm_cuda  # noqa: F401
    except ImportError:
        smm_module = types.ModuleType("smm_cuda")
        smm_module.SMM_QmK_forward_cuda = _smm_qmk_forward
        smm_module.SMM_QmK_backward_cuda = _smm_qmk_backward
        smm_module.SMM_AmV_forward_cuda = _smm_amv_forward
        smm_module.SMM_AmV_backward_cuda = _smm_amv_backward
        sys.modules["smm_cuda"] = smm_module


def _load_official_pft_class():
    global _PFT_CLASS
    if _PFT_CLASS is not None:
        return _PFT_CLASS

    _ensure_search_paths()
    _ensure_fairscale_stub()
    _ensure_smm_fallback()

    module = importlib.import_module("basicsr.archs.pft_arch")
    _PFT_CLASS = module.PFT
    return _PFT_CLASS


@ARCH_REGISTRY.register()
def pftsr(**kwargs):
    PFT = _load_official_pft_class()
    defaults = {
        "upscale": upscale,
        "in_chans": 3,
        "img_size": 64,
        "embed_dim": 240,
        "depths": [4, 4, 4, 6, 6, 6],
        "num_heads": 6,
        "num_topk": _PFTSR_NUM_TOPK,
        "window_size": 32,
        "convffn_kernel_size": 7,
        "img_range": 1.0,
        "mlp_ratio": 2.0,
        "upsampler": "pixelshuffle",
        "resi_connection": "1conv",
        "use_checkpoint": False,
    }
    defaults.update(kwargs)
    return PFT(**defaults)


@ARCH_REGISTRY.register()
def pftsr_light(**kwargs):
    PFT = _load_official_pft_class()
    defaults = {
        "upscale": upscale,
        "in_chans": 3,
        "img_size": 64,
        "embed_dim": 52,
        "depths": [2, 4, 6, 6, 6],
        "num_heads": 4,
        "num_topk": _PFTSR_LIGHT_NUM_TOPK,
        "window_size": 32,
        "convffn_kernel_size": 7,
        "img_range": 1.0,
        "mlp_ratio": 1.0,
        "upsampler": "pixelshuffledirect",
        "resi_connection": "1conv",
        "use_checkpoint": False,
    }
    defaults.update(kwargs)
    return PFT(**defaults)
