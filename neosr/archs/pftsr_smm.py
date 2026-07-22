from __future__ import annotations

import importlib
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable

if TYPE_CHECKING:
    from types import ModuleType

SMMBackend = Literal["torch", "cuda"]

DEFAULT_SMM_CHUNK_SIZE = 64
_CUDA_SYMBOLS = ("SMM_QmK_forward_cuda", "SMM_AmV_forward_cuda")


def _validate_chunk_size(chunk_size: int) -> None:
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        msg = "PFT-SR SMM chunk size must be an integer."
        raise TypeError(msg)
    if chunk_size < 1:
        msg = "PFT-SR SMM chunk size must be greater than zero."
        raise ValueError(msg)


@lru_cache(maxsize=1)
def _cuda_extension() -> ModuleType:
    try:
        module = importlib.import_module("smm_cuda")
    except (ImportError, OSError) as exc:
        msg = (
            "The 'pftsr_cuda' and 'pftsr_light_cuda' architectures require the "
            "optional smm_cuda extension. Install a compatible build, or use "
            "'pftsr'/'pftsr_light' for the self-contained PyTorch backend."
        )
        raise ModuleNotFoundError(msg, name="smm_cuda") from exc

    missing = [
        name for name in _CUDA_SYMBOLS if not callable(getattr(module, name, None))
    ]
    if missing:
        joined = ", ".join(missing)
        msg = f"The installed smm_cuda extension is missing required symbols: {joined}."
        raise ImportError(msg)
    return module


def ensure_smm_backend(backend: SMMBackend, chunk_size: int) -> None:
    """Validate a PFT-SR sparse-attention backend at model construction time."""
    _validate_chunk_size(chunk_size)
    if backend == "cuda":
        _cuda_extension()
    elif backend != "torch":
        msg = f"Unknown PFT-SR SMM backend: {backend!r}."
        raise ValueError(msg)


def _validate_common(a: Tensor, b: Tensor, index: Tensor) -> None:
    if a.device != b.device or a.device != index.device:
        msg = "PFT-SR SMM inputs must be on the same device."
        raise ValueError(msg)
    if not a.is_floating_point() or not b.is_floating_point():
        msg = "PFT-SR SMM inputs must be floating-point tensors."
        raise TypeError(msg)
    if index.dtype not in (torch.int32, torch.int64):
        msg = "PFT-SR SMM indices must use int32 or int64."
        raise TypeError(msg)


def _validate_qmk(a: Tensor, b: Tensor, index: Tensor) -> None:
    _validate_common(a, b, index)
    if a.ndim != 3 or b.ndim != 3 or index.ndim != 3:
        msg = "QmK expects three-dimensional A, B, and index tensors."
        raise ValueError(msg)
    if a.shape[0] != b.shape[0] or a.shape[0] != index.shape[0]:
        msg = "QmK batch dimensions do not match."
        raise ValueError(msg)
    if a.shape[1] != index.shape[1] or a.shape[2] != b.shape[1]:
        msg = "QmK tensor dimensions do not match."
        raise ValueError(msg)


def _validate_amv(a: Tensor, b: Tensor, index: Tensor) -> None:
    _validate_common(a, b, index)
    if a.ndim != 3 or b.ndim != 3 or index.ndim != 3:
        msg = "AmV expects three-dimensional A, B, and index tensors."
        raise ValueError(msg)
    if a.shape != index.shape or a.shape[0] != b.shape[0]:
        msg = "AmV attention and index dimensions do not match."
        raise ValueError(msg)


def _output_dtype(a: Tensor, b: Tensor) -> torch.dtype:
    return torch.promote_types(a.dtype, b.dtype)


def _accumulation_dtype(a: Tensor, b: Tensor) -> torch.dtype:
    dtype = _output_dtype(a, b)
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _qmk_forward_torch(a: Tensor, b: Tensor, index: Tensor, chunk_size: int) -> Tensor:
    batch, query_count, feature_dim = a.shape
    topk = index.shape[-1]
    output_dtype = _output_dtype(a, b)
    output = torch.empty(
        (batch, query_count, topk), device=a.device, dtype=output_dtype
    )
    expanded_b = b.to(output_dtype).unsqueeze(1)

    for start in range(0, query_count, chunk_size):
        end = min(start + chunk_size, query_count)
        a_chunk = a[:, start:end].to(output_dtype)
        index_chunk = index[:, start:end].long()
        gather_index = index_chunk.unsqueeze(2).expand(-1, -1, feature_dim, -1)
        selected_b = torch.gather(
            expanded_b.expand(-1, end - start, -1, -1), 3, gather_index
        )
        output[:, start:end] = torch.einsum("bnd,bndk->bnk", a_chunk, selected_b)

    return output.contiguous()


def _qmk_backward_torch(
    grad_output: Tensor, a: Tensor, b: Tensor, index: Tensor, chunk_size: int
) -> tuple[Tensor, Tensor]:
    batch, query_count, feature_dim = a.shape
    accumulation_dtype = _accumulation_dtype(a, b)
    grad_a = torch.empty_like(a)
    grad_b_acc = torch.zeros(b.shape, device=b.device, dtype=accumulation_dtype)
    b_acc = b.to(accumulation_dtype).unsqueeze(1)

    for start in range(0, query_count, chunk_size):
        end = min(start + chunk_size, query_count)
        chunk_count = end - start
        a_chunk = a[:, start:end].to(accumulation_dtype)
        grad_chunk = grad_output[:, start:end].to(accumulation_dtype)
        index_chunk = index[:, start:end].long()
        gather_index = index_chunk.unsqueeze(2).expand(-1, -1, feature_dim, -1)
        selected_b = torch.gather(
            b_acc.expand(-1, chunk_count, -1, -1), 3, gather_index
        )
        grad_a[:, start:end] = torch.einsum("bnk,bndk->bnd", grad_chunk, selected_b).to(
            a.dtype
        )

        contribution = (
            a_chunk
            .unsqueeze(-1)
            .mul(grad_chunk.unsqueeze(2))
            .permute(0, 2, 1, 3)
            .reshape(batch, feature_dim, -1)
        )
        scatter_index = (
            index_chunk
            .unsqueeze(1)
            .expand(-1, feature_dim, -1, -1)
            .reshape(batch, feature_dim, -1)
        )
        grad_b_acc.scatter_add_(2, scatter_index, contribution)

    return grad_a.contiguous(), grad_b_acc.to(b.dtype).contiguous()


def _amv_forward_torch(a: Tensor, b: Tensor, index: Tensor, chunk_size: int) -> Tensor:
    batch, query_count, _ = a.shape
    value_dim = b.shape[-1]
    output_dtype = _output_dtype(a, b)
    output = torch.empty(
        (batch, query_count, value_dim), device=a.device, dtype=output_dtype
    )
    expanded_b = b.to(output_dtype).unsqueeze(1)

    for start in range(0, query_count, chunk_size):
        end = min(start + chunk_size, query_count)
        a_chunk = a[:, start:end].to(output_dtype)
        index_chunk = index[:, start:end].long()
        gather_index = index_chunk.unsqueeze(-1).expand(-1, -1, -1, value_dim)
        selected_b = torch.gather(
            expanded_b.expand(-1, end - start, -1, -1), 2, gather_index
        )
        output[:, start:end] = torch.einsum("bnk,bnkd->bnd", a_chunk, selected_b)

    return output.contiguous()


def _amv_backward_torch(
    grad_output: Tensor, a: Tensor, b: Tensor, index: Tensor, chunk_size: int
) -> tuple[Tensor, Tensor]:
    batch, query_count, _ = a.shape
    value_dim = b.shape[-1]
    accumulation_dtype = _accumulation_dtype(a, b)
    grad_a = torch.empty_like(a)
    grad_b_acc = torch.zeros(b.shape, device=b.device, dtype=accumulation_dtype)
    b_acc = b.to(accumulation_dtype).unsqueeze(1)

    for start in range(0, query_count, chunk_size):
        end = min(start + chunk_size, query_count)
        chunk_count = end - start
        a_chunk = a[:, start:end].to(accumulation_dtype)
        grad_chunk = grad_output[:, start:end].to(accumulation_dtype)
        index_chunk = index[:, start:end].long()
        gather_index = index_chunk.unsqueeze(-1).expand(-1, -1, -1, value_dim)
        selected_b = torch.gather(
            b_acc.expand(-1, chunk_count, -1, -1), 2, gather_index
        )
        grad_a[:, start:end] = torch.einsum("bnd,bnkd->bnk", grad_chunk, selected_b).to(
            a.dtype
        )

        contribution = (
            a_chunk
            .unsqueeze(-1)
            .mul(grad_chunk.unsqueeze(2))
            .reshape(batch, -1, value_dim)
        )
        scatter_index = index_chunk.reshape(batch, -1, 1).expand(-1, -1, value_dim)
        grad_b_acc.scatter_add_(1, scatter_index, contribution)

    return grad_a.contiguous(), grad_b_acc.to(b.dtype).contiguous()


class _TorchQmK(Function):
    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor, index: Tensor, chunk_size: int) -> Tensor:
        _validate_qmk(a, b, index)
        _validate_chunk_size(chunk_size)
        ctx.save_for_backward(a, b, index)
        ctx.chunk_size = chunk_size
        return _qmk_forward_torch(a, b, index, chunk_size)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        a, b, index = ctx.saved_tensors
        grad_a, grad_b = _qmk_backward_torch(grad_output, a, b, index, ctx.chunk_size)
        return grad_a, grad_b, None, None


class _TorchAmV(Function):
    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor, index: Tensor, chunk_size: int) -> Tensor:
        _validate_amv(a, b, index)
        _validate_chunk_size(chunk_size)
        ctx.save_for_backward(a, b, index)
        ctx.chunk_size = chunk_size
        return _amv_forward_torch(a, b, index, chunk_size)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        a, b, index = ctx.saved_tensors
        grad_a, grad_b = _amv_backward_torch(grad_output, a, b, index, ctx.chunk_size)
        return grad_a, grad_b, None, None


class _CudaQmK(Function):
    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor, index: Tensor, chunk_size: int) -> Tensor:
        _validate_qmk(a, b, index)
        _validate_chunk_size(chunk_size)
        if a.device.type != "cuda":
            msg = "The pftsr_cuda backend requires CUDA tensors."
            raise RuntimeError(msg)

        ctx.save_for_backward(a, b, index)
        ctx.chunk_size = chunk_size
        function = _cuda_extension().SMM_QmK_forward_cuda
        output = function(
            a.float().contiguous(), b.float().contiguous(), index.int().contiguous()
        )
        return output.to(_output_dtype(a, b))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        a, b, index = ctx.saved_tensors
        grad_a, grad_b = _qmk_backward_torch(grad_output, a, b, index, ctx.chunk_size)
        return grad_a, grad_b, None, None


class _CudaAmV(Function):
    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor, index: Tensor, chunk_size: int) -> Tensor:
        _validate_amv(a, b, index)
        _validate_chunk_size(chunk_size)
        if a.device.type != "cuda":
            msg = "The pftsr_cuda backend requires CUDA tensors."
            raise RuntimeError(msg)

        ctx.save_for_backward(a, b, index)
        ctx.chunk_size = chunk_size
        function = _cuda_extension().SMM_AmV_forward_cuda
        output = function(
            a.float().contiguous(), b.float().contiguous(), index.int().contiguous()
        )
        return output.to(_output_dtype(a, b))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        a, b, index = ctx.saved_tensors
        grad_a, grad_b = _amv_backward_torch(grad_output, a, b, index, ctx.chunk_size)
        return grad_a, grad_b, None, None


def sparse_qmk(
    a: Tensor,
    b: Tensor,
    index: Tensor,
    backend: SMMBackend,
    chunk_size: int = DEFAULT_SMM_CHUNK_SIZE,
) -> Tensor:
    """Compute selected Q @ K entries with a bounded-memory backward pass."""
    if backend == "torch":
        return _TorchQmK.apply(a, b, index, chunk_size)
    if backend == "cuda":
        return _CudaQmK.apply(a, b, index, chunk_size)
    msg = f"Unknown PFT-SR SMM backend: {backend!r}."
    raise ValueError(msg)


def sparse_amv(
    a: Tensor,
    b: Tensor,
    index: Tensor,
    backend: SMMBackend,
    chunk_size: int = DEFAULT_SMM_CHUNK_SIZE,
) -> Tensor:
    """Multiply sparse attention values by V with bounded temporary memory."""
    if backend == "torch":
        return _TorchAmV.apply(a, b, index, chunk_size)
    if backend == "cuda":
        return _CudaAmV.apply(a, b, index, chunk_size)
    msg = f"Unknown PFT-SR SMM backend: {backend!r}."
    raise ValueError(msg)


__all__ = [
    "DEFAULT_SMM_CHUNK_SIZE",
    "SMMBackend",
    "ensure_smm_backend",
    "sparse_amv",
    "sparse_qmk",
]
