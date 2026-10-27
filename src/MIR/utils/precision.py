"""Mixed-precision helpers: one ``precision`` switch for training and inference.

MIR models run in float32 (default), float16 or bfloat16 mixed precision (``torch.autocast``) and in pure float16
inference (``model.half()``). Losses and regularizers always compute in float32 and return a float32 value; spatial
transformers, integrators and Fourier kernels compute their geometry in float32 and return the input dtype.

Example::

    from MIR.utils.precision import autocast, grad_scaler

    scaler = grad_scaler(precision)                    # active only for fp16 on CUDA
    with autocast(precision):
        flow = model((mov, fix))
        loss = sim(warped, fix) + reg(flow)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
"""

import contextlib
from typing import ContextManager

import torch

from MIR._amp import fp32_forward  # noqa: F401 -- public: decorate custom losses so they compute in float32

PRECISIONS = ("fp32", "fp16", "bf16")
_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def precision_dtype(precision: str) -> torch.dtype:
    """Return the torch dtype for ``"fp32"``, ``"fp16"`` or ``"bf16"``.

    Raises:
        ValueError: For any other value.
    """
    if precision not in _DTYPES:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    return _DTYPES[precision]


def check_precision(precision: str, device_type: str = "cuda") -> None:
    """Raise a clear error when ``precision`` is not usable on ``device_type``.

    float16 autocast needs CUDA; bfloat16 needs a GPU with native bf16 support (compute capability 8.0+) or the CPU.
    """
    precision_dtype(precision)
    if precision == "fp16" and device_type != "cuda":
        raise ValueError("fp16 mixed precision requires CUDA; use 'bf16' or 'fp32' on the CPU")
    if (
        precision == "bf16"
        and device_type == "cuda"
        and torch.cuda.is_available()
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("this GPU does not support bfloat16; use 'fp16' or 'fp32'")


def autocast(precision: str = "fp32", device_type: str = "cuda") -> ContextManager:
    """Context manager for the requested precision (a no-op for ``"fp32"``).

    Args:
        precision: ``"fp32"``, ``"fp16"`` or ``"bf16"``.
        device_type: ``"cuda"`` or ``"cpu"``.

    Returns:
        ``torch.autocast(device_type, dtype)`` for fp16/bf16, otherwise a null context.
    """
    if precision == "fp32":
        precision_dtype(precision)
        return contextlib.nullcontext()
    check_precision(precision, device_type)
    return torch.autocast(device_type=device_type, dtype=precision_dtype(precision))


def grad_scaler(precision: str = "fp32", device_type: str = "cuda") -> "torch.cuda.amp.GradScaler":
    """Gradient scaler for the requested precision.

    Loss scaling is needed only for float16; for fp32 and bf16 the returned scaler is disabled and its
    ``scale`` / ``step`` / ``update`` calls are pass-throughs, so the same training loop works for every precision.
    """
    precision_dtype(precision)
    return torch.cuda.amp.GradScaler(enabled=(precision == "fp16" and device_type == "cuda"))
