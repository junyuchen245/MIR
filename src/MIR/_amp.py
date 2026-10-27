"""Low-level mixed-precision helpers used inside MIR (torch-only, no MIR imports).

Public, user-facing helpers live in :mod:`MIR.utils.precision`; this module holds the pieces the models and losses
import directly:

- :func:`fp32_forward` -- decorator that runs a loss / regularizer ``forward`` in float32 regardless of the input
  dtype or an active ``torch.autocast`` region (no-op for float32 inputs).
- :func:`interpolate` and :class:`Upsample` -- ``F.interpolate`` / ``nn.Upsample`` that fall back to float32 for the
  dtype/device combinations PyTorch has no kernel for (bfloat16 3-D upsampling on CUDA before torch 2.1, float16 on
  CPU), and return the input dtype.
- :func:`pad` -- ``F.pad`` with the same fallback for ``replicate`` / ``reflect`` padding.
- :class:`Conv3d` -- ``nn.Conv3d`` whose grouped/depthwise case runs in float32 under bfloat16, for the same reason.
"""

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cast_floats(obj, dtype=torch.float32):
    """Cast floating-point tensors with fewer than 32 bits (fp16/bf16) to ``dtype``; leave everything else."""
    if isinstance(obj, torch.Tensor):
        return obj.to(dtype) if obj.dtype in (torch.float16, torch.bfloat16) else obj
    if isinstance(obj, tuple):
        return tuple(_cast_floats(v, dtype) for v in obj)
    if isinstance(obj, list):
        return [_cast_floats(v, dtype) for v in obj]
    if isinstance(obj, dict):
        return {k: _cast_floats(v, dtype) for k, v in obj.items()}
    return obj


def _device_type(objs) -> str:
    for o in objs:
        if isinstance(o, torch.Tensor):
            return o.device.type
        if isinstance(o, (list, tuple)):
            t = _device_type(o)
            if t:
                return t
    return ""


def fp32_forward(fn):
    """Decorate a ``forward`` so that it computes in float32.

    Half-precision (fp16 / bf16) tensor arguments are cast to float32 and ``torch.autocast`` is disabled inside the
    call, so similarity losses and regularizers keep full precision under mixed-precision training and return a
    float32 value. For float32 inputs nothing changes (the casts are no-ops).
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        args = _cast_floats(args)
        kwargs = _cast_floats(kwargs)
        device_type = _device_type(list(args) + list(kwargs.values()))
        if device_type in ("cuda", "cpu"):
            with torch.autocast(device_type=device_type, enabled=False):
                return fn(self, *args, **kwargs)
        return fn(self, *args, **kwargs)

    return wrapper


def _no_lowp_kernel(x: torch.Tensor) -> bool:
    """True when 3-D (trilinear/nearest) upsampling has no native kernel for this dtype/device."""
    return x.dtype == torch.bfloat16 or (x.dtype == torch.float16 and x.device.type == "cpu")


def interpolate(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """``torch.nn.functional.interpolate`` that also works for bf16 (CUDA) and fp16 (CPU) inputs.

    Those inputs are resampled in float32 and cast back; every other input is passed through unchanged.
    """
    if _no_lowp_kernel(x):
        return F.interpolate(x.float(), *args, **kwargs).to(x.dtype)
    return F.interpolate(x, *args, **kwargs)


def pad(x: torch.Tensor, pad: tuple, mode: str = "constant", value: float = 0.0) -> torch.Tensor:
    """``torch.nn.functional.pad`` that also works for bf16 inputs with ``mode="replicate"`` / ``"reflect"``.

    Those modes have no bf16 CUDA kernel for 3-D inputs before torch 2.1; such inputs are padded in float32 and cast
    back. Every other call is passed through unchanged.
    """
    if mode != "constant" and _no_lowp_kernel(x):
        return F.pad(x.float(), pad, mode=mode).to(x.dtype)
    if mode == "constant":
        return F.pad(x, pad, mode=mode, value=value)
    return F.pad(x, pad, mode=mode)


class Upsample(nn.Upsample):
    """``torch.nn.Upsample`` with the same low-precision fallback as :func:`interpolate` (no parameters)."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: D102
        if _no_lowp_kernel(input):
            return super().forward(input.float()).to(input.dtype)
        return super().forward(input)


def _bf16_autocast_active(x: torch.Tensor) -> bool:
    if x.dtype == torch.bfloat16:
        return True
    if x.device.type == "cuda":
        return torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.bfloat16
    return torch.is_autocast_cpu_enabled() and torch.get_autocast_cpu_dtype() == torch.bfloat16


class Conv3d(nn.Conv3d):
    """``torch.nn.Conv3d`` whose grouped / depthwise case runs in float32 under bfloat16.

    Depthwise 3-D convolution has no bf16 CUDA kernel before torch 2.1. Parameters and ``state_dict`` keys are the same
    as for ``nn.Conv3d``.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: D102
        if self.groups > 1 and _bf16_autocast_active(input):
            with torch.autocast(device_type=input.device.type, enabled=False):
                return F.conv3d(
                    input.float(),
                    self.weight.float(),
                    None if self.bias is None else self.bias.float(),
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                ).to(torch.bfloat16)
        return super().forward(input)


def shared_hyperparameter(h: torch.Tensor) -> torch.Tensor:
    """Return the single hyperparameter row of ``h`` (``[1, k]``), accepting one identical row per sample.

    Hypernetwork models generate one set of weights shared by the whole batch, so they need a ``[1, k]``
    hyperparameter. ``nn.DataParallel`` splits every input along dim 0, so for data-parallel use pass the value
    repeated per sample (``[B, k]``); all rows must then be equal.
    """
    if h.dim() >= 1 and h.shape[0] > 1:
        if not torch.all(h == h[:1]):
            raise ValueError(
                "hypernetwork models share one hyperparameter across the batch; got different values per sample "
                f"(shape {tuple(h.shape)}). Pass a [1, k] tensor or identical rows."
            )
        return h[:1]
    return h
