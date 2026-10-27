"""Multi-GPU helpers for training and inference with ``DistributedDataParallel`` or ``DataParallel``.

Launch with ``torchrun --nproc_per_node=N train.py`` and call :func:`init_distributed` once at start-up. The helpers
also work unchanged in a single process (one GPU or CPU), so the same script runs everywhere.

Checkpoints: wrapped models prefix their ``state_dict`` keys with ``module.``. Save with :func:`state_dict_for_saving`
(or ``unwrap_model(model).state_dict()``) so the file loads into a plain model with ``strict=True``, exactly like the
published MIR weights; :func:`strip_module_prefix` fixes checkpoints that were saved from a wrapped model.

Example::

    from MIR.utils import distributed as dist_utils

    device = dist_utils.init_distributed()                 # torchrun env vars, or a single process
    model = dist_utils.wrap_model(model.to(device), device=device)
    sampler = dist_utils.distributed_sampler(train_set)    # None in a single process
    loader = DataLoader(train_set, batch_size=1, sampler=sampler, shuffle=sampler is None)
    for epoch in range(n_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        ...
        loss_value = dist_utils.all_reduce_mean(loss.detach())
    if dist_utils.is_main_process():
        torch.save({"state_dict": dist_utils.state_dict_for_saving(model)}, path)
"""

import os
from collections import OrderedDict
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn


def is_distributed() -> bool:
    """True when a ``torch.distributed`` process group is initialised with more than one process."""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def get_rank() -> int:
    """Rank of this process (0 when not distributed)."""
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def get_world_size() -> int:
    """Number of processes (1 when not distributed)."""
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def is_main_process() -> bool:
    """True on rank 0 (and in a single process); use it to guard logging and checkpoint saving."""
    return get_rank() == 0


def init_distributed(backend: Optional[str] = None, device_type: Optional[str] = None) -> torch.device:
    """Initialise the default process group from the ``torchrun`` environment and return this process's device.

    Without ``torchrun`` (no ``WORLD_SIZE`` > 1 in the environment) nothing is initialised and the first GPU (or the
    CPU) is returned, so single-process runs need no special casing.

    Args:
        backend: Process-group backend; defaults to ``"nccl"`` on CUDA and ``"gloo"`` on the CPU.
        device_type: ``"cuda"`` or ``"cpu"``; defaults to CUDA when available.

    Returns:
        ``torch.device`` this process should use (``cuda:LOCAL_RANK``, ``cuda:0`` or ``cpu``).
    """
    use_cuda = torch.cuda.is_available() if device_type is None else device_type == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("device_type='cuda' requested but CUDA is not available")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not (dist.is_available() and dist.is_initialized()):
        if use_cuda:
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend or ("nccl" if use_cuda else "gloo"))
    if use_cuda:
        return torch.device("cuda", local_rank if world_size > 1 else torch.cuda.current_device())
    return torch.device("cpu")


def cleanup() -> None:
    """Destroy the process group (no-op when not distributed)."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    """Synchronise all processes (no-op when not distributed)."""
    if is_distributed():
        dist.barrier()


def wrap_model(
    model: nn.Module,
    mode: str = "auto",
    device: Optional[Union[str, torch.device]] = None,
    find_unused_parameters: bool = True,
    sync_batchnorm: bool = False,
    **ddp_kwargs: Any,
) -> nn.Module:
    """Wrap a model for multi-GPU training or inference.

    Args:
        model: The model, already on ``device``.
        mode: ``"ddp"`` (``DistributedDataParallel``, one process per GPU), ``"dp"`` (``DataParallel``, one process,
            all visible GPUs), ``"none"``, or ``"auto"``: DDP when a process group is initialised, DataParallel when
            the model is on a GPU and more than one GPU is visible, otherwise the model unchanged.
        device: This process's device (DDP); defaults to the device of the model's first parameter.
        find_unused_parameters: Passed to DDP. ``True`` by default because some MIR models own parameters that
            their ``forward`` does not use for every configuration (e.g. ``reg_head`` of ``TransMorphTVF``); set
            ``False`` for a small speed-up when every parameter receives a gradient.
        sync_batchnorm: Convert BatchNorm layers to ``SyncBatchNorm`` (DDP only).
        **ddp_kwargs: Further ``DistributedDataParallel`` arguments.

    Returns:
        The wrapped (or unchanged) model. Use :func:`unwrap_model` to reach the original module.
    """
    if mode == "auto":
        param = next(model.parameters(), None)
        on_cuda = param is not None and param.device.type == "cuda"
        mode = "ddp" if is_distributed() else ("dp" if on_cuda and torch.cuda.device_count() > 1 else "none")
    if mode == "none":
        return model
    if mode == "dp":
        return nn.DataParallel(model)
    if mode != "ddp":
        raise ValueError(f"mode must be 'auto', 'ddp', 'dp' or 'none', got {mode!r}")
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("mode='ddp' needs an initialised process group; call init_distributed() under torchrun")
    if sync_batchnorm:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if device is None:
        param = next(model.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")
    device = torch.device(device)
    device_ids = (
        [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else None
    )
    return nn.parallel.DistributedDataParallel(
        model, device_ids=device_ids, find_unused_parameters=find_unused_parameters, **ddp_kwargs
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module of a ``DataParallel`` / ``DistributedDataParallel`` wrapper (or ``model``)."""
    while isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module
    return model


def state_dict_for_saving(model: nn.Module) -> Dict[str, torch.Tensor]:
    """``state_dict`` without wrapper prefixes, loadable into the plain model with ``strict=True``."""
    return unwrap_model(model).state_dict()


def strip_module_prefix(state_dict: Dict[str, Any], prefix: str = "module.") -> Dict[str, Any]:
    """Remove a leading ``"module."`` (as written by wrapped models) from every key of ``state_dict``."""
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return OrderedDict((k[len(prefix) :] if k.startswith(prefix) else k, v) for k, v in state_dict.items())


def all_reduce_mean(value: Union[torch.Tensor, float]) -> Union[torch.Tensor, float]:
    """Average a scalar or tensor over all processes (returned unchanged when not distributed).

    Useful for logging losses and validation metrics that each rank computed on its shard of the data.
    """
    if not is_distributed():
        return value
    is_tensor = isinstance(value, torch.Tensor)
    device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else torch.device("cpu")
    t = (
        value.detach().clone().to(device=device, dtype=torch.float64)
        if is_tensor
        else torch.tensor(float(value), dtype=torch.float64, device=device)
    )
    dist.all_reduce(t)
    t /= get_world_size()
    return t.to(value.dtype).to(value.device) if is_tensor else float(t)


def distributed_sampler(dataset, shuffle: bool = True, seed: int = 0, drop_last: bool = False):
    """``DistributedSampler`` for ``dataset`` when distributed, otherwise ``None``.

    Pass the result as ``sampler=`` to the ``DataLoader`` (with ``shuffle=sampler is None``) and call
    ``sampler.set_epoch(epoch)`` at the start of every epoch.
    """
    if not is_distributed():
        return None
    from torch.utils.data.distributed import DistributedSampler

    return DistributedSampler(dataset, shuffle=shuffle, seed=seed, drop_last=drop_last)
