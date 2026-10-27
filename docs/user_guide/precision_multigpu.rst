Mixed precision and multi-GPU
=============================

Every MIR model trains and runs in float32 (the default), float16 or bfloat16 mixed precision, and on several GPUs
with ``DistributedDataParallel`` (DDP) or ``DataParallel``. Float32 behaviour is unchanged: the mixed-precision
support only activates when you ask for it.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Setting
     - How
   * - float32
     - Default; nothing to do.
   * - float16 mixed precision
     - ``torch.autocast("cuda", torch.float16)`` plus a gradient scaler (GPU only).
   * - bfloat16 mixed precision
     - ``torch.autocast(..., torch.bfloat16)`` on GPUs with compute capability 8.0+ (A100, H100, RTX 30xx/40xx) or
       on the CPU; no gradient scaler needed.
   * - float16 inference
     - ``model.half()`` with float16 inputs.
   * - Several GPUs, one process each
     - DDP, launched with ``torchrun`` (recommended).
   * - Several GPUs, one process
     - ``torch.nn.DataParallel``.

:mod:`MIR.utils.precision` and :mod:`MIR.utils.distributed` wrap all of this behind one ``precision`` switch and a
few helpers, and work unchanged in a single process on one GPU or the CPU. The runnable example
``tutorials/multi_gpu_mixed_precision/train_multi_gpu_amp.py`` (synthetic data, no download) shows every option.

Mixed precision
---------------

.. code-block:: python

   from MIR.utils.precision import autocast, grad_scaler

   precision = "fp16"                      # "fp32" | "fp16" | "bf16"
   scaler = grad_scaler(precision)         # active only for fp16 on CUDA; a pass-through otherwise
   for mov, fix in loader:
       optimizer.zero_grad(set_to_none=True)
       with autocast(precision):
           flow = model((mov, fix))
           loss = sim(fix, warp(mov, flow)) + w * reg(flow, None)
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()

What MIR does for you:

- **Losses and regularizers compute in float32.** Every similarity loss and regularizer in
  :mod:`MIR.image_similarity` and :mod:`MIR.deformation_regularizer` casts half-precision inputs to float32, disables
  autocast inside, and returns a float32 value. Local statistics (NCC windows, MI histograms, correlation ratio) are
  not accurate in 16 bits; under bfloat16 autocast they could be off by orders of magnitude. Decorate your own losses
  with :func:`MIR.utils.precision.fp32_forward` to get the same behaviour.
- **Geometry stays in float32.** ``SpatialTransformer``, ``VecInt`` (scaling and squaring), the VFA samplers, the
  LDDMM kernels and the Fourier layers compute sampling coordinates, integration and FFTs in float32 and return the
  input dtype. In float16, a sampling coordinate near voxel 200 would otherwise be off by up to 0.125 voxel.
- **No missing kernels.** PyTorch before 2.1 has no bfloat16 CUDA kernels for 3-D trilinear / nearest upsampling,
  replicate padding and depthwise 3-D convolution, and cuFFT only accepts power-of-two sizes in float16. MIR runs
  exactly those operations in float32 and casts back, so every model works with both precisions.

In tests with every MIR model on 32³ volumes, outputs under float16 autocast differ from float32 by typically
0.1-1% (relative L2), under bfloat16 by 1-6%, which is the expected precision of each format. Prefer float16 (with
the scaler) when the GPU supports both; bfloat16 avoids loss scaling and overflow.

**Inference in float16.** ``model.half()`` with float16 inputs works for every model; warping and integration still
run in float32 internally. For training, use autocast rather than ``model.half()`` so the weights stay float32.

Several GPUs
------------

DistributedDataParallel (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One process per GPU, launched with ``torchrun --nproc_per_node=N train.py`` (or
``python -m torch.distributed.run ...`` for the interpreter MIR is installed in):

.. code-block:: python

   from MIR.utils import distributed as dist_utils

   device = dist_utils.init_distributed()                  # nccl on GPUs; a plain single process without torchrun
   model = dist_utils.wrap_model(model.to(device), device=device)
   sampler = dist_utils.distributed_sampler(train_set)     # None in a single process
   loader = DataLoader(train_set, batch_size=1, sampler=sampler, shuffle=sampler is None)
   for epoch in range(n_epochs):
       if sampler is not None:
           sampler.set_epoch(epoch)
       for mov, fix in loader:
           ...                                              # training step as above
       val_dice = dist_utils.all_reduce_mean(val_dice)     # average metrics over ranks
       if dist_utils.is_main_process():                     # log and save on rank 0 only
           torch.save({"state_dict": dist_utils.state_dict_for_saving(model)}, path)
   dist_utils.cleanup()

``wrap_model`` uses ``find_unused_parameters=True`` by default: some MIR models own parameters that ``forward`` does
not use in every configuration (for example the legacy ``reg_head`` of ``TransMorphTVF``), and plain DDP fails on the
second iteration in that case. Pass ``find_unused_parameters=False`` for a small speed-up when every parameter gets a
gradient, and ``sync_batchnorm=True`` to synchronise BatchNorm statistics.

DataParallel
^^^^^^^^^^^^

``dist_utils.wrap_model(model, mode="dp")`` (or ``torch.nn.DataParallel(model)``) splits each batch across all
visible GPUs from one process. It is simpler than DDP but slower, and:

- scalar outputs, such as the energy terms of the LDDMM models, come back as one value per GPU; take ``.mean()``;
- hypernetwork models (``HyperVFA``, ``HyperVxmDense``, ``HyperTransMorph*``) share one hyperparameter across the
  batch; pass it repeated per sample (``[B, 1]`` instead of ``[1, 1]``) so every GPU receives a copy;
- running statistics updated in ``forward`` are lost on the replicas, so build templates (``TemplateCreation``) with
  DDP, which keeps its running mean synchronised across ranks.

Checkpoints
^^^^^^^^^^^

Wrapped models prefix their ``state_dict`` keys with ``module.``. Save with
:func:`~MIR.utils.distributed.state_dict_for_saving` (or ``unwrap_model(model).state_dict()``) so the checkpoint
loads into a plain model with ``load_state_dict(..., strict=True)``, exactly like the published MIR weights.
:func:`~MIR.utils.distributed.strip_module_prefix` fixes checkpoints already saved from a wrapped model, and
:func:`MIR.utils.load_partial_weights` strips the prefix automatically.

Multi-GPU inference
^^^^^^^^^^^^^^^^^^^

Use ``DataParallel`` for batched inference from one process, or DDP with ``distributed_sampler(test_set,
shuffle=False)`` so each rank processes its own share of the cases and writes its own outputs; gather summary
metrics with ``all_reduce_mean``.
