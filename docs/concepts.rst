Core concepts and conventions
=============================

This page explains the conventions shared by every model in MIR, the transformation models
available (how a network's output becomes a deformation), spatially varying regularization,
and the loss functions and metrics used to train and evaluate registration networks. Read it
once; afterwards every model page and tutorial will look familiar.

Tensors, displacements and warping
----------------------------------

- Images are PyTorch tensors of shape ``[B, C, D, H, W]`` (batch, channel, three spatial
  axes). Registration networks take a tuple ``(moving, fixed)`` of ``[B, 1, D, H, W]``
  tensors on the same grid, usually intensity-normalized to ``[0, 1]``.
- A **displacement field** ``u`` has shape ``[B, 3, D, H, W]`` and is expressed in
  **voxels** of the grid it lives on, channel ``i`` holding the displacement along spatial
  axis ``i``.
- ``MIR.models.SpatialTransformer(size, mode)`` warps any tensor with a displacement:
  ``warped(x) = src(x + u(x))``. ``mode='bilinear'`` for images and one-hot label maps,
  ``mode='nearest'`` for integer label maps. The transformer uses
  ``grid_sample(..., align_corners=True)``; keep training and inference consistent.
- Networks that run at a lower resolution than the images (for example TransMorph at half
  resolution) return a field on their own grid; upsample it to the image grid and multiply
  by the resolution ratio before warping:

  .. code-block:: python

     flow = F.interpolate(flow_half, scale_factor=2, mode='trilinear', align_corners=False) * 2

- Flows compose by warping: ``u_total = u_2 + SpatialTransformer(u_1, u_2)`` applies ``u_1``
  first and ``u_2`` second. This is how affine pre-alignment and deformable flows are chained
  in the brain-registration quick start.

Model calling convention
------------------------

.. code-block:: python

   flow = model((moving, fixed))                 # most models: displacement [B, 3, D, H, W]
   pos_flow, neg_flow, wts = spr_model((moving, fixed))   # SPR models: both directions + weight map
   out = lddmm_model((moving, fixed), return_dict=True)   # LDDMM: dict with flow, inv_flow, energy, ...

Models are configured with ``ml_collections.ConfigDict`` objects returned by
``MIR.models.configs_<Model>.get_*_config()``; set ``config.img_size`` to the grid the
network sees. Models that integrate their own field accept ``SVF=True/False`` and
``SVF_steps``.

Transformation models
---------------------

The network's output can be turned into a deformation in several ways. The choice decides
whether the deformation is diffeomorphic (smooth and invertible), whether an inverse is
available, and how it is regularized.

.. list-table::
   :header-rows: 1
   :widths: 18 34 16 16 16

   * - Model
     - Deformation
     - Diffeomorphic
     - Inverse
     - Where
   * - Displacement
     - the output is the displacement itself
     - no guarantee
     - no
     - all backbones, ``SVF=False``
   * - Stationary velocity field (SVF)
     - the output is a velocity ``v``; ``phi = exp(v)`` by scaling and squaring
       (``MIR.models.VecInt``)
     - yes, for well-behaved ``v``
     - ``exp(-v)``
     - ``SVF=True`` on TransMorph, VFA, LessNet, Fourier-Net, ConvexAdam
   * - Time-varying field (TVF)
     - a sequence of small displacements predicted by per-step heads and composed
     - approximately
     - no
     - ``TransMorphTVF``, ``TransMorphTVFSPR``
   * - LDDMM, geodesic shooting
     - the output is the initial momentum ``m_0``; the flow follows the geodesic of the
       kernel metric
     - yes
     - yes, integrated alongside
     - ``TransMorphLDDMM``, ``LDDMMWrapper`` (``param='shooting'``)
   * - LDDMM, relaxation
     - one velocity per time step, smoothed by the kernel and composed
     - yes
     - yes
     - ``TransMorphLDDMM``, ``LDDMMWrapper`` (``param='tvf'``)
   * - Symmetric, inverse consistent
     - SITReg's multi-resolution construction with deformation inversion layers
     - yes
     - yes
     - ``SITReg``

Stationary velocity fields are the workhorse: cheap, diffeomorphic in practice, with an
inverse obtained by negating the velocity. LDDMM goes further: the deformation is the end
point of a path whose kinetic energy is the regularizer, the inverse map is exact, and the
path itself is a geodesic that can be interpolated, used for statistics on deformations or
for atlas building. See :doc:`models/lddmm` for the mathematics and the model-agnostic
wrapper.

Spatially varying regularization (SPR)
--------------------------------------

Uniform regularization over-smooths where anatomy slides or changes abruptly and
under-smooths elsewhere. SPR models predict a per-voxel regularization-strength map
``lam`` in ``(0, 1]`` alongside the flow. Two mechanisms are available:

- **Loss-side SPR** (``*SPR`` models: ``TransMorphTVFSPR``, ``VFASPR``, ``LessNetSPR``,
  ``FourierNetSPR``, SITReg + ``SPRHead``): the model returns ``(pos_flow, neg_flow, lam)``;
  ``lam`` weights the local diffusion penalty ``MIR.deformation_regularizer.LocalGrad3d``
  and is pulled toward 1 by the Beta prior ``logBeta``. The map never changes the warp
  itself, only the training signal.
- **Metric-side SPR** (LDDMM models with ``spr='kernel'``): ``lam`` sets the per-voxel scale
  of the kernel that defines the velocity space, so the regularization strength is part of
  the metric and travels with the flow. ``spr='energy'`` is the literal LDDMM transcription
  of the loss-side variant.

Hypernetwork models (``HyperTransMorphTVF``, ``HyperVFA``, ``HyperVxmDense``) take the
global regularization weight as an input instead, so one network covers a range of
regularization strengths at inference time.

Losses and regularizers
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Group
     - Classes and functions (``MIR.image_similarity``, ``MIR.deformation_regularizer``)
   * - Intensity similarity
     - ``NCC_vxm``, ``NCC_gauss``, ``NCC_mok``, ``NCC_vfa``, ``FastNCC``, ``NCC_fp16``,
       ``SSIM3D``, ``PCC``, ``NormalizedGradientFieldLoss``
   * - Multi-modal similarity
     - ``MIND_loss``, ``MutualInformation``, ``localMutualInformation``,
       ``MattesMutualInformation``, ``NormalizedMutualInformation``, ``CorrRatio``,
       ``LocalCorrRatio``
   * - Label overlap
     - ``DiceLoss``, ``SparseDiceLoss``, ``sparse_dice_from_int_labels`` (memory-efficient
       Dice on a sampled subset of labels), ``DiceSegLoss``, ``SegmentationLoss``
   * - Global regularizers
     - ``Grad3d`` / ``Grad2D`` (diffusion or TV on the flow), ``Grad3DiTV``,
       ``GradICON3d``, ``GradICONExact3d``, ``DisplacementRegularizer``
   * - Spatially varying regularizers
     - ``LocalGrad3d`` (weighted diffusion), ``logBeta`` and ``logGaussian`` priors on the
       weight map
   * - LDDMM
     - the kinetic energy returned by the model (``out['energy']``), optional
       optimal-mass-transport penalty on kernel weights (``out['omt']``)
   * - Probabilistic
     - ``KL_divergence``, ``MultiVariateKL_divergence``

Evaluation metrics
------------------

``MIR.accuracy_measures`` provides label-overlap and deformation-quality metrics:

- ``dice_val_VOI``, ``dice_val_all``, ``dice_val_substruct``: Dice over label sets.
- ``calc_jac_dets``, ``calc_J_i``, ``calc_Jstar_1``, ``calc_Jstar_2``,
  ``calc_measurements``: finite-difference and *digital-diffeomorphism* Jacobian
  determinants and the non-diffeomorphic volume (NDV) of Liu et al., IJCV 2024.
- ``MIR.models.lddmm.jacobian_fd`` / ``det3x3``: differentiable Jacobian of a displacement
  field on the GPU, used by the LDDMM models and their diagnostics.

Data utilities
--------------

- ``MIR.utils.SpatialTransformer``, ``VecInt``, ``mk_grid_img`` (grid images for deformation
  figures), ``Logger``, ``AverageMeter``, ``load_partial_weights`` (load matching tensors
  from a checkpoint), patch samplers and sliding-window inference.
- ``MIR.data_augmentation``: ``RandomMultiContrastRemap`` (random intensity remapping for
  contrast-agnostic training), flips and affine augmentation.
- ``MIR.intensity_normalization``, ``MIR.random_image_generation`` (random shapes for
  pretraining), ``MIR.l2r_io`` (Learn2Reg I/O), ``MIR.label_reference``.
