Conventions
===========

All models, losses and utilities in MIR share a small set of conventions. They are what makes models
interchangeable.

Images
------

Images are PyTorch tensors of shape ``[B, C, D, H, W]``: batch, channel, and three spatial axes.
Registration networks take a tuple ``(moving, fixed)`` of ``[B, 1, D, H, W]`` tensors that live on the
**same grid**. Intensities are usually normalized to ``[0, 1]`` (for example ``img / 255`` for the LUMIR
data, or a clipped HU window for CT).

.. tip::

   Bring both images to a common grid (resampling, cropping, padding) and, for inter-subject
   registration, roughly align them (``MIR.models.AffineReg3D``) before deformable registration.

Displacement fields
-------------------

A displacement field ``u`` has shape ``[B, 3, D, H, W]`` and is expressed in **voxels** of the grid it
lives on. Channel ``i`` is the displacement along spatial axis ``i``. The deformation it represents is
``phi(x) = x + u(x)``.

Warping
-------

``MIR.models.SpatialTransformer`` warps any tensor with a displacement:

.. math::

   \text{warped}(x) = \text{src}\big(x + u(x)\big)

.. code-block:: python

   from MIR.models import SpatialTransformer

   warp = SpatialTransformer((D, H, W)).cuda()                     # images, one-hot labels
   warp_nn = SpatialTransformer((D, H, W), mode='nearest').cuda()  # integer label maps

   warped_image = warp(moving, flow)
   warped_labels = warp_nn(moving_labels.float(), flow)

The transformer samples with ``grid_sample(..., align_corners=True)``, which matches its normalization
of voxel coordinates. Use the same transformer at training and inference time. ``padding_mode`` can be
passed to ``forward`` (``'zeros'`` by default, ``'border'`` for fields).

Composing flows
---------------

Flows compose by warping. To apply ``u_1`` first and ``u_2`` second:

.. math::

   u_{\text{total}} = u_2 + u_1 \circ (\mathrm{id} + u_2)

.. code-block:: python

   flow_total = flow_2 + warp(flow_1, flow_2)

This is how an affine pre-alignment and a deformable flow are chained in
:doc:`../getting_started/brain_registration`.

Changing resolution
-------------------

Several networks (TransMorph-TVF on brain MRI, the PET/CT models) run at half resolution for speed and
memory. Their output lives on the half-resolution grid, in half-resolution voxels. Upsample it and
**multiply by the resolution ratio**:

.. code-block:: python

   import torch.nn.functional as F

   flow_half = model((F.avg_pool3d(moving, 2), F.avg_pool3d(fixed, 2)))
   flow = F.interpolate(flow_half, scale_factor=2, mode='trilinear',
                        align_corners=False) * 2              # half-res voxels -> full-res voxels

Calling conventions
-------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Model type
     - Call and output
   * - Plain models (TransMorph, TransMorph-TVF, VFA, VoxelMorph, LessNet, Fourier-Net)
     - ``flow = model((moving, fixed))``. Models with ``gen_output=True`` return ``(warped, flow)``.
   * - SPR models (``TransMorphTVFSPR``, ``VFASPR``, ``LessNetSPR``, ``FourierNet(Plus)SPR``)
     - ``pos_flow, neg_flow, wts = model((moving, fixed))`` with ``SVF=True``: forward and inverse flows
       and the spatial regularization map. With ``SVF=False``: ``flow, wts``.
   * - Hypernetwork models (``HyperTransMorphTVF``, ``HyperVFA``, ``HyperVxmDense``)
     - ``flow = model((moving, fixed), reg_code)`` with ``reg_code`` a ``[1, 1]`` tensor holding the
       regularization weight.
   * - SITReg
     - ``model(moving, fixed, mappings_for_levels=((0, False),))`` returns mapping pairs; sample the
       displacement with ``DataFormat.voxel_displacements()`` (see :doc:`../models/sitreg`).
   * - LDDMM models (``TransMorphLDDMM``, ``LDDMMWrapper``)
     - ``flow = model((moving, fixed))``, or ``model((moving, fixed), return_dict=True)`` for the inverse
       flow and the energy regularizer (see :doc:`../models/lddmm`).
   * - Optimization-based (``convex_adam_MIND`` and variants)
     - ``flow = convex_adam_MIND(moving, fixed, config)``; no training.

Configurations
--------------

Learned models are configured with ``ml_collections.ConfigDict`` objects from
``MIR.models.configs_<Model>``:

.. code-block:: python

   import MIR.models.configs_TransMorph as configs_TransMorph

   config = configs_TransMorph.get_3DTransMorph3Lvl_config()
   config.img_size = (80, 112, 96)   # the grid the network sees (half of 160 x 224 x 192)
   config.window_size = (2, 3, 3)    # attention window; must divide the feature-map size
   print(config)                     # every field is documented in the config module

Models that integrate their own field accept ``SVF=True/False`` and ``SVF_steps`` in their constructor.
