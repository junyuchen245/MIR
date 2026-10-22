TransMorph
==========

Overview
--------

TransMorph is the reference architecture of the toolkit: a Swin-Transformer encoder on the
concatenated moving/fixed pair, a convolutional decoder with skip connections, and a
registration head that predicts a dense field. It is accurate on brain MRI and whole-body
PET/CT, trains with any similarity loss in MIR, and is the backbone of the spatially varying
regularization, hypernetwork and LDDMM variants.

Variants
--------

.. list-table::
   :header-rows: 1
   :widths: 26 50 24

   * - Class
     - Description
     - Output
   * - ``TransMorph``
     - Original model; ``SVF=True`` integrates the output as a stationary velocity field
       (``SVF_steps`` scaling-and-squaring steps).
     - ``flow``
   * - ``TransMorphTVF``
     - Time-varying field: ``time_steps`` per-step heads each predict a small displacement
       from the currently warped moving image; the displacements are composed
       (``composition='composition'``) or added. The LUMIR24/25 brain models use this
       variant with ``time_steps=7``.
     - ``flow``
   * - ``TransMorphTVFSPR``
     - TransMorph-TVF with a spatially varying regularization head: returns the per-voxel
       weight map used by ``LocalGrad3d`` / ``logBeta``; ``SVF=True`` also returns the
       inverse flow.
     - ``(flow, wts)`` or ``(pos_flow, neg_flow, wts)``
   * - ``TransMorphTVFAutoPETAtlas``
     - PET/CT atlas-registration variant integrated at twice the network resolution.
     - ``(pos_flow, neg_flow)``
   * - ``TransMorphAffine``
     - Affine registration head on the TransMorph encoder.
     - affine parameters
   * - ``swin_type='dswin' / 'dswinv2'``
     - Deformable cross-attention transformer (TransMorph-DCA) encoders.
     - as above
   * - ``TransMorphLDDMM``
     - TransMorph with the LDDMM transformation model (:doc:`lddmm`).
     - ``flow``

Usage
-----

.. code-block:: python

   import torch.nn.functional as F
   from MIR.models import TransMorphTVF, SpatialTransformer
   import MIR.models.configs_TransMorph as configs_TransMorph

   H, W, D = 160, 224, 192                       # LUMIR image size; the network runs at half resolution
   config = configs_TransMorph.get_3DTransMorph3Lvl_config()
   config.img_size = (H // 2, W // 2, D // 2)
   config.window_size = (H // 64, W // 64, D // 64)
   model = TransMorphTVF(config, time_steps=7).cuda()

   flow_half = model((F.avg_pool3d(mov, 2), F.avg_pool3d(fix, 2)))
   flow = F.interpolate(flow_half, scale_factor=2, mode='trilinear', align_corners=False) * 2
   warped = SpatialTransformer((H, W, D)).cuda()(mov, flow)

Configurations live in ``MIR.models.configs_TransMorph`` (``get_3DTransMorph_config``,
``get_3DTransMorph3Lvl_config``, ``get_3DTransMorphAtlas3Lvl_config``, large / small / tiny
presets, ablations of the skip connections and positional embeddings). ``config.img_size`` must
be divisible by ``patch_size * 2 ** (levels - 1)`` and ``window_size`` by the feature-map size.

Pretrained weights
------------------

``ModelWeights['TransMorphTVF-LUMIR24-MonoModal']`` (brain, LUMIR24),
``ModelWeights['TransMorph-IXI-Atlas2Subject']``, ``ModelWeights['TransMorphTVF-IXI-Atlas2Subject']``,
``ModelWeights['MedIA-TM-SPR-Beta-autoPET']`` and ``['MedIA-TM-SPR-Gaussian-autoPET']``
(TransMorph-TVF-SPR, PET/CT).

Key modules
-----------

- ``MIR.models.TransMorph``, ``MIR.models.configs_TransMorph``
- ``MIR.models.Swin_Transformer``, ``MIR.models.Deformable_Swin_Transformer``,
  ``MIR.models.Deformable_Swin_Transformer_v2``
- ``MIR.models.HyperTransMorph`` (:doc:`hypertransmorph`)

Related tutorials
-----------------

- `LUMIR25 TransMorph (training template) <https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir25_TransMorph>`_
- `LUMIR24 TransMorph <https://github.com/junyuchen245/MIR/tree/main/tutorials/lumir24_TransMorph>`_
- `IXI benchmarking <https://github.com/junyuchen245/MIR/tree/main/tutorials/IXI_benchmarking>`_
- `IXI TransMorph-SPR <https://github.com/junyuchen245/MIR/tree/main/tutorials/IXI_TransMorphSPR>`_
- `autoPET TransMorph-SPR <https://github.com/junyuchen245/MIR/tree/main/tutorials/autoPET_TransMorphSPR>`_
- `Brain template construction <https://github.com/junyuchen245/MIR/tree/main/tutorials/build_brain_template>`_

References
----------

- Chen et al. TransMorph: Transformer for unsupervised medical image registration.
  *Medical Image Analysis* 2022. `arXiv <https://arxiv.org/abs/2111.10480>`_
- Chen et al. Unsupervised learning of diffeomorphic image registration via TransMorph.
  *WBIR* 2022. (TransMorph-TVF)
- Chen et al. Deformable cross-attention transformer for medical image registration.
  *MLMI, MICCAI* 2023. (TransMorph-DCA)
- Chen et al. Unsupervised learning of spatially varying regularization for diffeomorphic
  image registration. *Medical Image Analysis* 2025. (TransMorph-TVF-SPR)
